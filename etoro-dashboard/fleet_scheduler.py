"""Walk-forward scheduler — weekly re-optimization with a stability gate.

The regime lesson, turned into protocol: optimal exit parameters flip between
windows, so chasing every fresh optimum is curve-fitting with extra steps.
Once a week this scheduler re-runs the fleet sweep and then applies the new
per-plan best exits ONLY where they are STABLE:

  • bot already has overrides and the new optimum is within one grid step
    (|Δstop| ≤ 0.5, |Δtrail| ≤ 0.5, take-profit in the same bucket)  → apply
  • bot has no overrides yet and the new row passes the OOS gate
    (PF ≥ 1.0, n ≥ 5)                                                → apply
  • new optimum jumped across the grid                               → HOLD the
    old params and record the plan as unstable for the user to judge

It never promotes or demotes bots — fleet membership stays a human decision.
Report: /app/data/walk_forward.json (surfaced in the fleet section).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path(os.environ.get("WALK_FORWARD_PATH", "/app/data/walk_forward.json"))
FLEET_PATH = Path(os.environ.get("FLEET_OPT_PATH", "/app/data/fleet_opt.json"))
PERIOD_SEC = 7 * 86400          # weekly
CHECK_EVERY_SEC = 3600.0        # poll cadence

_thread = None
_started = False


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        tmp = STATE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_PATH)
    except Exception:
        log.warning("walk-forward state save failed", exc_info=True)


def _tp_same_bucket(a: float, b: float) -> bool:
    if (a or 0) == 0 and (b or 0) == 0:
        return True
    return (a or 0) > 0 and (b or 0) > 0 and abs(a - b) <= 0.6


def apply_with_stability_gate() -> dict:
    """Map the saved fleet table onto ON bots' overrides, stability-gated.
    Returns the report dict (also persisted into the state file)."""
    import instrument_config
    import strategies as strategies_mod
    import user_settings

    table = json.loads(FLEET_PATH.read_text(encoding="utf-8"))["rows"]
    names = strategies_mod.display_names()
    disabled_path = Path(os.environ.get("DISABLED_BOTS_PATH", "/app/data/disabled_bots.json"))
    try:
        disabled = set(json.loads(disabled_path.read_text(encoding="utf-8")))
    except Exception:
        disabled = set()

    cfg = user_settings.load()
    overrides = dict(cfg.get("bot_overrides") or {})
    applied, held, skipped = [], [], []

    def row_for(spec):
        sd, asset = names.get(spec.strategy, spec.strategy), spec.label.split()[0]
        for r in table:
            if (r.get("Status") == "ok" and r["Strategy"] == sd
                    and r["Asset"] == asset and r["Interval"] == spec.interval):
                return r
        return None

    for spec in instrument_config.load_specs():
        if spec.key in disabled:
            continue
        r = row_for(spec)
        if r is None:
            skipped.append((spec.key, "no qualified row"))
            continue
        if not (float(r["OOS PF"]) >= 1.0 and int(r["OOS n"]) >= 5):
            skipped.append((spec.key, f"weak OOS pf={r['OOS PF']} n={r['OOS n']}"))
            continue
        new = {
            "atr_stop_mult": float(r["Stop ×ATR"]),
            "atr_trail_mult": float(r["Trail ×ATR"]),
            "take_profit_pct": float(r["TP %"]),
        }
        old = overrides.get(spec.key) or {}
        has_old = "atr_stop_mult" in old or "atr_trail_mult" in old
        stable = (
            not has_old
            or (
                abs(float(old.get("atr_stop_mult", new["atr_stop_mult"])) - new["atr_stop_mult"]) <= 0.5
                and abs(float(old.get("atr_trail_mult", new["atr_trail_mult"])) - new["atr_trail_mult"]) <= 0.5
                and _tp_same_bucket(float(old.get("take_profit_pct", new["take_profit_pct"])),
                                    new["take_profit_pct"])
            )
        )
        if stable:
            merged = dict(old)
            merged.update(new)
            overrides[spec.key] = merged
            applied.append((spec.key, new["atr_stop_mult"], new["atr_trail_mult"],
                            new["take_profit_pct"], r["OOS PF"], r["OOS n"]))
        else:
            held.append((spec.key,
                         f"old {old.get('atr_stop_mult')}/{old.get('atr_trail_mult')}/{old.get('take_profit_pct')}",
                         f"new {new['atr_stop_mult']}/{new['atr_trail_mult']}/{new['take_profit_pct']}"))

    if applied:
        user_settings.save(bot_overrides=overrides)
        try:
            import trading_engine
            trading_engine.refresh_all_exit_params()
        except Exception:
            pass

    report = {
        "ts": datetime.now(tz=timezone.utc).isoformat(timespec="minutes"),
        "applied": applied, "held_unstable": held, "skipped": skipped,
    }
    log.info("Walk-forward apply: %d applied, %d held (unstable), %d skipped",
             len(applied), len(held), len(skipped))
    return report


def _run_sweep() -> bool:
    """Run the full fleet sweep in a subprocess (same script the user runs)."""
    try:
        proc = subprocess.run(
            ["python", "/app/_fleet_run.py"],
            capture_output=True, text=True, timeout=2 * 3600,
        )
        ok = proc.returncode == 0 and "saved ->" in (proc.stdout or "")
        if not ok:
            log.error("Walk-forward sweep failed (rc=%s): %s",
                      proc.returncode, (proc.stderr or "")[-400:])
        return ok
    except Exception:
        log.exception("Walk-forward sweep crashed")
        return False


def _loop() -> None:
    # First boot ever: seed the clock so the first automatic sweep happens a
    # week from NOW — never surprise-burn an hour of CPU on a fresh deploy
    # (and never collide with a manually launched sweep).
    state = _load_state()
    if "last_run_epoch" not in state:
        state["last_run_epoch"] = time.time()
        _save_state(state)
    while True:
        try:
            state = _load_state()
            last = float(state.get("last_run_epoch", 0.0))
            if time.time() - last >= PERIOD_SEC:
                log.info("Walk-forward due — running weekly fleet sweep")
                if _run_sweep():
                    report = apply_with_stability_gate()
                    state = {"last_run_epoch": time.time(), "report": report}
                    _save_state(state)
                else:
                    # Failed — retry in a day rather than hammering hourly.
                    state["last_run_epoch"] = time.time() - PERIOD_SEC + 86400
                    _save_state(state)
        except Exception:
            log.exception("Walk-forward scheduler iteration failed")
        time.sleep(CHECK_EVERY_SEC)


def ensure_scheduler() -> None:
    global _thread, _started
    if _started and _thread and _thread.is_alive():
        return
    _started = True
    _thread = threading.Thread(target=_loop, daemon=True, name="walk-forward")
    _thread.start()
    log.info("Walk-forward scheduler started (weekly, stability-gated)")


def last_report() -> dict:
    return _load_state().get("report") or {}
