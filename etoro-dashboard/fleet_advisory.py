"""
Per-bot fleet-optimization advisory — informational only.

When walk-forward / fleet optimization disagrees with a bot's current setup
(weak OOS, no qualified row, unstable param jump), we record the reason here
and surface it on the Bots tab.  We NEVER stop, disable, or delete bots based
on this — open positions stay managed until the user closes them.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_ADVISORY_PATH = Path(os.environ.get("FLEET_ADVISORY_PATH", "/app/data/fleet_bot_advisory.json"))
_WF_PATH = Path(os.environ.get("WALK_FORWARD_PATH", "/app/data/walk_forward.json"))
_FLEET_PATH = Path(os.environ.get("FLEET_OPT_PATH", "/app/data/fleet_opt.json"))
_lock = threading.RLock()
_cache: dict[str, dict] = {}


def _save(data: dict) -> None:
    try:
        _ADVISORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ADVISORY_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _ADVISORY_PATH)
    except Exception:
        log.warning("Could not persist fleet advisory map", exc_info=True)


def _load() -> dict[str, dict]:
    try:
        if not _ADVISORY_PATH.exists():
            return {}
        data = json.loads(_ADVISORY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fleet_row_for(spec, table: list, names: dict):
    import instrument_config
    return instrument_config.fleet_row_for_spec(table, spec, names)


def _open_position_ids() -> set[str]:
    """Position ids currently open on eToro."""
    import positions_cache

    cached = positions_cache.get_positions()
    if cached:
        return {str(p.get("position_id")) for p in cached if p.get("position_id") is not None}
    try:
        import os
        from etoro_client import get_shared_client

        api_key = os.environ.get("ETORO_API_KEY", "")
        user_key = os.environ.get("ETORO_USER_KEY", "")
        if not api_key or not user_key:
            return set()
        client = get_shared_client(api_key, user_key)
        positions = client.get_open_positions(demo=True)
        return {str(p.get("position_id")) for p in positions if p.get("position_id") is not None}
    except Exception:
        return set()


def _bot_has_owned_open_position(bot_key: str, open_pids: set[str]) -> bool:
    """True when this bot owns an open position on eToro right now."""
    import bot_registry
    import trade_manager

    uid = bot_registry.get(bot_key)
    if not uid:
        return False
    pid = None
    if trade_manager.has_open(uid):
        trade = trade_manager.get_open(uid)
        pid = trade.etoro_position_id if trade else None
    if pid is None:
        pid = trade_manager.owned_position_id(uid)
    if pid is None:
        return False
    return str(pid) in open_pids


def rebuild_advisories() -> dict[str, dict]:
    """Rebuild the advisory map from walk-forward report + fleet_opt.json."""
    import instrument_config
    import strategies as strategies_mod

    names = strategies_mod.display_names()
    advisories: dict[str, dict] = {}
    now = datetime.now(tz=timezone.utc).isoformat(timespec="minutes")

    try:
        wf = json.loads(_WF_PATH.read_text(encoding="utf-8"))
        report = wf.get("report") or {}
    except Exception:
        report = {}

    for entry in report.get("held_unstable") or []:
        if not entry:
            continue
        key = entry[0]
        detail = (
            f"params unstable — kept current overrides ({entry[1]}) "
            f"instead of new optimum ({entry[2]})"
        )
        advisories[key] = {
            "kind": "unstable",
            "reason": "Fleet optimization disagrees (unstable params)",
            "detail": detail,
            "ts": report.get("ts") or now,
        }

    for entry in report.get("skipped") or []:
        if not entry:
            continue
        key, reason = entry[0], entry[1] if len(entry) > 1 else ""
        advisories[key] = {
            "kind": "skipped",
            "reason": "Fleet optimization disagrees",
            "detail": reason or "no qualified fleet row",
            "ts": report.get("ts") or now,
        }

    applied_keys = {a[0] for a in (report.get("applied") or []) if a}
    open_pids = _open_position_ids()

    table: list = []
    if _FLEET_PATH.exists():
        try:
            table = json.loads(_FLEET_PATH.read_text(encoding="utf-8")).get("rows") or []
        except Exception:
            pass

    if table and open_pids:
        for spec in instrument_config.load_specs():
            if spec.key in advisories or spec.key in applied_keys:
                continue
            if not _bot_has_owned_open_position(spec.key, open_pids):
                continue
            row = _fleet_row_for(spec, table, names)
            if row is None:
                advisories[spec.key] = {
                    "kind": "no_row",
                    "reason": "Fleet optimization disagrees (no qualified plan)",
                    "detail": "No passing fleet row for this strategy × asset × interval",
                    "ts": now,
                }
                continue
            oos_pf = float(row.get("OOS PF") or 0)
            oos_n = int(row.get("OOS n") or 0)
            if oos_pf < 1.0 or oos_n < 5:
                pf_txt = "∞" if oos_pf >= 99 else f"{oos_pf:.2f}"
                advisories[spec.key] = {
                    "kind": "weak_oos",
                    "reason": "Fleet optimization disagrees (weak OOS)",
                    "detail": f"OOS PF={pf_txt}, n={oos_n} — not applying new exits",
                    "ts": now,
                }

    with _lock:
        _cache.clear()
        _cache.update(advisories)
        _save(advisories)
    return advisories


def get_advisory(bot_key: str) -> Optional[dict]:
    with _lock:
        if not _cache:
            _cache.update(_load())
        return _cache.get(bot_key)


def disagrees(bot_key: str) -> bool:
    return get_advisory(bot_key) is not None


def card_caption(bot_key: str) -> str:
    """Short Bots-tab caption; empty when fleet opt is aligned."""
    adv = get_advisory(bot_key)
    if not adv:
        return ""
    detail = (adv.get("detail") or "").strip()
    reason = adv.get("reason") or "Fleet optimization disagrees"
    if detail:
        return f"⚠️ {reason}: {detail} — position kept; you decide whether to close"
    return f"⚠️ {reason} — position kept; you decide whether to close"


# Warm cache on import
with _lock:
    _cache.update(_load())
