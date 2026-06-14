"""Pick and enable the top-N bots from the saved fleet optimization table."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

FLEET_PATH = Path(os.environ.get("FLEET_OPT_PATH", "/app/data/fleet_opt.json"))

_reset_lock = threading.Lock()
_reset_state: dict = {"running": False, "message": "", "ok": None}


@dataclass
class PickResult:
    picked: list[str] = field(default_factory=list)          # bot keys
    picked_rows: list[dict] = field(default_factory=list)  # fleet rows
    created: list[str] = field(default_factory=list)       # newly appended toml keys
    turned_off: list[str] = field(default_factory=list)
    unmatched_rows: int = 0
    gated_out: int = 0
    skipped_cap: int = 0
    skipped_assets: int = 0
    requested_n: int = 0
    fleet_ts: str = ""
    message: str = ""


def reset_in_progress() -> bool:
    with _reset_lock:
        return bool(_reset_state.get("running"))


def get_reset_status() -> dict:
    with _reset_lock:
        return dict(_reset_state)


def load_fleet_rows() -> tuple[list[dict], str]:
    """Return (ok rows, saved timestamp)."""
    try:
        data = json.loads(FLEET_PATH.read_text(encoding="utf-8"))
    except Exception:
        return [], ""
    rows = [r for r in (data.get("rows") or []) if r.get("Status") == "ok"]
    return rows, str(data.get("ts") or "")


def fleet_asset_options() -> list[str]:
    """Asset short names for the Bots-tab fleet picker (hardcoded diversification set)."""
    import instrument_config

    return sorted(instrument_config.KNOWN_ASSET_LABELS.keys())


def assets_in_saved_fleet() -> set[str]:
    """KNOWN asset short names that have at least one ok row in the saved fleet run."""
    import instrument_config

    out: set[str] = set()
    for row in load_fleet_rows()[0]:
        short = instrument_config.normalize_fleet_asset_short(
            (row.get("Asset") or "").strip(),
            (row.get("Label") or "").strip(),
        )
        if short:
            out.add(short)
    return out


def _normalize_interval_label(lbl: str) -> str:
    import instrument_config as ic

    lbl = (lbl or "").strip()
    inv = {v: k for k, v in ic._INTERVAL_LABELS.items()}
    if lbl in inv:
        return lbl
    if lbl.endswith("m") and lbl[:-1].isdigit():
        return ic.interval_label_for_secs(int(lbl[:-1]) * 60)
    return lbl


def _oos_pf_value(raw) -> float:
    try:
        pf = float(raw or 0)
    except (TypeError, ValueError):
        return 0.0
    return 3.0 if pf >= 99 else pf


def row_score(row: dict) -> float:
    """OOS expectancy when available; legacy proxy otherwise."""
    oos_n = int(row.get("OOS n") or 0)
    if oos_n <= 0:
        return -1e9
    raw_pnl = row.get("OOS P&L $")
    if raw_pnl is not None:
        try:
            return float(raw_pnl) / oos_n
        except (TypeError, ValueError):
            pass
    pf = _oos_pf_value(row.get("OOS PF"))
    return (pf - 1.0) * math.sqrt(oos_n)


def _passes_gates(row: dict, *, min_oos_n: int, min_oos_pf: float) -> bool:
    oos_n = int(row.get("OOS n") or 0)
    if oos_n < min_oos_n:
        return False
    return _oos_pf_value(row.get("OOS PF")) >= min_oos_pf


def _strategy_key(strat_disp: str, names: dict) -> Optional[str]:
    inv = {v: k for k, v in names.items()}
    return inv.get((strat_disp or "").strip())


def resolve_fleet_label(row: dict, specs: list) -> Optional[str]:
    """Full eToro label for a fleet table row (KNOWN_ASSET_LABELS only)."""
    import instrument_config

    return instrument_config.label_for_asset_name(
        (row.get("Asset") or "").strip(),
        specs,
        full_label=(row.get("Label") or "").strip(),
    )


def spec_for_row(row: dict, specs, names: dict):
    """Match one fleet row to an instruments.toml spec, or None."""
    strat_disp = (row.get("Strategy") or "").strip()
    interval = _normalize_interval_label(row.get("Interval") or "")
    row_label = resolve_fleet_label(row, specs)
    if not row_label:
        return None
    for spec in specs:
        if names.get(spec.strategy, spec.strategy) != strat_disp:
            continue
        if spec.label != row_label:
            continue
        if spec.interval != interval:
            continue
        return spec
    return None


def create_bot_from_row(row: dict, specs, names: dict):
    """Append a new instruments.toml bot for a fleet row; return spec or None."""
    import instrument_config

    label = resolve_fleet_label(row, specs)
    if not label:
        log.warning("Cannot create bot — unknown asset %r", row.get("Asset"))
        return None

    strategy = _strategy_key(row.get("Strategy") or "", names)
    if not strategy:
        log.warning("Cannot create bot — unknown strategy %r", row.get("Strategy"))
        return None

    interval = _normalize_interval_label(row.get("Interval") or "")
    interval_secs = instrument_config.interval_secs_from_label(interval)
    raw_ci = row.get("check_in_secs")
    try:
        check_in_secs = int(raw_ci) if raw_ci is not None else 0
    except (TypeError, ValueError):
        check_in_secs = 0
    existing_keys = {s.key for s in specs}
    key = instrument_config.suggest_bot_key(label, strategy, interval_secs, existing_keys)

    try:
        return instrument_config.append_bot(
            key=key,
            label=label,
            strategy=strategy,
            interval=interval,
            interval_secs=interval_secs,
            auto_trade=True,
            created_via=instrument_config.BOT_SOURCE_FLEET,
            check_in_secs=check_in_secs or interval_secs,
        )
    except Exception:
        log.warning("Failed to append bot %s for fleet row", key, exc_info=True)
        return None


def _resolve_spec_for_row(
    row: dict,
    specs: list,
    names: dict,
    *,
    create_missing: bool,
) -> tuple[Optional[object], list, bool]:
    """Return (spec, updated_specs_list, was_created)."""
    spec = spec_for_row(row, specs, names)
    if spec is not None:
        return spec, specs, False
    if not create_missing:
        return None, specs, False
    spec = create_bot_from_row(row, specs, names)
    if spec is None:
        return None, specs, False
    return spec, [*specs, spec], True


def pick_top_bots(
    n: int,
    *,
    min_oos_n: int = 8,
    min_oos_pf: float = 1.0,
    max_per_asset: Optional[int] = None,
    create_missing: bool = True,
    assets: Optional[list[str]] = None,
) -> PickResult:
    """Rank fleet rows and return up to *n* bot keys.

    *assets*: optional fleet table Asset short names (e.g. Bitcoin, Tesla).
    When set, only rows for those assets are considered.

    *max_per_asset*: cap bots per asset; ``0``/unset means no cap when creating.
    """
    import instrument_config
    import strategies as strategies_mod

    result = PickResult(requested_n=max(1, int(n)))
    rows, result.fleet_ts = load_fleet_rows()
    if not rows:
        result.message = "No saved fleet optimization — run it on the Strategies tab first."
        return result

    asset_filter: Optional[set[str]] = None
    if assets:
        asset_filter = {(a or "").strip() for a in assets if (a or "").strip()}
        if not asset_filter:
            result.message = "Select at least one asset to include."
            return result

    # Default: no per-asset cap when building the configured fleet.
    cap = 0 if max_per_asset is None else int(max_per_asset)
    specs = list(instrument_config.load_specs(enabled_only=False))
    names = strategies_mod.display_names()

    candidates: list[tuple[float, dict]] = []
    for row in rows:
        row_asset = instrument_config.normalize_fleet_asset_short(
            (row.get("Asset") or "").strip(),
            (row.get("Label") or "").strip(),
        )
        if row_asset is None:
            result.skipped_assets += 1
            continue
        if asset_filter is not None and row_asset not in asset_filter:
            result.skipped_assets += 1
            continue
        if not _passes_gates(row, min_oos_n=min_oos_n, min_oos_pf=min_oos_pf):
            result.gated_out += 1
            continue
        candidates.append((row_score(row), row, row_asset))

    candidates.sort(key=lambda x: (-x[0], -int(x[1].get("OOS n") or 0)))

    picked_keys: set[str] = set()
    per_asset: dict[str, int] = {}
    for _score, row, row_asset in candidates:
        if len(result.picked) >= result.requested_n:
            break

        if cap > 0 and per_asset.get(row_asset, 0) >= cap:
            result.skipped_cap += 1
            continue

        spec, specs, was_created = _resolve_spec_for_row(
            row, specs, names, create_missing=create_missing,
        )
        if spec is None:
            result.unmatched_rows += 1
            continue
        if spec.key in picked_keys:
            continue

        if was_created:
            result.created.append(spec.key)

        per_asset[row_asset] = per_asset.get(row_asset, 0) + 1
        picked_keys.add(spec.key)
        result.picked.append(spec.key)
        result.picked_rows.append(row)

    if result.created:
        instrument_config.invalidate_cache()

    if not result.picked:
        asset_note = ""
        if asset_filter:
            asset_note = f" Assets filter: {', '.join(sorted(asset_filter))}."
        result.message = (
            f"No fleet rows became bots after gates "
            f"(OOS n≥{min_oos_n}, OOS PF≥{min_oos_pf})."
            f"{asset_note} Gated out: {result.gated_out}, unmatched: {result.unmatched_rows}."
        )
    else:
        created_note = f" ({len(result.created)} new)" if result.created else ""
        shortfall = ""
        if len(result.picked) < result.requested_n:
            shortfall = (
                f" Only {len(result.picked)} of {result.requested_n} requested — "
                f"{len(candidates)} rows passed filters"
            )
            if asset_filter:
                shortfall += f" for {', '.join(sorted(asset_filter))}"
            if result.unmatched_rows:
                shortfall += f", {result.unmatched_rows} unmatched (unknown asset/strategy)"
            if result.skipped_cap:
                shortfall += f", {result.skipped_cap} skipped (per-asset cap {cap})"
            shortfall += "."
        asset_note = ""
        if asset_filter:
            asset_note = f" Assets: {', '.join(sorted(asset_filter))}."
        result.message = (
            f"Selected {len(result.picked)} bot(s) from fleet "
            f"{result.fleet_ts or '(saved run)'}{created_note}.{asset_note}{shortfall}"
        )
    return result


def create_from_pick(
    pick: PickResult,
    *,
    apply_exit_params: bool = True,
) -> PickResult:
    """Register picked bots in instruments.toml with fleet tags and auto-trade ON."""
    import instrument_config

    if not pick.picked:
        return pick

    instrument_config.invalidate_cache()

    if apply_exit_params:
        try:
            import fleet_scheduler
            fleet_scheduler.apply_with_stability_gate()
        except Exception:
            log.warning("Fleet exit apply after create failed", exc_info=True)

    import trading_engine

    specs = instrument_config.load_specs(enabled_only=False)
    spec_by_key = {s.key: s for s in specs}
    for bot_key in pick.picked:
        spec = spec_by_key.get(bot_key)
        if spec is None or not spec.auto_trade:
            continue
        trading_engine.set_auto_trade(spec.instrument_id, True, bot_id=bot_key)

    created_note = (
        f" Created {len(pick.created)} new bot(s) in instruments.toml."
        if pick.created else ""
    )
    existing = len(pick.picked) - len(pick.created)
    existing_note = (
        f" {existing} were already in instruments.toml before this run."
        if existing else ""
    )
    pick.message = (
        f"Added {len(pick.picked)} bot(s) from fleet "
        f"{pick.fleet_ts or '(saved run)'} (auto-trade ON)."
        f"{created_note}{existing_note}"
    )
    return pick


def apply_pick(
    pick: PickResult,
    *,
    turn_off_others: bool = True,
    apply_exit_params: bool = True,
    api_key: str = "",
    user_key: str = "",
    is_demo: bool = True,
    all_instruments: Optional[dict] = None,
) -> PickResult:
    """Enable picked bots (and optionally disable the rest) in the trading engine."""
    import instrument_config
    import trading_engine

    if not pick.picked:
        return pick

    instrument_config.invalidate_cache()
    specs = instrument_config.load_specs(enabled_only=False)
    spec_by_key = {s.key: s for s in specs}
    picked_set = set(pick.picked)

    if turn_off_others:
        for spec in specs:
            if spec.key not in picked_set:
                trading_engine.set_auto_trade(spec.instrument_id, False, bot_id=spec.key)
                pick.turned_off.append(spec.key)

    pairs: list[tuple[str, int]] = []
    for bot_key in pick.picked:
        spec = spec_by_key.get(bot_key)
        if spec is None:
            continue
        trading_engine.set_auto_trade(spec.instrument_id, True, bot_id=bot_key)
        pairs.append((bot_key, spec.instrument_id))

    if apply_exit_params:
        try:
            import fleet_scheduler
            fleet_scheduler.apply_with_stability_gate()
        except Exception:
            log.warning("Fleet exit apply after top-pick failed", exc_info=True)

    import threading
    import time

    _ak, _uk = api_key, user_key
    _instruments = all_instruments or {}
    _off = list(pick.turned_off)

    def _apply_engine() -> None:
        for bot_key in _off:
            try:
                trading_engine.stop_bot(bot_key)
            except Exception:
                log.warning("Fleet top-pick stop failed for %s", bot_key, exc_info=True)
            time.sleep(0.05)
        for bot_key, iid in pairs:
            try:
                spec = spec_by_key.get(bot_key)
                if spec is None:
                    continue
                resolved = instrument_config.resolve_ids([spec], _instruments)
                if resolved:
                    trading_engine.start_instrument(
                        resolved[0], api_key=_ak, user_key=_uk, is_demo=is_demo,
                    )
            except Exception:
                log.warning("Fleet top-pick start failed for %s", bot_key, exc_info=True)
            time.sleep(0.1)

    threading.Thread(target=_apply_engine, daemon=True, name="fleet-top-pick").start()

    off_note = f" Turned off {len(pick.turned_off)} other bot(s)." if pick.turned_off else ""
    created_note = f" Created {len(pick.created)} in instruments.toml." if pick.created else ""
    pick.message = (
        f"Turned ON {len(pairs)} bot(s) from fleet "
        f"{pick.fleet_ts or ''}.{created_note}{off_note}"
    )
    return pick


def reset_all_bots(
    *,
    clear_overrides: bool = True,
    clear_registry: bool = True,
    close_positions: bool = True,
    api_key: str = "",
    user_key: str = "",
    is_demo: bool = True,
    skip_fast_phase: bool = False,
) -> dict:
    """Delete every bot from instruments.toml and purge running engines.

    Fast phase: clear toml + stop engines.  Slow phase: close open positions
    from the positions cache only (not one REST call per configured bot).
    """
    import bot_registry
    import instrument_config
    import positions_cache
    import trade_manager
    import trading_engine
    import user_settings
    from etoro_client import get_shared_client

    if close_positions and not is_demo:
        return {
            "ok": False,
            "message": (
                "Cannot reset on a live account while closing positions via API. "
                "Close all positions in eToro first, or switch to Demo."
            ),
        }

    removed_toml = 0
    purged = 0
    if not skip_fast_phase:
        removed_toml = instrument_config.clear_all_bots()
        purged = trading_engine.force_purge_all()

    close_errors: list[tuple[str, str]] = []
    if close_positions and is_demo:
        try:
            client = get_shared_client(api_key, user_key)
            positions = list(positions_cache.get_positions() or [])
            close_errors = trading_engine.close_all_bot_positions(client, positions=positions)
        except Exception as exc:
            log.warning("Fleet reset position close failed", exc_info=True)
            close_errors = [("positions", str(exc))]

    registry_cleared = bot_registry.remove_all() if clear_registry else 0
    owners_cleared = trade_manager.clear_all_position_owners()

    if clear_overrides:
        try:
            user_settings.save(bot_overrides={})
        except Exception:
            log.warning("Could not clear bot_overrides during fleet reset", exc_info=True)

    err_note = ""
    if close_errors:
        err_note = f" {len(close_errors)} position close(s) failed."

    return {
        "ok": True,
        "removed_toml": removed_toml,
        "purged_engines": purged,
        "registry_cleared": registry_cleared,
        "owners_cleared": owners_cleared,
        "close_errors": close_errors,
        "message": (
            f"Removed {removed_toml} bot(s) from instruments.toml and "
            f"stopped {purged} engine(s)."
            + (f" Cleared {registry_cleared} bot UUID(s)." if registry_cleared else "")
            + (f" Cleared {owners_cleared} position owner(s)." if owners_cleared else "")
            + err_note
        ),
    }


def reset_all_bots_fast() -> tuple[int, int]:
    """Clear instruments.toml and stop all engines (instant UI relief)."""
    import instrument_config
    import trading_engine

    removed = instrument_config.clear_all_bots()
    purged = trading_engine.force_purge_all()
    return removed, purged


def start_reset_in_background(
    *,
    api_key: str = "",
    user_key: str = "",
    is_demo: bool = True,
) -> tuple[bool, str]:
    """Kick off fleet reset on a worker thread so the UI stays responsive."""
    with _reset_lock:
        if _reset_state.get("running"):
            return False, "A fleet reset is already running."

    def _run() -> None:
        with _reset_lock:
            _reset_state["running"] = True
            _reset_state["message"] = "Closing open positions and clearing saved state…"
            _reset_state["ok"] = None
        try:
            result = reset_all_bots(
                api_key=api_key,
                user_key=user_key,
                is_demo=is_demo,
                skip_fast_phase=True,
            )
            with _reset_lock:
                _reset_state["ok"] = result.get("ok", True)
                _reset_state["message"] = result.get("message", "Reset complete.")
        except Exception as exc:
            log.exception("Fleet reset failed")
            with _reset_lock:
                _reset_state["ok"] = False
                _reset_state["message"] = f"Fleet reset failed: {exc}"
        finally:
            with _reset_lock:
                _reset_state["running"] = False

    threading.Thread(target=_run, daemon=True, name="fleet-reset").start()
    return True, "Fleet reset started — closing open positions in background."
