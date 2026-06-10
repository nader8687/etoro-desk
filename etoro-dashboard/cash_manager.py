"""
Cash liberation — free spendable cash so a strong signal can be funded WITHOUT
full liquidation.

Two levers, cheapest first:
  1. Reserve relaxation — the CALLER re-sizes with position_sizer.RESERVE_HARD_PCT
     (a lower cash-reserve floor) for a high-edge signal.  Closes nothing.
  2. Partial trim — this module shaves units off the WEAKEST open bot positions
     until enough cash is freed.  Never a full close.

"Weakest" = low strategy performance, OVERSTAYED its average holding time, or
near its stop.  The whole thing is:
  • edge-gated   — only trim a position the new signal clearly out-ranks, and
                   only free cash at all for a genuinely strong signal;
  • cooldown'd   — a position can't be trimmed more than once per window;
  • bounded      — free only what's needed, never more than MAX_TRIM_FRACTION of
                   a position, and never leave a sub-minimum dust position.

The final trade size is always re-computed from the REAL post-trim account
snapshot, so an optimistic "freed" estimate here can never over-fund a trade.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from etoro_client import EToroClient

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
MIN_EDGE_TO_FREE     = 0.50   # new-signal edge must clear this to free cash at all
EDGE_MARGIN          = 0.15   # new edge must beat a position's forward edge by this
TRIM_COOLDOWN_SEC    = 120.0  # min interval between trims of the SAME position
MIN_POSITION_AGE_SEC = 120.0  # don't trim a position younger than this
MAX_TRIM_FRACTION    = 0.75   # never shave more than this off one position
KEEP_MIN_USD         = 200.0  # don't leave a trimmed position smaller than this
OVERSTAY_REF         = 1.0    # overstay = age / avg_hold; >1 means past the average

_cooldown_lock = threading.Lock()
_last_trim_at: dict[str, float] = {}   # str(position_id) -> monotonic ts


def signal_edge(strategy: str, confidence: float) -> float:
    """Edge proxy for a NEW signal: strategy performance × confidence (~0..1.4)."""
    import position_sizer
    pm, _ = position_sizer.performance_multiplier(strategy)
    conf = max(0.0, min(100.0, float(confidence or 0.0)))
    return pm * (conf / 100.0)


def _forward_edge(strategy: str, overstay: float, near_stop: float) -> float:
    """Remaining edge of an OPEN position.  A hot strategy is worth keeping; an
    overstayed or near-stop position has mostly spent its expected move:
        decay = 1 / (1 + overstay_excess)   (1x->1.0, 2x->0.5, 3x->0.33)
        near-stop halves it at most."""
    import position_sizer
    pm, _ = position_sizer.performance_multiplier(strategy)
    decay = 1.0 / (1.0 + max(0.0, overstay - OVERSTAY_REF))
    return pm * decay * (1.0 - 0.5 * max(0.0, min(1.0, near_stop)))


def _near_stop_frac(direction: str, entry: float, stop: float, cur: float) -> float:
    """0 at/above entry, 1 at the stop — how far price has traveled toward the stop."""
    if not (entry and stop and cur) or entry == stop:
        return 0.0
    if (direction or "").upper() == "LONG":
        frac = (entry - cur) / (entry - stop)
    else:
        frac = (cur - entry) / (stop - entry)
    return max(0.0, min(1.0, frac))


def _rank_candidates(new_edge: float):
    """Open bot positions joined with their live eToro rows, scored worst-first.

    Returns a list of dicts sorted by ascending forward edge (weakest first),
    each carrying everything needed to plan a trim."""
    import positions_cache, tick_manager, trade_manager, trade_journal

    pos_by_id = {str(p.get("position_id")): p for p in positions_cache.get_positions()}
    out = []
    for t in trade_manager.get_all_open():
        pid = t.etoro_position_id
        if pid is None:
            continue
        epos = pos_by_id.get(str(pid))
        if not epos:
            continue
        amount = float(epos.get("amount") or 0.0)
        units  = float(epos.get("units") or 0.0)
        if amount <= 0 or units <= 0:
            continue
        try:
            age_min = max(0.0, (datetime.now(tz=timezone.utc) - t.entry_time).total_seconds() / 60.0)
        except Exception:
            age_min = 0.0
        avg_hold = trade_journal.avg_hold_min(t.strategy)
        overstay = (age_min / avg_hold) if avg_hold > 0 else 0.0
        quote = tick_manager.get_latest_quote(t.instrument_id)
        cur = (quote[1] if (t.direction or "").upper() == "LONG" else quote[0]) if quote else t.entry_price
        near = _near_stop_frac(t.direction, t.entry_price, t.stop_loss_price, cur)
        out.append({
            "trade": t, "pid": pid, "iid": t.instrument_id,
            "amount": amount, "units": units, "age_min": age_min,
            "overstay": overstay, "near": near,
            "fwd_edge": _forward_edge(t.strategy, overstay, near),
        })
    out.sort(key=lambda c: c["fwd_edge"])
    return out


def try_free_cash(
    client: "EToroClient",
    *,
    is_demo: bool,
    needed_usd: float,
    new_strategy: str,
    new_confidence: float,
) -> dict:
    """Partial-trim the weakest open positions to free ~needed_usd.

    Returns {"freed": float, "actions": [str], "reason": str}.  freed=0 means
    nothing was trimmed (signal too weak, or no position weak enough to displace).
    """
    new_edge = signal_edge(new_strategy, new_confidence)
    if new_edge < MIN_EDGE_TO_FREE:
        return {"freed": 0.0, "actions": [],
                "reason": f"signal edge {new_edge:.2f} < {MIN_EDGE_TO_FREE:.2f} floor"}

    import positions_cache, position_sizer, trade_manager

    now = time.monotonic()
    freed = 0.0
    actions: list[str] = []

    for c in _rank_candidates(new_edge):
        if freed >= needed_usd:
            break
        # Edge gate — never trim a position the new signal doesn't clearly beat.
        if new_edge <= c["fwd_edge"] + EDGE_MARGIN:
            continue
        # Don't churn a fresh position.
        if c["age_min"] * 60.0 < MIN_POSITION_AGE_SEC:
            continue
        # Plan the shave: enough for the remaining need, bounded by MAX_TRIM_FRACTION
        # and by leaving at least KEEP_MIN_USD in the position.
        remaining = needed_usd - freed
        max_free_here = min(c["amount"] * MAX_TRIM_FRACTION, max(0.0, c["amount"] - KEEP_MIN_USD))
        if max_free_here < 1.0:
            continue
        free_here = min(remaining, max_free_here)
        frac = free_here / c["amount"]
        units_to_deduct = round(c["units"] * frac, 6)
        if units_to_deduct <= 0:
            continue
        # Claim the cooldown slot BEFORE the network call so a parallel engine
        # thread can't trim the same position at the same instant.
        pid_s = str(c["pid"])
        with _cooldown_lock:
            if time.monotonic() - _last_trim_at.get(pid_s, 0.0) < TRIM_COOLDOWN_SEC:
                continue
            _last_trim_at[pid_s] = time.monotonic()
        try:
            client.close_demo_position(int(c["pid"]), int(c["iid"]), units_to_deduct=units_to_deduct)
        except Exception as exc:
            log.warning("Cash freeing: trim failed for position %s: %s", c["pid"], exc)
            with _cooldown_lock:          # release the claim so a retry is possible
                _last_trim_at.pop(pid_s, None)
            continue
        trade_manager.record_trim(c["pid"], frac)
        trade_manager.claim_trim_history_slices(client, c["trade"], is_demo=is_demo)
        freed += free_here
        actions.append(
            f"{c['trade'].instrument_label} −${free_here:,.0f} ({frac*100:.0f}%, "
            f"overstay {c['overstay']:.1f}x)"
        )
        log.info(
            "Cash freed: trimmed pos %s by %.0f%% (~$%.0f) — fwd_edge %.2f < new_edge %.2f",
            c["pid"], frac * 100, free_here, c["fwd_edge"], new_edge,
        )

    if freed > 0:
        position_sizer.invalidate_account_cache()   # next size_trade sees real cash
        positions_cache.invalidate()
    return {"freed": freed, "actions": actions, "reason": ""}
