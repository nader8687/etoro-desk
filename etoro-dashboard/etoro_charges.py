"""eToro virtual-account charges — API fees + published spread-fee estimates.

eToro's trade-history API exposes a ``fees`` field (often 0 on demo) while
opening/closing spread fees are documented separately (e.g. 1% crypto, 0.15%
stocks/ETFs per side).  This module surfaces both so operators can see virtual
costs even when the API does not itemise them on every row.
"""
from __future__ import annotations

from typing import Callable, Optional

# Published opening/closing spread fee (% of position value, each side).
# https://www.etoro.com/trading/fees/
_SPREAD_FEE_PCT: dict[str, float] = {
    "crypto": 1.0,
    "stock": 0.15,
    "etf": 0.15,
    "index": 0.15,
    "commodity": 0.15,
    "forex": 0.0,
}


def spread_fee_pct_for_label(label: str) -> float:
    """Per-side spread fee % from instrument label / asset class."""
    try:
        import exit_profiles
        klass = exit_profiles.asset_class(label or "")
    except Exception:
        klass = ""
    if klass in _SPREAD_FEE_PCT:
        return _SPREAD_FEE_PCT[klass]
    # Fleet-heavy default: treat unknown as crypto CFD.
    return _SPREAD_FEE_PCT["crypto"]


def estimate_side_spread_fee(invested: float, label: str = "") -> float:
    """One open OR one close spread fee in dollars."""
    if invested <= 0:
        return 0.0
    return invested * spread_fee_pct_for_label(label) / 100.0


def estimate_round_trip_spread(invested: float, label: str = "") -> float:
    """Open + close spread fees (eToro charges each side on CFDs)."""
    return estimate_side_spread_fee(invested, label) * 2.0


def _float(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def enrich_closed_trade(
    trade: dict,
    *,
    label_for_id: Optional[Callable[[int | None], str]] = None,
) -> dict:
    """Attach charge fields to one eToro history row (mutates in place)."""
    iid = trade.get("instrumentId") or trade.get("instrument_id")
    try:
        iid_int = int(iid) if iid is not None else None
    except (TypeError, ValueError):
        iid_int = None

    label = ""
    if label_for_id and iid_int is not None:
        label = label_for_id(iid_int) or ""

    invested = _float(trade.get("investment") or trade.get("initialInvestment"))
    api_fees = _float(trade.get("fees"))
    est_open = estimate_side_spread_fee(invested, label)
    est_close = estimate_side_spread_fee(invested, label)
    est_spread = est_open + est_close
    pct = spread_fee_pct_for_label(label)

    trade["_charge_label"] = label
    trade["_charge_api_fees"] = api_fees
    trade["_charge_est_open"] = est_open
    trade["_charge_est_close"] = est_close
    trade["_charge_est_spread"] = est_spread
    trade["_charge_spread_pct"] = pct
    trade["_charge_total"] = api_fees if api_fees > 0 else est_spread
    return trade


def enrich_open_position(pos: dict) -> dict:
    """Attach charge fields to a normalized open position (mutates in place)."""
    label = str(pos.get("name") or pos.get("symbol") or "")
    invested = _float(pos.get("amount"))
    api_fees = _float(pos.get("total_fees")) + _float(pos.get("total_external_fees"))
    est_open = estimate_side_spread_fee(invested, label)
    est_close = estimate_side_spread_fee(invested, label)
    pos["_charge_api_fees"] = api_fees
    pos["_charge_est_open"] = est_open
    pos["_charge_est_close_if_close_now"] = est_close
    pos["_charge_est_round_trip"] = est_open + est_close
    pos["_charge_spread_pct"] = spread_fee_pct_for_label(label)
    return pos


def summarize_closed_trades(trades: list[dict]) -> dict:
    """Aggregate charge stats for a list of enriched history rows."""
    n = len(trades)
    api = sum(_float(t.get("_charge_api_fees")) for t in trades)
    est = sum(_float(t.get("_charge_est_spread")) for t in trades)
    est_open = sum(_float(t.get("_charge_est_open")) for t in trades)
    est_close = sum(_float(t.get("_charge_est_close")) for t in trades)
    pnl = sum(_float(t.get("netProfit")) for t in trades if t.get("netProfit") is not None)
    invested = sum(_float(t.get("investment")) for t in trades)
    return {
        "count": n,
        "api_fees": api,
        "est_spread": est,
        "est_open": est_open,
        "est_close": est_close,
        "net_pnl": pnl,
        "invested": invested,
        "pnl_after_est_charges": pnl - est if est else pnl,
    }


def summarize_open_positions(positions: list[dict]) -> dict:
    """Aggregate charge stats for enriched open positions."""
    api = sum(_float(p.get("_charge_api_fees")) for p in positions)
    est_open = sum(_float(p.get("_charge_est_open")) for p in positions)
    est_close = sum(_float(p.get("_charge_est_close_if_close_now")) for p in positions)
    return {
        "count": len(positions),
        "api_fees_paid": api,
        "est_open_spread": est_open,
        "est_close_if_all_closed": est_close,
        "est_full_round_trip": est_open + est_close,
    }
