"""
Dynamic position sizing — risk-based, performance-adaptive, account-aware.

Replaces the flat demo_amount with a per-trade dollar size computed at the
moment of entry from three live inputs:

1. RISK BUDGET (how much we're willing to LOSE on this trade)
       risk_$ = equity × RISK_PCT_PER_TRADE × performance_multiplier
   The performance multiplier comes from the strategy's own recent closed
   trades in the journal (profit factor over a rolling window), so strategies
   that are currently winning size up and strategies that are bleeding size
   down — automatically, in real time, per (strategy × instrument-class).

2. STOP DISTANCE (how far the stop is, from exit_profiles)
       notional_$ = risk_$ / stop_pct
   A tight-stop strategy can take a larger notional for the same dollar risk;
   a wide-stop trend trade takes a smaller one.  Capped at MAX_POSITION_PCT
   of equity so tight stops never produce silly sizes.

3. PER-TRADE CAP — min(config, MAX_TRADE_USD), then × strategy-kind % from
   exit_profiles.size_pct() (arb 50%, mean-revert 75%, trend/llm 100%).
4. CASH RESERVE — keep CASH_RESERVE_PCT of free cash untouched; shrink to
   spendable; skip if below MIN_TRADE_USD.

Account snapshot (equity / free cash / invested) is pulled from eToro's P&L
endpoint and cached for ACCOUNT_TTL_SEC, so sizing never adds an API call to
the per-tick hot path.

All decisions are logged and returned with a breakdown so the journal can
record WHY a trade was sized the way it was.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from etoro_client import EToroClient

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
RISK_PCT_PER_TRADE = 0.75   # % of equity risked per trade at neutral performance
MAX_POSITION_PCT   = 6.0    # hard ceiling: one position ≤ this % of equity
CASH_RESERVE_PCT   = 10.0   # keep this % of free cash untouched (90% deployable)
RESERVE_HARD_PCT   = 5.0    # cash-freeing may relax to this % of free cash (95% deployable)
                            # for a strong-enough signal (see cash_manager)
MIN_TRADE_USD      = 200.0  # below this, skip the trade (eToro minimums / dust)
MAX_TRADE_USD      = 1000.0 # default ceiling; Settings tab may override via user_settings


def _trading():
    try:
        import user_settings
        return user_settings.trading_settings()
    except Exception:
        return None


def max_trade_usd() -> float:
    """Absolute $ cap per new position (Settings tab → user_settings)."""
    ts = _trading()
    return ts.max_trade_usd if ts else MAX_TRADE_USD


def min_trade_usd() -> float:
    ts = _trading()
    return ts.min_trade_usd if ts else MIN_TRADE_USD


def risk_pct_per_trade() -> float:
    ts = _trading()
    return ts.risk_pct_per_trade if ts else RISK_PCT_PER_TRADE


def max_position_pct() -> float:
    ts = _trading()
    return ts.max_position_pct if ts else MAX_POSITION_PCT


def cash_reserve_pct() -> float:
    ts = _trading()
    return ts.cash_reserve_pct if ts else CASH_RESERVE_PCT


def reserve_hard_pct() -> float:
    ts = _trading()
    return ts.reserve_hard_pct if ts else RESERVE_HARD_PCT


def demo_trade_default() -> float:
    ts = _trading()
    if ts:
        return ts.demo_trade_amount
    import os
    return float(os.environ.get("ETORO_DEMO_TRADE_AMOUNT", "1000"))
PERF_WINDOW        = 20     # rolling closed-trade window per strategy
PERF_MIN_TRADES    = 5      # below this sample, multiplier stays neutral (1.0)
PERF_MULT_MIN      = 0.4    # floor for cold strategies
PERF_MULT_MAX      = 1.5    # ceiling for hot strategies
ACCOUNT_TTL_SEC    = 60.0   # account snapshot cache lifetime

# Fallback flat size when the account snapshot is unavailable (API hiccup):
# behave like the legacy fixed-amount system rather than not trading at all.
FALLBACK_TO_CONFIG_AMOUNT = True


@dataclass
class SizeDecision:
    amount: float            # dollars to invest (0 = skip the trade)
    risk_dollars: float      # intended max loss at the stop
    perf_mult: float         # performance multiplier applied
    stop_pct: float          # stop distance used (% of entry)
    equity: float            # account equity at decision time
    free_cash: float         # spendable cash before this trade
    reserve: float           # cash floor we must not break
    reason: str              # human-readable explanation
    capped_by: str = ""      # which constraint bound the size ("" = risk budget)

    def summary(self) -> str:
        return (
            f"${self.amount:,.0f} (risk ${self.risk_dollars:,.0f} @ stop {self.stop_pct:.2f}%, "
            f"perf×{self.perf_mult:.2f}{', cap: ' + self.capped_by if self.capped_by else ''})"
        )


# ── Account snapshot cache ────────────────────────────────────────────────────
_acct_lock = threading.Lock()
_acct_cache: dict = {}        # equity / cash / invested / unrealized snapshot


def _account_snapshot(client: "EToroClient", is_demo: bool) -> Optional[dict]:
    now = time.time()
    with _acct_lock:
        if _acct_cache and now - _acct_cache.get("at", 0) < ACCOUNT_TTL_SEC:
            return _acct_cache
    try:
        data = client.get_pnl(demo=is_demo) or {}
        cp = data.get("clientPortfolio", {}) or {}
        positions = cp.get("positions", []) or []
        cash     = float(cp.get("credit") or 0.0)
        bonus    = float(cp.get("bonusCredit") or 0.0)
        unreal   = float(cp.get("unrealizedPnL") or 0.0)
        invested = sum(float(p.get("amount") or 0) for p in positions)
        snap = {
            "at": now,
            "equity": cash + bonus + invested + unreal,
            "free_cash": cash,
            "bonus": bonus,
            "invested": invested,
            "unrealized": unreal,
            "open_value": invested + unreal,
            "position_count": len(positions),
        }
        with _acct_lock:
            _acct_cache.clear()
            _acct_cache.update(snap)
        return snap
    except Exception as exc:
        log.warning("Account snapshot failed (sizing will fall back): %s", exc)
        with _acct_lock:
            return dict(_acct_cache) if _acct_cache else None


def invalidate_account_cache() -> None:
    """Call after an open/close so the next sizing sees fresh cash numbers."""
    with _acct_lock:
        _acct_cache.clear()


def reserve_and_spendable(free_cash: float, reserve_pct: float) -> tuple[float, float]:
    """Cash floor and deployable amount from a % of free cash (not equity)."""
    pct = max(0.0, float(reserve_pct))
    reserve = free_cash * pct / 100.0
    return reserve, max(0.0, free_cash - reserve)


def liquidity_summary(client: "EToroClient", is_demo: bool) -> dict:
    """Account liquidity for the UI — reuses the same cached P&L snapshot as sizing.

    Returns keys: equity, free_cash, reserve, spendable, invested, unrealized,
    open_value, position_count (None when the account API is unavailable).
    Matches eToro's Virtual Portfolio bar: cash + invested + unrealized = equity.
    """
    snap = _account_snapshot(client, is_demo)
    if not snap or snap.get("equity", 0) <= 0:
        empty = {
            "equity": None, "free_cash": None, "reserve": None, "spendable": None,
            "invested": None, "unrealized": None, "open_value": None,
            "position_count": None,
        }
        return empty
    equity = float(snap["equity"])
    free   = float(snap["free_cash"])
    invested = float(snap.get("invested") or 0.0)
    unreal = float(snap.get("unrealized") or 0.0)
    reserve, spendable = reserve_and_spendable(free, cash_reserve_pct())
    return {
        "equity": equity,
        "free_cash": free,
        "reserve": reserve,
        "spendable": spendable,
        "invested": invested,
        "unrealized": unreal,
        "open_value": invested + unreal,
        "position_count": int(snap.get("position_count") or 0),
    }


# ── Performance multiplier ────────────────────────────────────────────────────
def performance_multiplier(strategy: str) -> tuple[float, str]:
    """Rolling profit-factor → size multiplier in [PERF_MULT_MIN, PERF_MULT_MAX].

    profit factor = gross wins / gross losses over the last PERF_WINDOW closed
    bot trades of this strategy (all instruments — strategy skill, not luck on
    one symbol).  Mapping:
        pf ≥ 1.5         → 1.5   (hot — size up 50%)
        pf 1.0 … 1.5     → 1.0 … 1.5  linear
        pf 0.0 … 1.0     → 0.4 … 1.0  linear (bleeding — size down)
        < PERF_MIN_TRADES samples → 1.0 (neutral, no evidence either way)
    """
    import trade_journal
    try:
        rows = trade_journal.closed_records(strategy=strategy)
    except Exception:
        return 1.0, "journal unavailable — neutral"
    rows = [r for r in rows if (r.get("bot_id") or "").strip()][:PERF_WINDOW]
    if len(rows) < PERF_MIN_TRADES:
        return 1.0, f"only {len(rows)} recent trades — neutral"

    gross_win  = sum(float(r.get("pnl_dollars") or 0) for r in rows if float(r.get("pnl_dollars") or 0) > 0)
    gross_loss = -sum(float(r.get("pnl_dollars") or 0) for r in rows if float(r.get("pnl_dollars") or 0) < 0)
    if gross_loss <= 0:
        pf = 2.0 if gross_win > 0 else 1.0
    else:
        pf = gross_win / gross_loss

    if pf >= 1.5:
        mult = PERF_MULT_MAX
    elif pf >= 1.0:
        mult = 1.0 + (pf - 1.0) / 0.5 * (PERF_MULT_MAX - 1.0)
    else:
        mult = PERF_MULT_MIN + pf * (1.0 - PERF_MULT_MIN)
    return round(mult, 2), f"pf={pf:.2f} over last {len(rows)} trades"


# ── Main entry point ──────────────────────────────────────────────────────────
def size_trade(
    client: "EToroClient",
    *,
    strategy: str,
    instrument_label: str,
    is_demo: bool,
    config_amount: float,
    reserve_pct: Optional[float] = None,
    inflight_usd: float = 0.0,
) -> SizeDecision:
    """Compute the dollar amount for a new trade.  amount=0 means SKIP.

    reserve_pct overrides the cash-reserve floor (defaults to CASH_RESERVE_PCT);
    cash-freeing passes RESERVE_HARD_PCT to deploy a strong signal into the
    reserve band without closing anything.

    inflight_usd: dollars already committed by OTHER bots' entry orders still in
    flight (order_executor ledger).  Subtracted from spendable so a same-second
    burst of entries can't collectively overshoot the cash reserve that each
    one individually fits."""
    import exit_profiles

    stop_pct = max(exit_profiles.stop_loss_min_pct(strategy, instrument_label), 0.1)
    perf_mult, perf_why = performance_multiplier(strategy)

    strat_pct = exit_profiles.size_pct(strategy)
    cap = max_trade_usd()
    base_cap  = (
        min(config_amount, cap)
        if config_amount >= min_trade_usd()
        else cap
    )
    strategy_cap = base_cap * strat_pct / 100.0

    snap = _account_snapshot(client, is_demo)
    if snap is None or snap.get("equity", 0) <= 0:
        amount = (
            strategy_cap if FALLBACK_TO_CONFIG_AMOUNT else 0.0
        )
        return SizeDecision(
            amount=amount, risk_dollars=amount * stop_pct / 100, perf_mult=perf_mult,
            stop_pct=stop_pct, equity=0.0, free_cash=0.0, reserve=0.0,
            reason=(
                f"account snapshot unavailable — fallback "
                f"{strat_pct:.0f}% of ${base_cap:,.0f} cap"
            ),
            capped_by="fallback",
        )

    rpct      = cash_reserve_pct() if reserve_pct is None else max(0.0, reserve_pct)
    equity    = snap["equity"]
    free_cash = snap["free_cash"]
    reserve, spendable = reserve_and_spendable(free_cash, rpct)
    if inflight_usd > 0:
        spendable = max(0.0, spendable - inflight_usd)

    # 1. Risk budget → notional via stop distance
    risk_dollars = equity * risk_pct_per_trade() / 100.0 * perf_mult
    amount = risk_dollars / (stop_pct / 100.0)
    capped_by = ""

    # 2. Hard position cap (% of equity)
    max_pos_pct = max_position_pct()
    max_pos = equity * max_pos_pct / 100.0
    if amount > max_pos:
        amount = max_pos
        capped_by = f"max position {max_pos_pct:.0f}% of equity"

    # 3. Per-trade ceiling: global max ($1000), then strategy-kind % share.
    if amount > strategy_cap:
        amount = strategy_cap
        capped_by = (
            f"{strat_pct:.0f}% of ${base_cap:,.0f} cap "
            f"({exit_profiles.profile(strategy).kind})"
        )

    # 4. Cash reserve
    if amount > spendable:
        amount = spendable
        capped_by = f"cash reserve ({rpct:.0f}% of free cash = ${reserve:,.0f} idle)"

    # 5. Minimum viable trade
    min_usd = min_trade_usd()
    if amount < min_usd:
        return SizeDecision(
            amount=0.0, risk_dollars=risk_dollars, perf_mult=perf_mult,
            stop_pct=stop_pct, equity=equity, free_cash=free_cash, reserve=reserve,
            reason=(
                f"skipped — spendable ${spendable:,.0f} after reserve can't fund "
                f"min trade ${min_usd:,.0f} ({perf_why})"
            ),
            capped_by="insufficient cash",
        )

    actual_risk = amount * stop_pct / 100.0
    return SizeDecision(
        amount=round(amount, 2), risk_dollars=round(actual_risk, 2),
        perf_mult=perf_mult, stop_pct=stop_pct,
        equity=equity, free_cash=free_cash, reserve=reserve,
        reason=(
            f"risk ${actual_risk:,.0f} ({RISK_PCT_PER_TRADE}% × {perf_mult:.2f} perf "
            f"[{perf_why}]) / stop {stop_pct:.2f}% · size {strat_pct:.0f}% of "
            f"${base_cap:,.0f} cap"
        ),
        capped_by=capped_by,
    )
