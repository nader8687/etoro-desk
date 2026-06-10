"""
Bot performance analytics — expectancy, profit factor, drawdown, Sharpe/Sortino
and a risk-adjusted ranking — computed from the durable trade journal.

This is the layer that answers "which bots actually make money on a risk-adjusted
basis once enough data exists?".  It is intentionally read-only and pandas-free in
the core path so it can run anywhere (dashboard, a cron, the backtester).

IMPORTANT — do NOT judge a bot on a handful of trades.  Every metric carries the
sample size, and `rank_bots` flags anything below MIN_SAMPLE as "insufficient
data" rather than ranking it.  Small samples are noise.

Metric definitions
------------------
expectancy_$      mean P&L per trade in dollars  (the single most important number)
expectancy_R      mean P&L per trade in R-multiples (pnl / risk-at-entry); unitless,
                  comparable across bots of different size
profit_factor     gross wins / gross losses  (>1 profitable, >1.5 good)
win_rate          fraction of winning trades
payoff_ratio      avg win / avg loss
max_drawdown_$    deepest peak-to-trough dip of the cumulative-P&L curve
sharpe            mean(per-trade return) / std(per-trade return)        [per-trade]
sortino           mean(per-trade return) / std(downside returns)       [per-trade]
                  (per-trade, not annualised — annualisation needs a stable trade
                  cadence we don't assume; multiply by sqrt(trades/yr) if desired)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Optional

# Below this many closed trades, a bot/strategy is "insufficient data" — reported
# but never ranked, so we never crown a winner on noise.
MIN_SAMPLE = 20


@dataclass
class PerfMetrics:
    key: str
    n: int
    wins: int
    losses: int
    win_rate: float
    gross_win: float
    gross_loss: float          # ≥ 0 (absolute)
    net_pnl: float
    profit_factor: float
    avg_win: float
    avg_loss: float            # ≥ 0
    payoff_ratio: float
    expectancy_usd: float
    expectancy_r: float
    max_drawdown_usd: float
    sharpe: float
    sortino: float
    sufficient: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _max_drawdown(cum: list[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def _std(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def compute(rows: list[dict], key: str = "") -> PerfMetrics:
    """Compute metrics for a list of closed-trade records (any order).

    Records are sorted by exit time so the drawdown curve is chronological.
    Each record needs at least `pnl_dollars`; `trade_amount`/`stop_pct_entry`
    enable R-multiples; `ts` enables chronological ordering.
    """
    rows = sorted(rows, key=lambda r: str(r.get("ts", "")))
    pnls = [float(r.get("pnl_dollars") or 0.0) for r in rows]
    n = len(pnls)
    if n == 0:
        return PerfMetrics(key, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    net = sum(pnls)
    win_rate = len(wins) / n
    avg_win = (gross_win / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    payoff = (avg_win / avg_loss) if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)
    expectancy_usd = net / n

    # R-multiples: pnl / dollar-risk-at-entry (trade_amount × stop_pct/100)
    r_vals: list[float] = []
    for r in rows:
        amt = float(r.get("trade_amount") or 0.0)
        stop_pct = float(r.get("stop_pct_entry") or 0.0)
        risk = amt * stop_pct / 100.0
        if risk > 0:
            r_vals.append(float(r.get("pnl_dollars") or 0.0) / risk)
    expectancy_r = (sum(r_vals) / len(r_vals)) if r_vals else 0.0

    # Per-trade return series for Sharpe/Sortino: pnl / trade_amount (fallback pnl$)
    rets: list[float] = []
    for r in rows:
        amt = float(r.get("trade_amount") or 0.0)
        p = float(r.get("pnl_dollars") or 0.0)
        rets.append(p / amt if amt > 0 else p)
    mean_ret = sum(rets) / n
    sd = _std(rets)
    downside = _std([min(0.0, x) for x in rets])
    sharpe = (mean_ret / sd) if sd > 0 else 0.0
    sortino = (mean_ret / downside) if downside > 0 else 0.0

    cum, run = [], 0.0
    for p in pnls:
        run += p
        cum.append(run)
    mdd = _max_drawdown(cum)

    return PerfMetrics(
        key=key, n=n, wins=len(wins), losses=len(losses), win_rate=round(win_rate, 4),
        gross_win=round(gross_win, 2), gross_loss=round(gross_loss, 2), net_pnl=round(net, 2),
        profit_factor=round(pf, 3) if math.isfinite(pf) else pf,
        avg_win=round(avg_win, 2), avg_loss=round(avg_loss, 2),
        payoff_ratio=round(payoff, 3) if math.isfinite(payoff) else payoff,
        expectancy_usd=round(expectancy_usd, 4), expectancy_r=round(expectancy_r, 4),
        max_drawdown_usd=round(mdd, 2), sharpe=round(sharpe, 4), sortino=round(sortino, 4),
        sufficient=n >= MIN_SAMPLE,
    )


def _group(rows: list[dict], keyfn: Callable[[dict], str]) -> dict[str, PerfMetrics]:
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(keyfn(r) or "?", []).append(r)
    return {k: compute(v, k) for k, v in buckets.items()}


def by_strategy(rows: list[dict]) -> dict[str, PerfMetrics]:
    return _group(rows, lambda r: (r.get("strategy") or "manual").strip() or "manual")


def by_bot(rows: list[dict]) -> dict[str, PerfMetrics]:
    return _group(rows, lambda r: (r.get("bot_id") or "").strip() or "manual")


def by_regime(rows: list[dict]) -> dict[str, PerfMetrics]:
    return _group(rows, lambda r: (r.get("regime") or "unknown").strip() or "unknown")


def rank_bots(
    rows: list[dict],
    *,
    by: str = "expectancy_r",
    group: str = "strategy",
    min_sample: int = MIN_SAMPLE,
) -> list[PerfMetrics]:
    """Rank groups (strategy or bot) by a risk-adjusted metric, best first.

    Groups below `min_sample` are returned last, flagged insufficient — never
    ranked above a statistically meaningful peer.  `by` ∈ {expectancy_r,
    expectancy_usd, profit_factor, sharpe, sortino, net_pnl}.
    """
    grouper = by_bot if group == "bot" else by_strategy
    metrics = list(grouper(rows).values())

    def sort_key(m: PerfMetrics):
        val = getattr(m, by, 0.0)
        val = val if math.isfinite(val) else 1e9   # inf PF sorts to the top
        return (1 if m.n >= min_sample else 0, val)

    return sorted(metrics, key=sort_key, reverse=True)


def summary(rows: list[dict]) -> dict:
    """One-call overview for the dashboard / a report."""
    overall = compute(rows, "ALL")
    return {
        "overall": overall.to_dict(),
        "by_strategy": {k: v.to_dict() for k, v in by_strategy(rows).items()},
        "by_regime": {k: v.to_dict() for k, v in by_regime(rows).items()},
        "ranking": [m.to_dict() for m in rank_bots(rows)],
        "min_sample": MIN_SAMPLE,
    }
