"""
Backtesting engine — replays eToro history through the LIVE trading stack.

Fidelity principles
-------------------
* SAME code paths as production: strategies.get(key).generate() for signals,
  exit_profiles for stop/TP/trail parameters (which read the Settings tab —
  a backtest therefore reflects your CURRENT exit configuration).
* NO LOOKAHEAD:
    - signals are computed on candles[: i+1] (closed bars only, like live
      candle-close dispatch) and FILLED AT THE NEXT BAR'S OPEN;
    - the chandelier trail/peak is updated with a bar's extremes only AFTER
      that bar has been tested against the PREVIOUS bar's level.
* COSTS: half-spread paid on entry and on exit (spread_pct input), plus the
  optional TRADE_FEE_PCT round-trip fee used elsewhere in analytics.
* CONSERVATIVE intrabar ordering: when several exit levels are touched within
  one bar, the WORST for us fires first (stop → trail → take-profit).

Honest limitations (also shown on the Backtest page)
----------------------------------------------------
* The LLM strategy can't be replayed (per-candle vision calls + journal
  memory would leak future knowledge) — rule strategies only.
* eToro's history endpoint caps the window (~1000 candles), so a 1m backtest
  covers hours, 15m ≈ ten days, 4h ≈ half a year.
* Recovery/breakeven-floor exits are not simulated (they depend on the live
  journal's average-hold statistics).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

WARMUP_BARS = 60   # min closed candles before the first signal is taken


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class BTTrade:
    direction: str          # LONG | SHORT
    entry_idx: int
    exit_idx: int
    entry_time: object
    exit_time: object
    entry_price: float      # cost-adjusted fill
    exit_price: float       # cost-adjusted fill
    pnl_dollars: float
    pnl_pct: float
    reason: str             # stop_loss | trailing_stop | take_profit | reversal | end_of_data
    confidence: int = 0


@dataclass
class BTResult:
    strategy: str
    instrument_label: str
    interval_secs: int
    n_bars: int
    amount: float
    spread_pct: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)   # cumulative $ after each closed trade

    # ── metrics ──────────────────────────────────────────────────────────────
    def _pnls(self, trades=None):
        return [t.pnl_dollars for t in (trades if trades is not None else self.trades)]

    def summary(self, trades=None) -> dict:
        pnls = self._pnls(trades)
        n = len(pnls)
        if n == 0:
            return {"n": 0, "pnl": 0.0, "win_rate": 0.0, "pf": 0.0,
                    "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "max_dd": 0.0, "avg_hold_bars": 0.0}
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gw, gl = sum(wins), -sum(losses)
        eq, peak, dd = 0.0, 0.0, 0.0
        for p in pnls:
            eq += p
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        tl = trades if trades is not None else self.trades
        return {
            "n": n,
            "pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / n, 3),
            "pf": round(gw / gl, 2) if gl > 0 else (math.inf if gw > 0 else 0.0),
            "expectancy": round(sum(pnls) / n, 2),
            "avg_win": round(gw / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gl / len(losses), 2) if losses else 0.0,
            "max_dd": round(dd, 2),
            "avg_hold_bars": round(sum(t.exit_idx - t.entry_idx for t in tl) / n, 1),
        }

    def oos_split(self, frac: float = 0.7) -> tuple[dict, dict]:
        """In-sample (first frac of bars) vs out-of-sample (rest) by ENTRY bar."""
        cut = int(self.n_bars * frac)
        ins = [t for t in self.trades if t.entry_idx < cut]
        oos = [t for t in self.trades if t.entry_idx >= cut]
        return self.summary(ins), self.summary(oos)

    def by_reason(self) -> dict:
        out: dict[str, dict] = {}
        for t in self.trades:
            d = out.setdefault(t.reason, {"n": 0, "pnl": 0.0})
            d["n"] += 1
            d["pnl"] = round(d["pnl"] + t.pnl_dollars, 2)
        return out


# ── ATR (same math as regime.py — Wilder RMA on True Range) ──────────────────

def _atr_pct_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (atr / c * 100.0).fillna(0.0)


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy_key: str,
    instrument_label: str,
    instrument_id: int,
    interval_secs: int,
    *,
    amount: float = 1000.0,
    spread_pct: float = 0.05,
    progress_cb=None,
) -> Optional[BTResult]:
    """Replay one strategy over one candle DataFrame.  None if not runnable."""
    import strategies
    import exit_profiles

    try:
        strat = strategies.get(strategy_key)
    except Exception:
        return None
    if strat is None or getattr(strat, "is_async", False):
        return None   # async (LLM) strategies are not replayable

    df = df.reset_index(drop=True)
    n = len(df)
    if n < WARMUP_BARS + 10:
        return None

    opens  = df["Open"].astype(float).tolist()
    highs  = df["High"].astype(float).tolist()
    lows   = df["Low"].astype(float).tolist()
    closes = df["Close"].astype(float).tolist()
    times  = df["time"].tolist()
    atrp   = _atr_pct_series(df).tolist()

    half_spread = spread_pct / 100.0 / 2.0
    fee_rt = 0.0
    try:
        import trade_journal
        fee_rt = float(getattr(trade_journal, "_FEE_PCT", 0.0)) / 100.0
    except Exception:
        pass

    trail_mult = exit_profiles.atr_trail_mult(strategy_key)
    trailing_cfg, tp_cfg = exit_profiles.resolve(strategy_key, instrument_label=instrument_label)

    result = BTResult(
        strategy=strategy_key, instrument_label=instrument_label,
        interval_secs=interval_secs, n_bars=n, amount=amount, spread_pct=spread_pct,
    )

    in_pos = False
    direction = ""
    entry_fill = 0.0
    entry_idx = 0
    entry_conf = 0
    stop_level = 0.0
    tp_level = 0.0
    trail_level = 0.0
    peak = 0.0
    pending_signal = None   # (direction, confidence) — filled at next bar open
    equity = 0.0

    def _close(i: int, raw_px: float, reason: str) -> None:
        nonlocal in_pos, equity
        if direction == "LONG":
            exit_fill = raw_px * (1.0 - half_spread)
            move = (exit_fill - entry_fill) / entry_fill
        else:
            exit_fill = raw_px * (1.0 + half_spread)
            move = (entry_fill - exit_fill) / entry_fill
        pnl = amount * move - amount * fee_rt
        equity += pnl
        result.trades.append(BTTrade(
            direction=direction, entry_idx=entry_idx, exit_idx=i,
            entry_time=times[entry_idx], exit_time=times[i],
            entry_price=round(entry_fill, 6), exit_price=round(exit_fill, 6),
            pnl_dollars=round(pnl, 4), pnl_pct=round(move * 100.0, 4),
            reason=reason, confidence=entry_conf,
        ))
        result.equity_curve.append(round(equity, 4))
        in_pos = False

    for i in range(WARMUP_BARS, n):
        if progress_cb and i % 100 == 0:
            progress_cb(i / n)

        # ── 1. Fill a pending entry at THIS bar's open (no lookahead) ─────────
        if pending_signal is not None and not in_pos:
            d, conf = pending_signal
            pending_signal = None
            raw = opens[i]
            if raw > 0 and atrp[i - 1] > 0:
                direction = d
                entry_conf = conf
                entry_fill = raw * (1.0 + half_spread) if d == "LONG" else raw * (1.0 - half_spread)
                entry_idx = i
                stop_pct = exit_profiles.adaptive_stop_pct(
                    strategy_key, instrument_label, atr_pct=atrp[i - 1],
                )
                if d == "LONG":
                    stop_level = entry_fill * (1.0 - stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 + tp_cfg / 100.0) if tp_cfg > 0 else 0.0
                    peak = entry_fill
                else:
                    stop_level = entry_fill * (1.0 + stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 - tp_cfg / 100.0) if tp_cfg > 0 else 0.0
                    peak = entry_fill
                trail_level = stop_level   # chandelier starts AT the hard stop
                in_pos = True

        # ── 2. Manage an open position against THIS bar ──────────────────────
        if in_pos:
            lo, hi = lows[i], highs[i]
            closed_this_bar = False
            # Conservative ordering: stop → trail → TP, tested on PRIOR levels.
            if direction == "LONG":
                if lo <= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_this_bar = True
                elif trailing_cfg > 0 and lo <= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_this_bar = True
                elif tp_level > 0 and hi >= tp_level:
                    _close(i, tp_level, "take_profit"); closed_this_bar = True
            else:
                if hi >= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_this_bar = True
                elif trailing_cfg > 0 and hi >= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_this_bar = True
                elif tp_level > 0 and lo <= tp_level:
                    _close(i, tp_level, "take_profit"); closed_this_bar = True

            # AFTER testing, update peak + ratchet the chandelier with this bar
            if in_pos and not closed_this_bar and atrp[i] > 0:
                atr_px = atrp[i] / 100.0
                if direction == "LONG":
                    peak = max(peak, hi)
                    trail_level = max(trail_level, peak * (1.0 - trail_mult * atr_px))
                else:
                    peak = min(peak, lo)
                    trail_level = min(trail_level, peak * (1.0 + trail_mult * atr_px))

        # ── 3. Signal on THIS closed bar (window = bars 0..i, like live) ─────
        window = df.iloc[: i + 1]
        c = closes[i]
        ask = c * (1.0 + half_spread)
        bid = c * (1.0 - half_spread)
        try:
            sig = strat.generate(window, ask, bid, instrument_id)
        except Exception:
            sig = None
        s = (getattr(sig, "signal", "") or "").upper() if sig else ""

        if in_pos:
            # Reversal exit at the close of the signal bar (mirrors live).
            if (direction == "LONG" and s == "SELL") or (direction == "SHORT" and s == "BUY"):
                _close(i, c, "reversal")
        elif s in ("BUY", "SELL") and pending_signal is None:
            pending_signal = ("LONG" if s == "BUY" else "SHORT",
                              int(getattr(sig, "confidence", 0) or 0))

    if in_pos:
        _close(n - 1, closes[n - 1], "end_of_data")
    if progress_cb:
        progress_cb(1.0)
    return result
