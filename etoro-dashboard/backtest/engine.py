"""
Event-driven backtester for EtoroDesk's classical strategies.

Goals
-----
* Reuse the SAME strategy classes the live system runs (strategies.registry), so
  a backtest measures the real signal logic, not a re-implementation.
* Model real execution costs: bid-ask spread, commission/fees, ATR-based
  slippage, and one-bar execution LATENCY (a signal at bar i fills at bar i+1's
  open) — which also structurally prevents look-ahead bias.
* Apply the SAME exit rules as live: regime-aware (ATR) hard stop, hard
  take-profit, and trailing stop, all sourced from exit_profiles.
* Support WALK-FORWARD evaluation: split the timeline into rolling out-of-sample
  folds and report per-fold metrics so we can see whether an edge is stable or
  an artefact of one lucky period.

No look-ahead
-------------
At bar i the strategy only ever sees df.iloc[:i+1] (closed bars up to and
including i).  The resulting order fills at bar i+1's open.  Stops/TPs are
checked against each subsequent bar's high/low.  Nothing reads a future bar.

This module is dependency-light (pandas + numpy) and imports no live runtime
state, so it runs in CI, a notebook, or a cron without a broker connection.

LLM and cross-instrument strategies (llm, stat_arb, rate_arb) are async / need a
second feed and are skipped with a clear message — backtest the deterministic
single-instrument strategies here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Strategies that can't be backtested in this single-instrument, synchronous loop
SKIP_STRATEGIES = {"llm", "stat_arb", "rate_arb"}


@dataclass
class BacktestConfig:
    strategy: str
    instrument_label: str = "Bitcoin  (BTC)"
    instrument_id: int = 100000
    # Costs
    spread_pct: float = 0.06        # half is paid on each side of a round trip
    fee_pct: float = 0.0            # commission per side as % of notional
    slippage_atr_frac: float = 0.05 # extra adverse fill = this × ATR, in price
    # Sizing (mirrors the live risk-based sizer, self-contained)
    start_equity: float = 10000.0
    risk_pct_per_trade: float = 0.75
    max_position_pct: float = 6.0
    # Exits — pulled from exit_profiles when None
    confidence_min: int = 0         # ignore signals below this confidence
    atr_period: int = 14
    max_hold_bars: int = 0          # 0 = no time stop


@dataclass
class BTTrade:
    direction: str
    entry_time: object
    entry_price: float
    exit_time: object
    exit_price: float
    amount: float
    stop_pct: float
    pnl_dollars: float
    pnl_pct: float
    reason: str
    confidence: int
    bars_held: int

    def to_journal_row(self, strategy: str, label: str) -> dict:
        return {
            "ts": str(self.exit_time),
            "strategy": strategy,
            "bot_id": f"bt::{strategy}",
            "instrument_label": label,
            "direction": self.direction,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "trade_amount": self.amount,
            "stop_pct_entry": self.stop_pct,
            "pnl_dollars": round(self.pnl_dollars, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "reason": self.reason,
            "win": self.pnl_dollars > 0,
        }


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    skipped: Optional[str] = None

    def journal_rows(self) -> list[dict]:
        return [t.to_journal_row(self.config.strategy, self.config.instrument_label)
                for t in self.trades]

    def metrics(self) -> dict:
        import analytics
        return analytics.compute(self.journal_rows(), self.config.strategy).to_dict()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


class Backtester:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

    def run(self, df: pd.DataFrame) -> BacktestResult:
        cfg = self.cfg
        res = BacktestResult(cfg)
        if cfg.strategy in SKIP_STRATEGIES:
            res.skipped = f"{cfg.strategy} needs LLM/second-feed — not backtestable here"
            return res

        import strategies, exit_profiles
        strat = strategies.get(cfg.strategy)
        trailing_pct, tp_pct = exit_profiles.resolve(cfg.strategy, instrument_label=cfg.instrument_label)

        df = df.reset_index(drop=True).copy()
        for col in ("Open", "High", "Low", "Close"):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column {col!r}")
        if "time" not in df.columns:
            df["time"] = pd.RangeIndex(len(df))
        atr = _atr(df, cfg.atr_period)

        equity = cfg.start_equity
        res.equity_curve.append(equity)
        pos = None  # open position dict
        n = len(df)
        half_spread = cfg.spread_pct / 100.0 / 2.0

        for i in range(cfg.atr_period + 2, n - 1):
            window = df.iloc[: i + 1]
            price = float(df["Close"].iloc[i])
            atr_pct = (float(atr.iloc[i]) / price * 100.0) if price and not np.isnan(atr.iloc[i]) else None

            # ── Manage an open position against THIS bar's range (i is "next bar"
            #    relative to the signal that opened it) ──────────────────────────
            if pos is not None:
                bar = df.iloc[i]
                exit_price, reason = self._check_exit(pos, bar, atr.iloc[i], trailing_pct, tp_pct)
                if exit_price is None and cfg.max_hold_bars and (i - pos["i_entry"]) >= cfg.max_hold_bars:
                    exit_price, reason = float(bar["Close"]), "time_stop"
                if exit_price is not None:
                    equity = self._close(res, pos, exit_price, df["time"].iloc[i], i, reason, equity)
                    pos = None

            # ── Generate a signal on closed bar i; fill at bar i+1 open ─────────
            if pos is None:
                try:
                    ask = price * (1 + half_spread)
                    bid = price * (1 - half_spread)
                    sig = strat.generate(window, ask, bid, cfg.instrument_id)
                except Exception as exc:
                    log.debug("strategy error at bar %d: %s", i, exc)
                    sig = None
                if sig and sig.signal in ("BUY", "SELL") and sig.confidence >= cfg.confidence_min:
                    pos = self._open(cfg, df, i + 1, sig, atr_pct, equity, exit_profiles)
            res.equity_curve.append(equity)

        # Force-close any position still open at the end
        if pos is not None:
            last = df.iloc[n - 1]
            equity = self._close(res, pos, float(last["Close"]), df["time"].iloc[n - 1], n - 1, "eod", equity)
        return res

    # ── helpers ───────────────────────────────────────────────────────────────
    def _open(self, cfg, df, fill_i, sig, atr_pct, equity, exit_profiles):
        if fill_i >= len(df):
            return None
        direction = "LONG" if sig.signal == "BUY" else "SHORT"
        open_px = float(df["Open"].iloc[fill_i])
        half_spread = cfg.spread_pct / 100.0 / 2.0
        atr_val = float(_atr(df, cfg.atr_period).iloc[fill_i]) if fill_i < len(df) else 0.0
        slip = cfg.slippage_atr_frac * (atr_val or 0.0)
        # Adverse fill: pay half-spread + slippage in the worse direction
        if direction == "LONG":
            entry = open_px * (1 + half_spread) + slip
        else:
            entry = open_px * (1 - half_spread) - slip

        stop_pct = exit_profiles.adaptive_stop_pct(cfg.strategy, cfg.instrument_label, atr_pct)
        risk_dollars = equity * cfg.risk_pct_per_trade / 100.0
        amount = risk_dollars / (stop_pct / 100.0) if stop_pct > 0 else 0.0
        amount = min(amount, equity * cfg.max_position_pct / 100.0)
        if amount <= 0:
            return None
        # entry-side fee
        amount_after_fee = amount
        stop_price = entry * (1 - stop_pct / 100.0) if direction == "LONG" else entry * (1 + stop_pct / 100.0)
        trailing_pct, tp_pct = exit_profiles.resolve(cfg.strategy, instrument_label=cfg.instrument_label)
        tp_price = 0.0
        if tp_pct > 0:
            tp_price = entry * (1 + tp_pct / 100.0) if direction == "LONG" else entry * (1 - tp_pct / 100.0)
        return {
            "direction": direction, "entry": entry, "amount": amount_after_fee,
            "stop_pct": stop_pct, "stop": stop_price, "tp": tp_price,
            "trailing_pct": trailing_pct, "peak": entry, "i_entry": fill_i,
            "entry_time": df["time"].iloc[fill_i], "confidence": int(sig.confidence),
        }

    def _check_exit(self, pos, bar, atr_now, trailing_pct, tp_pct):
        hi, lo, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        d = pos["direction"]
        # Update trailing peak/trough and trailing stop level
        if d == "LONG":
            pos["peak"] = max(pos["peak"], hi)
            trail = pos["peak"] * (1 - trailing_pct / 100.0) if trailing_pct > 0 else None
            # Conservative ordering: hard stop first, then trailing, then TP
            if lo <= pos["stop"]:
                return pos["stop"], "stop_loss"
            if trail is not None and lo <= trail:
                return trail, "trailing_stop"
            if pos["tp"] and hi >= pos["tp"]:
                return pos["tp"], "take_profit"
        else:
            pos["peak"] = min(pos["peak"], lo)
            trail = pos["peak"] * (1 + trailing_pct / 100.0) if trailing_pct > 0 else None
            if hi >= pos["stop"]:
                return pos["stop"], "stop_loss"
            if trail is not None and hi >= trail:
                return trail, "trailing_stop"
            if pos["tp"] and lo <= pos["tp"]:
                return pos["tp"], "take_profit"
        return None, ""

    def _close(self, res, pos, exit_px, exit_time, i, reason, equity):
        cfg = self.cfg
        half_spread = cfg.spread_pct / 100.0 / 2.0
        d = pos["direction"]
        # Exit-side spread cost
        fill = exit_px * (1 - half_spread) if d == "LONG" else exit_px * (1 + half_spread)
        units = pos["amount"] / pos["entry"] if pos["entry"] else 0.0
        gross = (fill - pos["entry"]) * units if d == "LONG" else (pos["entry"] - fill) * units
        fees = pos["amount"] * cfg.fee_pct / 100.0 * 2  # entry + exit
        pnl = gross - fees
        pnl_pct = (pnl / pos["amount"] * 100.0) if pos["amount"] else 0.0
        equity += pnl
        res.trades.append(BTTrade(
            direction=d, entry_time=pos["entry_time"], entry_price=round(pos["entry"], 6),
            exit_time=exit_time, exit_price=round(fill, 6), amount=round(pos["amount"], 2),
            stop_pct=round(pos["stop_pct"], 4), pnl_dollars=pnl, pnl_pct=pnl_pct,
            reason=reason, confidence=pos["confidence"], bars_held=i - pos["i_entry"],
        ))
        return equity


# ── Walk-forward evaluation ───────────────────────────────────────────────────
def walk_forward(cfg: BacktestConfig, df: pd.DataFrame, folds: int = 5) -> dict:
    """Run the backtest over `folds` consecutive out-of-sample windows.

    These strategies have fixed parameters (nothing is fitted), so this is
    walk-forward *evaluation* (stability across regimes), not parameter
    optimisation: if an edge only appears in one fold, it is not robust.
    """
    import analytics
    n = len(df)
    if n < folds * (cfg.atr_period + 30):
        folds = max(1, n // (cfg.atr_period + 30))
    size = n // folds
    fold_metrics = []
    all_rows: list[dict] = []
    for f in range(folds):
        lo = f * size
        hi = n if f == folds - 1 else (f + 1) * size
        sub = df.iloc[lo:hi]
        res = Backtester(cfg).run(sub)
        if res.skipped:
            return {"skipped": res.skipped}
        rows = res.journal_rows()
        all_rows.extend(rows)
        m = analytics.compute(rows, f"fold{f+1}").to_dict()
        m["bars"] = len(sub)
        fold_metrics.append(m)

    agg = analytics.compute(all_rows, "ALL").to_dict()
    pfs = [m["expectancy_usd"] for m in fold_metrics]
    profitable_folds = sum(1 for x in pfs if x > 0)
    return {
        "aggregate": agg,
        "folds": fold_metrics,
        "profitable_folds": profitable_folds,
        "total_folds": len(fold_metrics),
        "stable": profitable_folds >= max(1, int(0.6 * len(fold_metrics))),
    }


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    """Load an OHLC CSV. Accepts columns Open/High/Low/Close (+ optional time/Volume),
    case-insensitively."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for want in ("open", "high", "low", "close", "volume", "time", "date", "timestamp"):
        if want in cols:
            rename[cols[want]] = want.capitalize() if want in ("open", "high", "low", "close", "volume") else "time"
    df = df.rename(columns=rename)
    return df


def synthetic_ohlc(n: int = 2000, seed: int = 7, start: float = 100.0,
                   trend: float = 0.0, vol: float = 0.01) -> pd.DataFrame:
    """Generate a synthetic OHLC series (GBM-ish) for smoke tests / demos.

    `trend` is per-bar drift; `vol` is per-bar volatility. Produces realistic
    intrabar High/Low so stop/TP touch logic can be exercised.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    wick = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    t = pd.date_range("2025-01-01", periods=n, freq="min")
    return pd.DataFrame({"time": t, "Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": rng.integers(100, 1000, n)})
