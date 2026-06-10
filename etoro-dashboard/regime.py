"""
Market-regime detection — a shared, cheap classifier used by:

  • the entry gate (trading_engine) to silence strategy FAMILIES that are in the
    wrong regime — e.g. don't let mean-reversion bots knife-catch a strong trend,
    and don't let breakout bots fire in dead, rangebound volatility; and
  • the ATR-adaptive stop sizing (exit_profiles.adaptive_stop_pct), which needs a
    live ATR% to widen stops in volatile regimes.

Everything here is pure pandas/numpy on the candle DataFrame already held by
market_data_hub — no API calls, no state.  Designed to FAIL OPEN: any error or
insufficient data returns an "unknown" regime that allows every strategy, so a
detector hiccup can never freeze trading.

Regime is described on two independent axes:
  trend:  "up" | "down" | "range"     (direction + strength, via ADX + EMA slope)
  vol:    "low" | "normal" | "high"   (ATR% percentile vs its own recent history)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
ADX_PERIOD          = 14
ATR_PERIOD          = 14
ADX_TREND_MIN       = 25.0    # ADX ≥ this ⇒ a real trend (Wilder's classic threshold)
ADX_STRONG          = 40.0
VOL_LOOKBACK        = 100     # candles for the ATR% percentile baseline
VOL_LOW_PCTILE      = 0.30    # ATR% below this percentile ⇒ "low" vol
VOL_HIGH_PCTILE     = 0.75    # ATR% above this percentile ⇒ "high" vol


@dataclass(frozen=True)
class RegimeState:
    trend:    str      # "up" | "down" | "range" | "unknown"
    vol:      str      # "low" | "normal" | "high" | "unknown"
    adx:      float
    atr_pct:  float    # ATR as % of last close
    ema_slope_pct: float
    label:    str      # compact human label, e.g. "up/high"

    @property
    def is_trending(self) -> bool:
        return self.trend in ("up", "down")

    @property
    def is_ranging(self) -> bool:
        return self.trend == "range"


_UNKNOWN = RegimeState("unknown", "unknown", 0.0, 0.0, 0.0, "unknown")


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def atr_pct(df: pd.DataFrame, period: int = ATR_PERIOD) -> Optional[float]:
    """ATR as a percentage of the latest close. None if not computable."""
    try:
        if df is None or len(df) < period + 1:
            return None
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        close = df["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = _wilder(tr, period).iloc[-1]
        last = float(close.iloc[-1])
        if not last or pd.isna(atr):
            return None
        return float(atr) / last * 100.0
    except Exception:
        return None


def _adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> float:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up = high.diff()
    down = -low.diff()
    pos_dm = up.where((up > down) & (up > 0), 0.0)
    neg_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = _wilder(tr, period).replace(0, np.nan)
    pos_di = 100 * _wilder(pos_dm, period) / atr
    neg_di = 100 * _wilder(neg_dm, period) / atr
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    adx = _wilder(dx, period).iloc[-1]
    return float(adx) if pd.notna(adx) else 0.0


def classify(df: pd.DataFrame) -> RegimeState:
    """Classify the current regime from an OHLC DataFrame. Fails open to unknown."""
    try:
        if df is None or len(df) < max(ADX_PERIOD, ATR_PERIOD) * 2:
            return _UNKNOWN
        close = df["Close"].astype(float)
        last = float(close.iloc[-1])

        ap = atr_pct(df) or 0.0
        adx = _adx(df)

        # EMA slope over ~ADX_PERIOD candles, as % of price — direction tiebreaker
        ema = close.ewm(span=ADX_PERIOD, adjust=False).mean()
        slope_pct = (
            (float(ema.iloc[-1]) - float(ema.iloc[-ADX_PERIOD])) / last * 100.0
            if len(ema) > ADX_PERIOD and last else 0.0
        )

        # Trend axis
        if adx >= ADX_TREND_MIN and slope_pct > 0:
            trend = "up"
        elif adx >= ADX_TREND_MIN and slope_pct < 0:
            trend = "down"
        else:
            trend = "range"

        # Volatility axis: percentile of current ATR% within its recent history
        vol = "normal"
        hist = []
        n = min(VOL_LOOKBACK, len(df) - ATR_PERIOD - 1)
        if n >= 20:
            sub = df.tail(n + ATR_PERIOD + 1)
            high = sub["High"].astype(float)
            low = sub["Low"].astype(float)
            c = sub["Close"].astype(float)
            pc = c.shift(1)
            tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
            atr_series = _wilder(tr, ATR_PERIOD) / c * 100.0
            hist = atr_series.dropna().tolist()
        if len(hist) >= 20:
            arr = np.array(hist)
            pct_rank = float((arr < ap).mean())
            if pct_rank <= VOL_LOW_PCTILE:
                vol = "low"
            elif pct_rank >= VOL_HIGH_PCTILE:
                vol = "high"

        return RegimeState(trend, vol, round(adx, 2), round(ap, 4), round(slope_pct, 4),
                           f"{trend}/{vol}")
    except Exception as exc:
        log.warning("regime.classify failed (%s) — unknown", exc)
        return _UNKNOWN


# ── Strategy-family suitability ───────────────────────────────────────────────
# Maps strategy kind (from exit_profiles) → which regimes it should trade in.
# Returns (allowed, reason).  Fails open: unknown regime / unknown kind allows.
def allows(strategy_key: str, rs: RegimeState) -> tuple[bool, str]:
    """Should this strategy trade in the current regime?"""
    if rs.trend == "unknown":
        return True, ""
    try:
        import exit_profiles
        kind = exit_profiles.profile(strategy_key).kind
    except Exception:
        return True, ""

    # Mean-reversion / oscillators: avoid STRONG trends (they fade the trend).
    if kind == "mean_revert":
        if rs.is_trending and rs.adx >= ADX_STRONG:
            return False, f"mean-reversion suppressed in strong {rs.trend}-trend (ADX {rs.adx:.0f})"
        return True, ""

    # Arbitrage / pairs: want range or normal conditions, not a violent trend.
    if kind == "arb":
        if rs.is_trending and rs.adx >= ADX_STRONG and rs.vol == "high":
            return False, f"arb suppressed in strong high-vol {rs.trend}-trend"
        return True, ""

    # Trend / LLM: avoid dead, directionless low-vol chop (whipsaw risk).
    if kind in ("trend", "llm"):
        if rs.is_ranging and rs.vol == "low":
            return False, "trend strategy suppressed in low-vol range (whipsaw risk)"
        return True, ""

    return True, ""
