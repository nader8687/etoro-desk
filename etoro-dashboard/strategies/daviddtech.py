"""DaviddTech / NNFX-style stacked strategy.

The DaviddTech method (NNFX school): every indicator has exactly ONE job and
an entry fires only when all modules agree on the same candle —

  • BASELINE      — Ehlers 2-Pole Super Smoother (20).  Close above it = longs
                    only, below = shorts only.  Direction gate, not a timer.
  • CONFIRMATION  — Schaff Trend Cycle (23/50/10).  Times the entry: BUY when
                    STC crosses UP out of its oversold zone (25), SELL when it
                    crosses DOWN out of overbought (75).
  • FILTER        — Choppiness Index (14).  Vetoes any entry above 61.8
                    (ranging tape); readings under 38.2 mark a strong trend
                    and add confidence.

Exits are deliberately NOT this module's job — the engine's ATR golden-rule
entry stop and chandelier trail ARE the NNFX "ATR exit" module, applied
per-bot from Settings like every other strategy.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy, StrategySignal


def _super_smoother(closes: np.ndarray, period: int) -> np.ndarray:
    """Ehlers 2-Pole Super Smoother filter (recursive IIR)."""
    a1 = math.exp(-math.sqrt(2.0) * math.pi / period)
    b1 = 2.0 * a1 * math.cos(math.sqrt(2.0) * math.pi / period)
    c2, c3 = b1, -a1 * a1
    c1 = 1.0 - c2 - c3
    ss = np.empty_like(closes)
    ss[0] = closes[0]
    ss[1] = closes[1] if len(closes) > 1 else closes[0]
    for i in range(2, len(closes)):
        ss[i] = c1 * (closes[i] + closes[i - 1]) / 2.0 + c2 * ss[i - 1] + c3 * ss[i - 2]
    return ss


def _stc(closes: pd.Series, fast: int, slow: int, cycle: int) -> pd.Series:
    """Schaff Trend Cycle — a double-stochastic of MACD, 0..100."""
    macd = (
        closes.ewm(span=fast, adjust=False).mean()
        - closes.ewm(span=slow, adjust=False).mean()
    )
    ll = macd.rolling(cycle).min()
    rng = (macd.rolling(cycle).max() - ll).replace(0.0, np.nan)
    f1 = (100.0 * (macd - ll) / rng).ffill().fillna(50.0)
    pf = f1.ewm(alpha=0.5, adjust=False).mean()
    ll2 = pf.rolling(cycle).min()
    rng2 = (pf.rolling(cycle).max() - ll2).replace(0.0, np.nan)
    f2 = (100.0 * (pf - ll2) / rng2).ffill().fillna(50.0)
    return f2.ewm(alpha=0.5, adjust=False).mean()


def _choppiness(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
    """Choppiness Index (last value): 100·log10(ΣTR/range)/log10(n).
    > 61.8 = ranging (veto) · < 38.2 = strong trend."""
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    tr_sum = tr.rolling(period).sum()
    rng = high.rolling(period).max() - low.rolling(period).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        chop = 100.0 * np.log10(tr_sum / rng.replace(0.0, np.nan)) / math.log10(period)
    val = float(chop.iloc[-1])
    return val if math.isfinite(val) else 100.0


class DaviddTechStrategy(Strategy):
    key          = "daviddtech"
    display_name = "DaviddTech Stack (SS·STC·CHOP)"
    description  = (
        "NNFX-style module stack: Ehlers Super Smoother baseline gates the "
        "direction, a Schaff Trend Cycle cross out of its extreme zone times "
        "the entry, and the Choppiness Index vetoes ranging markets.  All "
        "three must agree on the same candle; the engine's ATR stop and "
        "chandelier trail do the exits."
    )

    def __init__(
        self,
        ss_period: int     = 20,
        stc_fast: int      = 23,
        stc_slow: int      = 50,
        stc_cycle: int     = 10,
        chop_period: int   = 14,
        chop_max: float    = 61.8,
        chop_strong: float = 38.2,
        stc_low: float     = 25.0,
        stc_high: float    = 75.0,
    ) -> None:
        self.ss_period   = ss_period
        self.stc_fast    = stc_fast
        self.stc_slow    = stc_slow
        self.stc_cycle   = stc_cycle
        self.chop_period = chop_period
        self.chop_max    = chop_max
        self.chop_strong = chop_strong
        self.stc_low     = stc_low
        self.stc_high    = stc_high

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        need = max(self.stc_slow + 2 * self.stc_cycle, self.ss_period + 3,
                   self.chop_period + 2) + 10
        if len(df) < need:
            return None

        closes = df["Close"].astype(float)
        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)

        ss = _super_smoother(closes.to_numpy(dtype=float), self.ss_period)
        baseline, baseline_prev = float(ss[-1]), float(ss[-2])
        price = float(closes.iloc[-1])
        above = price > baseline
        rising = baseline > baseline_prev

        stc = _stc(closes, self.stc_fast, self.stc_slow, self.stc_cycle)
        stc_curr, stc_prev = float(stc.iloc[-1]), float(stc.iloc[-2])
        crossed_up   = stc_prev <= self.stc_low  < stc_curr
        crossed_down = stc_prev >= self.stc_high > stc_curr

        chop = _choppiness(highs, lows, closes, self.chop_period)
        chop_ok = chop < self.chop_max
        strong_trend = chop < self.chop_strong

        obs = [
            f"Baseline SS({self.ss_period}): {baseline:.5f} — price "
            f"{'ABOVE (longs only)' if above else 'BELOW (shorts only)'}"
            f"{', rising' if rising else ', falling'}",
            f"STC: {stc_prev:.1f} → {stc_curr:.1f}"
            f"{'  · crossed UP out of oversold' if crossed_up else ''}"
            f"{'  · crossed DOWN out of overbought' if crossed_down else ''}",
            f"CHOP({self.chop_period}): {chop:.1f} — "
            + ("strong trend" if strong_trend else
               ("tradeable" if chop_ok else f"choppy (veto > {self.chop_max})")),
        ]

        if crossed_up and above and chop_ok:
            confidence = 62 + (14 if rising else 0) + (14 if strong_trend else 0)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 90),
                reasoning=(
                    f"All modules aligned LONG: price above Super Smoother baseline"
                    f"{' (rising)' if rising else ''}, STC crossed up through "
                    f"{self.stc_low:.0f} ({stc_prev:.1f}→{stc_curr:.1f}), "
                    f"CHOP {chop:.1f} {'< ' + str(self.chop_strong) + ' strong trend' if strong_trend else 'tradeable'}"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if crossed_down and not above and chop_ok:
            confidence = 62 + (14 if not rising else 0) + (14 if strong_trend else 0)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 90),
                reasoning=(
                    f"All modules aligned SHORT: price below Super Smoother baseline"
                    f"{' (falling)' if not rising else ''}, STC crossed down through "
                    f"{self.stc_high:.0f} ({stc_prev:.1f}→{stc_curr:.1f}), "
                    f"CHOP {chop:.1f} {'< ' + str(self.chop_strong) + ' strong trend' if strong_trend else 'tradeable'}"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # HOLD — say which module vetoed (NNFX dashboard style)
        if crossed_up or crossed_down:
            veto = "baseline disagrees" if (crossed_up != above or crossed_down == above) else ""
            if not chop_ok:
                veto = (veto + " · " if veto else "") + f"CHOP {chop:.1f} choppy"
            reason = f"STC fired but vetoed: {veto or 'modules not aligned'}"
        else:
            side = "long" if above else "short"
            reason = f"Baseline allows {side}s; STC {stc_curr:.1f} — no cross this candle"
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=reason,
            risk_level="LOW",
            observations=obs,
        )
