"""ADX (Average Directional Index) strategy.

Measures both the STRENGTH of a trend (ADX line) and its DIRECTION
(+DI vs −DI crossovers).  Unique in the strategy set: it can indicate
"strong trend, no clear direction" — useful as a standalone signal and
as a meta-filter for other strategies.

Algorithm (Wilder's Directional Movement System)
-------------------------------------------------
  +DM = max(high − prev_high, 0)  when high−prev_high > prev_low−low
  −DM = max(prev_low − low, 0)    when prev_low−low > high−prev_high

  ATR, +DI, −DI = Wilder-smoothed (alpha = 1/period)

  DX  = 100 × |+DI − −DI| / (+DI + −DI)
  ADX = Wilder-smoothed DX

Signal logic
------------
  BUY  — +DI crosses above −DI AND ADX ≥ adx_threshold
  SELL — −DI crosses above +DI AND ADX ≥ adx_threshold
  Confidence scales with ADX strength above the threshold.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class ADXStrategy(Strategy):
    key          = "adx"
    display_name = "ADX Trend Strength (14)"
    description  = (
        "Fires on +DI/−DI crossovers when ADX confirms a strong trend.  "
        "ADX < 25 = ranging market; no signals when trend is weak."
    )

    def __init__(self, period: int = 14, adx_threshold: float = 25.0) -> None:
        self.period        = period
        self.adx_threshold = adx_threshold

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        need = self.period * 3 + 3
        if len(df) < need:
            return None

        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        closes = df["Close"].astype(float)

        # Directional movement
        high_diff = highs.diff()
        low_diff  = -lows.diff()

        pos_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
        neg_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0.0)

        # True Range
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift(1)).abs(),
            (lows  - closes.shift(1)).abs(),
        ], axis=1).max(axis=1)

        alpha = 1.0 / self.period

        # Wilder smoothing via EWM
        atr_s   = tr.ewm(alpha=alpha,    adjust=False, min_periods=self.period).mean()
        pos_dm_s = pos_dm.ewm(alpha=alpha, adjust=False, min_periods=self.period).mean()
        neg_dm_s = neg_dm.ewm(alpha=alpha, adjust=False, min_periods=self.period).mean()

        pos_di = 100 * pos_dm_s / atr_s.replace(0, float("nan"))
        neg_di = 100 * neg_dm_s / atr_s.replace(0, float("nan"))

        di_sum = (pos_di + neg_di).replace(0, float("nan"))
        dx     = 100 * (pos_di - neg_di).abs() / di_sum
        adx    = dx.ewm(alpha=alpha, adjust=False, min_periods=self.period).mean()

        pdi_curr = float(pos_di.iloc[-1])
        pdi_prev = float(pos_di.iloc[-2])
        ndi_curr = float(neg_di.iloc[-1])
        ndi_prev = float(neg_di.iloc[-2])
        adx_curr = float(adx.iloc[-1])

        if any(pd.isna(v) for v in (pdi_curr, ndi_curr, adx_curr)):
            return None

        crossed_up   = pdi_prev <= ndi_prev and pdi_curr > ndi_curr
        crossed_down = pdi_prev >= ndi_prev and pdi_curr < ndi_curr

        trend_regime = (
            "strong"   if adx_curr >= 40
            else "trending" if adx_curr >= self.adx_threshold
            else "weak/ranging"
        )

        obs = [
            f"+DI: {pdi_curr:.2f}",
            f"−DI: {ndi_curr:.2f}",
            f"ADX: {adx_curr:.2f}  (regime: {trend_regime})",
        ]

        if adx_curr < self.adx_threshold:
            return StrategySignal(
                signal="HOLD",
                confidence=35,
                reasoning=(
                    f"ADX {adx_curr:.1f} < {self.adx_threshold} — market ranging, "
                    f"trend signals unreliable"
                ),
                risk_level="LOW",
                observations=obs,
            )

        adx_strength = min((adx_curr - self.adx_threshold) / 15, 1.0)   # 0–1

        if crossed_up:
            confidence = int(62 + adx_strength * 28)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 92),
                reasoning=(
                    f"+DI ({pdi_curr:.1f}) crossed above −DI ({ndi_curr:.1f}) — "
                    f"bullish directional move confirmed by ADX {adx_curr:.1f}"
                ),
                risk_level="MEDIUM" if adx_curr < 40 else "LOW",
                observations=obs,
            )

        if crossed_down:
            confidence = int(62 + adx_strength * 28)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 92),
                reasoning=(
                    f"−DI ({ndi_curr:.1f}) crossed above +DI ({pdi_curr:.1f}) — "
                    f"bearish directional move confirmed by ADX {adx_curr:.1f}"
                ),
                risk_level="MEDIUM" if adx_curr < 40 else "LOW",
                observations=obs,
            )

        # Trending but no fresh crossover
        dominant = "+DI bullish" if pdi_curr > ndi_curr else "−DI bearish"
        return StrategySignal(
            signal="HOLD",
            confidence=45,
            reasoning=(
                f"ADX {adx_curr:.1f} = {trend_regime}, {dominant} — "
                f"no fresh DI crossover this candle"
            ),
            risk_level="LOW",
            observations=obs,
        )
