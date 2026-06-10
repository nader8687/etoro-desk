"""Stochastic RSI strategy.

Applies the Stochastic oscillator formula to RSI values, making it far
more sensitive than plain RSI — useful for 1-minute scalping where plain
RSI is too slow to react.

%K = (RSI − lowest_RSI_N) / (highest_RSI_N − lowest_RSI_N) × 100
%D = SMA(3) of %K   ← signal line

Signals fire ONLY on %K / %D crossovers inside extreme zones:
  BUY  — %K crosses above %D while both ≤ oversold (default 20)
  SELL — %K crosses below %D while both ≥ overbought (default 80)

Being in an extreme zone without a crossover returns HOLD with a
contextual note — not a directional trade signal.  Firing trades on
mere proximity to the extreme level (without a turning-point crossover)
generates excessive false positives.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class StochRSIStrategy(Strategy):
    key          = "stoch_rsi"
    display_name = "Stochastic RSI"
    description  = (
        "Applies the Stochastic formula to RSI values for faster signals.  "
        "BUY on %K/%D bullish crossover in oversold zone; SELL in overbought zone.  "
        "Extreme zone without a crossover = HOLD (awaiting turning-point confirmation)."
    )

    def __init__(
        self,
        rsi_period:   int   = 14,
        stoch_period: int   = 14,
        d_period:     int   = 3,
        oversold:     float = 20.0,
        overbought:   float = 80.0,
    ) -> None:
        self.rsi_period   = rsi_period
        self.stoch_period = stoch_period
        self.d_period     = d_period
        self.oversold     = oversold
        self.overbought   = overbought

    @staticmethod
    def _rsi(closes: pd.Series, period: int) -> pd.Series:
        delta    = closes.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs       = avg_gain / avg_loss.replace(0, float("nan"))
        return 100 - (100 / (1 + rs))

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        need = self.rsi_period + self.stoch_period + self.d_period + 3
        if len(df) < need:
            return None

        closes  = df["Close"].astype(float)
        rsi_ser = self._rsi(closes, self.rsi_period)

        min_rsi = rsi_ser.rolling(self.stoch_period).min()
        max_rsi = rsi_ser.rolling(self.stoch_period).max()
        stoch_k = 100 * (rsi_ser - min_rsi) / (max_rsi - min_rsi + 1e-10)
        stoch_d = stoch_k.rolling(self.d_period).mean()

        k_curr  = float(stoch_k.iloc[-1])
        k_prev  = float(stoch_k.iloc[-2])
        d_curr  = float(stoch_d.iloc[-1])
        d_prev  = float(stoch_d.iloc[-2])
        rsi_now = float(rsi_ser.iloc[-1])

        if any(pd.isna(v) for v in (k_curr, k_prev, d_curr, d_prev)):
            return None

        crossed_up   = k_prev <= d_prev and k_curr > d_curr
        crossed_down = k_prev >= d_prev and k_curr < d_curr

        obs = [
            f"StochRSI %K: {k_curr:.1f}",
            f"StochRSI %D: {d_curr:.1f}",
            f"RSI({self.rsi_period}): {rsi_now:.1f}",
        ]

        # ── BUY: bullish crossover in oversold territory ──────────────────────
        if crossed_up and k_curr <= self.oversold:
            depth      = self.oversold - k_curr
            confidence = min(int(62 + depth * 1.2), 90)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"%K ({k_curr:.1f}) crossed above %D ({d_curr:.1f}) "
                    f"in oversold zone — momentum turning bullish"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # ── SELL: bearish crossover in overbought territory ───────────────────
        if crossed_down and k_curr >= self.overbought:
            depth      = k_curr - self.overbought
            confidence = min(int(62 + depth * 1.2), 90)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"%K ({k_curr:.1f}) crossed below %D ({d_curr:.1f}) "
                    f"in overbought zone — momentum turning bearish"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # ── Extreme zone but no crossover yet — HOLD with context ────────────
        if k_curr <= self.oversold:
            return StrategySignal(
                signal="HOLD",
                confidence=40,
                reasoning=(
                    f"StochRSI %K ({k_curr:.1f}) in oversold zone — "
                    f"awaiting bullish %K/%D crossover"
                ),
                risk_level="LOW",
                observations=obs,
            )

        if k_curr >= self.overbought:
            return StrategySignal(
                signal="HOLD",
                confidence=40,
                reasoning=(
                    f"StochRSI %K ({k_curr:.1f}) in overbought zone — "
                    f"awaiting bearish %K/%D crossover"
                ),
                risk_level="LOW",
                observations=obs,
            )

        # ── Neutral zone ─────────────────────────────────────────────────────
        zone = "lower half" if k_curr < 50 else ("upper half" if k_curr > 50 else "midpoint")
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"StochRSI %K {k_curr:.1f} in {zone} — no extreme zone signal",
            risk_level="LOW",
            observations=obs,
        )
