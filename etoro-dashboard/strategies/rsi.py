"""RSI (Relative Strength Index) strategy."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class RSIStrategy(Strategy):
    key          = "rsi"
    display_name = "RSI (14)"
    description  = (
        "Buys when RSI drops below the oversold threshold and sells when it "
        "rises above the overbought threshold.  Default: oversold=30, overbought=70."
    )

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period     = period
        self.oversold   = oversold
        self.overbought = overbought

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
        if len(df) < self.period + 2:
            return None

        closes = df["Close"].astype(float)
        rsi    = float(self._rsi(closes, self.period).iloc[-1])

        if pd.isna(rsi):
            return None

        obs = [f"RSI({self.period}): {rsi:.1f}"]

        if rsi <= self.oversold:
            severity   = self.oversold - rsi                      # 0–30
            confidence = min(int(55 + severity / self.oversold * 40), 93)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"RSI {rsi:.1f} ≤ {self.oversold} — oversold, "
                    f"mean-reversion bounce expected"
                ),
                risk_level="LOW" if rsi < 20 else "MEDIUM",
                observations=obs,
            )

        if rsi >= self.overbought:
            severity   = rsi - self.overbought                    # 0–30
            confidence = min(int(55 + severity / (100 - self.overbought) * 40), 93)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"RSI {rsi:.1f} ≥ {self.overbought} — overbought, "
                    f"pullback expected"
                ),
                risk_level="LOW" if rsi > 80 else "MEDIUM",
                observations=obs,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=50,
            reasoning=f"RSI {rsi:.1f} neutral ({self.oversold}–{self.overbought})",
            risk_level="LOW",
            observations=obs,
        )
