"""Moving Average Crossover strategy (EMA fast/slow)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class MACrossoverStrategy(Strategy):
    key          = "ma_crossover"
    display_name = "MA Crossover (9/21 EMA)"
    description  = (
        "Golden cross (fast EMA crosses above slow EMA) → BUY.  "
        "Death cross (fast crosses below slow) → SELL."
    )

    def __init__(self, fast: int = 9, slow: int = 21) -> None:
        self.fast = fast
        self.slow = slow

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < self.slow + 3:
            return None

        closes   = df["Close"].astype(float)
        fast_ema = closes.ewm(span=self.fast, adjust=False).mean()
        slow_ema = closes.ewm(span=self.slow, adjust=False).mean()

        cf, cs = float(fast_ema.iloc[-1]), float(slow_ema.iloc[-1])
        pf, ps = float(fast_ema.iloc[-2]), float(slow_ema.iloc[-2])

        spread_pct = abs(cf - cs) / cs * 100 if cs else 0

        obs = [
            f"EMA{self.fast}: {cf:.5f}",
            f"EMA{self.slow}: {cs:.5f}",
            f"Spread: {spread_pct:.3f}%",
        ]

        crossed_up   = pf <= ps and cf > cs
        crossed_down = pf >= ps and cf < cs

        if crossed_up:
            confidence = min(int(62 + spread_pct * 8), 92)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Golden cross: EMA{self.fast} ({cf:.5f}) crossed above "
                    f"EMA{self.slow} ({cs:.5f})"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if crossed_down:
            confidence = min(int(62 + spread_pct * 8), 92)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Death cross: EMA{self.fast} ({cf:.5f}) crossed below "
                    f"EMA{self.slow} ({cs:.5f})"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        direction = "above" if cf > cs else "below"
        return StrategySignal(
            signal="HOLD",
            confidence=45,
            reasoning=(
                f"EMA{self.fast} {direction} EMA{self.slow} — "
                f"no fresh cross this candle"
            ),
            risk_level="LOW",
            observations=obs,
        )
