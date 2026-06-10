"""Mean Reversion (Z-score) strategy.

Computes a rolling Z-score of the close price relative to its own moving
average.  Trades the expected snap-back when price strays too far from
its historical mean.

This is the single-instrument counterpart to Statistical Arbitrage —
no second running engine required.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class MeanReversionStrategy(Strategy):
    key          = "mean_reversion"
    display_name = "Mean Reversion (Z-score)"
    description  = (
        "Buys when price is ≥2σ below its rolling mean, sells when ≥2σ above.  "
        "Works well in ranging markets; complements trend-following strategies."
    )

    def __init__(
        self,
        lookback:    int   = 20,
        z_threshold: float = 2.0,
    ) -> None:
        self.lookback    = lookback
        self.z_threshold = z_threshold

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < self.lookback + 2:
            return None

        closes = df["Close"].astype(float)
        window = closes.iloc[-(self.lookback + 1):]

        mean = float(window.mean())
        std  = float(window.std())

        if std < 1e-10:
            return None

        curr_close = float(closes.iloc[-1])
        z          = (curr_close - mean) / std

        # How far is the current price from mean as % of mean?
        dev_pct = (curr_close - mean) / mean * 100 if mean else 0

        obs = [
            f"Z-score: {z:+.2f}  (threshold ±{self.z_threshold})",
            f"Rolling mean ({self.lookback}): {mean:.5f}",
            f"σ: {std:.5f}",
            f"Deviation: {dev_pct:+.3f}%",
        ]

        if z <= -self.z_threshold:
            severity   = abs(z) - self.z_threshold
            confidence = min(int(58 + severity * 14), 90)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Price {curr_close:.5f} is {abs(z):.2f}σ below {self.lookback}-period "
                    f"mean ({mean:.5f}) — mean-reversion bounce expected"
                ),
                risk_level="MEDIUM" if abs(z) < 3 else "LOW",
                observations=obs,
            )

        if z >= self.z_threshold:
            severity   = z - self.z_threshold
            confidence = min(int(58 + severity * 14), 90)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Price {curr_close:.5f} is {z:.2f}σ above {self.lookback}-period "
                    f"mean ({mean:.5f}) — mean-reversion pullback expected"
                ),
                risk_level="MEDIUM" if z < 3 else "LOW",
                observations=obs,
            )

        # Within normal range
        direction = "above" if z > 0 else "below"
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"Z-score {z:+.2f} — price {direction} mean but within ±{self.z_threshold}σ",
            risk_level="LOW",
            observations=obs,
        )
