"""Turtle Soup (Connors & Raschke, "Street Smarts") — failed-breakout fade.

The original liquidity-sweep reversal: when price takes out an ESTABLISHED
20-bar low (sweeping the stops resting under it) and snaps straight back,
the breakdown has failed — the stop-run was the move.  Modern ICT/SMC
"liquidity grab" trading is the same mechanic.

  • Sweep   — this bar trades BELOW the prior 20-bar low…
  • Age     — …and that prior low is at least 4 bars old (an established
              level with stops under it, not churn from the last candle).
  • Snap    — the bar CLOSES back above the swept level → BUY the failure.
  • Mirror for 20-bar highs → SELL.

Deliberately anti-correlated with the fleet's Donchian/ORB breakout bots:
this strategy is paid in exactly the chop that stops them out.  Exits run on
the engine's mean-revert profile (quick take-profit banks the snap-back).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class TurtleSoupStrategy(Strategy):
    key          = "turtle_soup"
    display_name = "Turtle Soup (20-bar sweep fade)"
    description  = (
        "Fades failed breakouts: when a bar sweeps an established 20-bar "
        "low/high (a stop-run) and closes back inside, it trades the snap-back. "
        "The classic Connors & Raschke liquidity-grab reversal."
    )

    def __init__(
        self,
        lookback: int  = 20,
        min_age: int   = 4,
    ) -> None:
        self.lookback = lookback
        self.min_age  = min_age

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        lb = self.lookback
        if len(df) < lb + 6:
            return None

        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        closes = df["Close"].astype(float)

        # Prior 20-bar extremes EXCLUDING the current bar
        win_low  = lows.iloc[-(lb + 1):-1]
        win_high = highs.iloc[-(lb + 1):-1]
        prior_low,  prior_high  = float(win_low.min()), float(win_high.max())
        # Age = bars since that extreme printed (1 = previous bar)
        low_age  = len(win_low)  - int(win_low.to_numpy().argmin())  - 1
        high_age = len(win_high) - int(win_high.to_numpy().argmax()) - 1

        bar_low, bar_high = float(lows.iloc[-1]), float(highs.iloc[-1])
        close = float(closes.iloc[-1])
        rng = max(bar_high - bar_low, 1e-12)

        swept_low  = bar_low  < prior_low  and close > prior_low  and low_age  >= self.min_age
        swept_high = bar_high > prior_high and close < prior_high and high_age >= self.min_age

        obs = [
            f"Prior {lb}-bar low: {prior_low:.5f} (age {low_age} bars) · high: {prior_high:.5f} (age {high_age})",
            f"This bar: low {bar_low:.5f} / high {bar_high:.5f} / close {close:.5f}",
        ]

        if swept_low and not swept_high:
            sweep_depth = (prior_low - bar_low) / rng           # how far the stops were run
            recovery = (close - bar_low) / rng                   # how strongly it snapped back
            confidence = 62 + int(10 * min(sweep_depth * 2, 1.0)) + int(16 * recovery)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 90),
                reasoning=(
                    f"Turtle Soup: swept the {low_age}-bar-old {lb}-bar low "
                    f"({prior_low:.5f}) and closed back above it — failed breakdown, "
                    f"buying the snap-back (recovered {recovery * 100:.0f}% of the bar)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if swept_high and not swept_low:
            sweep_depth = (bar_high - prior_high) / rng
            recovery = (bar_high - close) / rng
            confidence = 62 + int(10 * min(sweep_depth * 2, 1.0)) + int(16 * recovery)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 90),
                reasoning=(
                    f"Turtle Soup: swept the {high_age}-bar-old {lb}-bar high "
                    f"({prior_high:.5f}) and closed back below it — failed breakout, "
                    f"selling the snap-back (gave back {recovery * 100:.0f}% of the bar)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        state = ("inside the range" if prior_low <= close <= prior_high
                 else "beyond the range but no qualifying sweep")
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"No failed-breakout sweep this candle — price {state}",
            risk_level="LOW",
            observations=obs,
        )
