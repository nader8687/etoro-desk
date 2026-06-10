"""Donchian Channel Breakout strategy.

Defines a rolling N-period price channel (highest high / lowest low).
A close above the channel high signals a sustained upside breakout;
a close below the channel low signals a downside breakout.

Unlike ORB (session-based, first N candles only), Donchian is a rolling
window that adapts throughout the day and across sessions — signals
represent genuine multi-candle momentum rather than opening-range dynamics.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class DonchianStrategy(Strategy):
    key          = "donchian"
    display_name = "Donchian Breakout (20)"
    description  = (
        "Rolling 20-period high/low channel.  BUY when close breaks above "
        "the channel high; SELL when it breaks below the channel low."
    )

    def __init__(self, period: int = 20) -> None:
        self.period = period

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

        # Channel is defined by the N periods BEFORE the current candle
        channel_window = df.iloc[-(self.period + 1):-1]
        ch_high = float(channel_window["High"].max())
        ch_low  = float(channel_window["Low"].min())
        ch_mid  = (ch_high + ch_low) / 2
        ch_rng  = ch_high - ch_low if ch_high > ch_low else 1e-10

        curr_close = float(df["Close"].iloc[-1])

        obs = [
            f"Channel high ({self.period}p): {ch_high:.5f}",
            f"Channel low  ({self.period}p): {ch_low:.5f}",
            f"Channel mid:                  {ch_mid:.5f}",
            f"Channel range:                {ch_rng:.5f}",
        ]

        if curr_close > ch_high:
            excess_pct = (curr_close - ch_high) / ch_rng * 100
            confidence = min(int(60 + excess_pct * 3), 92)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Close {curr_close:.5f} broke above {self.period}-period "
                    f"channel high {ch_high:.5f} (+{excess_pct:.1f}% of range)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if curr_close < ch_low:
            excess_pct = (ch_low - curr_close) / ch_rng * 100
            confidence = min(int(60 + excess_pct * 3), 92)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Close {curr_close:.5f} broke below {self.period}-period "
                    f"channel low {ch_low:.5f} (-{excess_pct:.1f}% of range)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # Inside channel — show relative position
        pos_pct = (curr_close - ch_low) / ch_rng * 100
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=(
                f"Close {curr_close:.5f} inside channel "
                f"({pos_pct:.0f}% from low)"
            ),
            risk_level="LOW",
            observations=obs,
        )
