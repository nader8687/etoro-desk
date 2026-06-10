"""MACD (Moving Average Convergence Divergence) strategy.

Measures momentum *acceleration*, not just direction — the key difference
from a plain MA crossover.  Signals fire on MACD-line / signal-line
crossovers; histogram divergence from price is noted in observations.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class MACDStrategy(Strategy):
    key          = "macd"
    display_name = "MACD (12/26/9)"
    description  = (
        "Trades MACD-line / signal-line crossovers.  BUY on bullish crossover, "
        "SELL on bearish crossover.  Confidence scales with histogram magnitude."
    )

    def __init__(
        self,
        fast: int   = 12,
        slow: int   = 26,
        signal: int = 9,
    ) -> None:
        self.fast   = fast
        self.slow   = slow
        self.signal = signal

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        need = self.slow + self.signal + 3
        if len(df) < need:
            return None

        closes      = df["Close"].astype(float)
        fast_ema    = closes.ewm(span=self.fast,   adjust=False).mean()
        slow_ema    = closes.ewm(span=self.slow,   adjust=False).mean()
        macd_line   = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram   = macd_line - signal_line

        m_curr = float(macd_line.iloc[-1])
        m_prev = float(macd_line.iloc[-2])
        s_curr = float(signal_line.iloc[-1])
        s_prev = float(signal_line.iloc[-2])
        h_curr = float(histogram.iloc[-1])
        h_prev = float(histogram.iloc[-2])

        # Crossovers
        crossed_up   = m_prev <= s_prev and m_curr > s_curr
        crossed_down = m_prev >= s_prev and m_curr < s_curr

        # Divergence: histogram direction vs MACD direction
        hist_growing  = h_curr > h_prev
        hist_shrinking = h_curr < h_prev

        # Confidence: histogram magnitude relative to recent range
        hist_range = float(histogram.iloc[-self.signal:].abs().max()) or 1e-10
        hist_strength = min(abs(h_curr) / hist_range, 1.0)

        obs = [
            f"MACD line:  {m_curr:+.6f}",
            f"Signal line: {s_curr:+.6f}",
            f"Histogram:  {h_curr:+.6f}",
        ]

        if crossed_up:
            confidence = int(62 + hist_strength * 28)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 92),
                reasoning=(
                    f"MACD ({m_curr:+.6f}) crossed above signal ({s_curr:+.6f}) — "
                    f"bullish momentum{'  · histogram expanding' if hist_growing else ''}"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if crossed_down:
            confidence = int(62 + hist_strength * 28)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 92),
                reasoning=(
                    f"MACD ({m_curr:+.6f}) crossed below signal ({s_curr:+.6f}) — "
                    f"bearish momentum{'  · histogram expanding' if hist_shrinking else ''}"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # No crossover — report current state
        side = "bullish" if m_curr > s_curr else "bearish"
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"MACD {side}, no fresh crossover this candle",
            risk_level="LOW",
            observations=obs,
        )
