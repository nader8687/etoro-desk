"""Supertrend strategy.

Computes the Supertrend indicator over the full OHLCV series, then signals
on the candle where the trend flips.  Nothing fires while the trend is
sustained — only the flip candle generates a BUY or SELL.

Algorithm
---------
    basic_upper[i] = HL2[i] + mult × ATR[i]
    basic_lower[i] = HL2[i] − mult × ATR[i]

    final_upper[i] = min(basic_upper[i], final_upper[i−1])
                     if close[i−1] ≤ final_upper[i−1] else basic_upper[i]

    final_lower[i] = max(basic_lower[i], final_lower[i−1])
                     if close[i−1] ≥ final_lower[i−1] else basic_lower[i]

    direction[i]: +1 (bullish) or −1 (bearish)
        flips to −1 when close[i] ≤ final_upper[i] (was bullish)
        flips to +1 when close[i] ≥ final_lower[i] (was bearish)
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class SupertrendStrategy(Strategy):
    key          = "supertrend"
    display_name = "Supertrend (10, 3)"
    description  = (
        "ATR-based trailing indicator.  Fires BUY when the trend flips bullish, "
        "SELL when it flips bearish.  Silent while trend is sustained."
    )

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0) -> None:
        self.atr_period = atr_period
        self.multiplier = multiplier

    # ── Internal: compute full Supertrend series ──────────────────────────────

    def _compute(self, df: pd.DataFrame) -> tuple[pd.Series, float]:
        """Return (direction Series of +1/−1, final ATR value).

        Both outputs are produced in a single pass so callers avoid the cost
        of a second ATR computation just for display purposes.
        """
        highs  = df["High"].astype(float).values
        lows   = df["Low"].astype(float).values
        closes = df["Close"].astype(float).values
        n      = len(closes)

        # Wilder ATR
        tr = pd.Series(
            [max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i]  - closes[i - 1]))
             if i > 0 else highs[i] - lows[i]
             for i in range(n)]
        )
        atr_series = tr.ewm(alpha=1 / self.atr_period, adjust=False).mean()
        atr        = atr_series.values

        hl2         = (highs + lows) / 2
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        direction   = [1] * n   # start bullish

        for i in range(1, n):
            # Ratchet upper band down (or reset)
            final_upper[i] = (
                min(basic_upper[i], final_upper[i - 1])
                if closes[i - 1] <= final_upper[i - 1]
                else basic_upper[i]
            )
            # Ratchet lower band up (or reset)
            final_lower[i] = (
                max(basic_lower[i], final_lower[i - 1])
                if closes[i - 1] >= final_lower[i - 1]
                else basic_lower[i]
            )
            # Determine direction
            prev_dir = direction[i - 1]
            if prev_dir == -1:                          # was bearish
                direction[i] = 1 if closes[i] > final_upper[i] else -1
            else:                                       # was bullish
                direction[i] = -1 if closes[i] < final_lower[i] else 1

        return pd.Series(direction, index=df.index), float(atr_series.iloc[-1])

    # ── Strategy interface ────────────────────────────────────────────────────

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < self.atr_period + 3:
            return None

        direction, atr = self._compute(df)   # single-pass — ATR reused, not recomputed
        curr_dir  = int(direction.iloc[-1])
        prev_dir  = int(direction.iloc[-2])

        close_curr = float(df["Close"].iloc[-1])
        atr_pct = atr / close_curr * 100 if close_curr else 0

        obs = [
            f"Direction: {'▲ bullish' if curr_dir == 1 else '▼ bearish'}",
            f"ATR: {atr:.5f}  ({atr_pct:.3f}%)",
        ]

        if prev_dir == -1 and curr_dir == 1:
            return StrategySignal(
                signal="BUY",
                confidence=78,
                reasoning=(
                    f"Supertrend flipped bullish at {close_curr:.5f} — "
                    f"ATR volatility {atr_pct:.3f}%"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if prev_dir == 1 and curr_dir == -1:
            return StrategySignal(
                signal="SELL",
                confidence=78,
                reasoning=(
                    f"Supertrend flipped bearish at {close_curr:.5f} — "
                    f"ATR volatility {atr_pct:.3f}%"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        # No flip — trend is sustained
        trend_txt = "bullish" if curr_dir == 1 else "bearish"
        return StrategySignal(
            signal="HOLD",
            confidence=45,
            reasoning=f"Supertrend {trend_txt} and sustained — no flip this candle",
            risk_level="LOW",
            observations=obs,
        )
