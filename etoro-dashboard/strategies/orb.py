"""Opening Range Breakout (ORB) strategy.

The opening range is defined as the first N candles of the current UTC trading
session (midnight boundary).  If fewer than N candles exist for today, the
strategy falls back to the first N candles of the loaded history window.

Breakout logic
--------------
  BUY  — current close > opening-range high
  SELL — current close < opening-range low
  Confidence scales with the distance broken past the range as a % of range size.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class ORBStrategy(Strategy):
    key          = "orb"
    display_name = "Opening Range Breakout"
    description  = (
        "Defines an opening range from the first N candles of the current UTC day, "
        "then trades breakouts above the range high (BUY) or below the range low (SELL)."
    )

    def __init__(self, opening_candles: int = 15) -> None:
        self.opening_candles = opening_candles

    def _opening_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the opening-range slice.

        Prefer the first N candles of the current UTC day.  Fall back to the
        first N candles of the full history window if today has too few candles.
        """
        try:
            today = datetime.now(timezone.utc).date()
            times = df["time"]
            # Normalise timezone-naive timestamps to UTC for comparison
            if hasattr(times.iloc[0], "tzinfo") and times.iloc[0].tzinfo is None:
                times = pd.to_datetime(times, utc=True)
            else:
                times = pd.to_datetime(times, utc=True)
            today_mask = times.dt.date == today
            today_df   = df[today_mask]
            if len(today_df) >= 3:  # at least a few candles from today
                return today_df.head(self.opening_candles)
        except Exception:
            pass
        # Fallback: oldest N candles in the window
        return df.head(self.opening_candles)

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < self.opening_candles + 2:
            return None

        opening    = self._opening_range(df)
        if len(opening) < 3:
            return None

        orb_high   = float(opening["High"].max())
        orb_low    = float(opening["Low"].min())
        current    = float(df.iloc[-1]["Close"])
        range_size = orb_high - orb_low

        if range_size <= 0:
            return None

        obs = [
            f"ORB high: {orb_high:.5f}",
            f"ORB low:  {orb_low:.5f}",
            f"ORB window: {len(opening)} candles (current session)",
        ]

        if current > orb_high:
            pct_above  = (current - orb_high) / range_size * 100
            confidence = min(int(55 + pct_above * 4), 92)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Price {current:.5f} broke above ORB high {orb_high:.5f} "
                    f"(+{pct_above:.1f}% of range)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if current < orb_low:
            pct_below  = (orb_low - current) / range_size * 100
            confidence = min(int(55 + pct_below * 4), 92)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Price {current:.5f} broke below ORB low {orb_low:.5f} "
                    f"(-{pct_below:.1f}% of range)"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        pct_pos = (current - orb_low) / range_size * 100
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=(
                f"Price {current:.5f} inside opening range "
                f"({pct_pos:.0f}% from low)"
            ),
            risk_level="LOW",
            observations=obs,
        )
