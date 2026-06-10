"""Statistical Arbitrage (Pairs Trading) strategy."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class StatArbStrategy(Strategy):
    key          = "stat_arb"
    display_name = "Statistical Arbitrage"
    description  = (
        "Pairs trading: computes a normalised spread between this instrument "
        "and the first other running instrument.  Trades mean-reversion when "
        "the spread Z-score exceeds ±2."
    )

    def __init__(self, z_threshold: float = 2.0, lookback: int = 30) -> None:
        self.z_threshold = z_threshold
        self.lookback    = lookback

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        # Import here to avoid circular imports at module load time
        import market_data_hub

        all_snaps = market_data_hub.get_all_snapshots()
        ref_iid   = next((iid for iid in all_snaps if iid != instrument_id), None)

        if ref_iid is None:
            return StrategySignal(
                signal="HOLD",
                confidence=30,
                reasoning="No reference instrument running — pairs trading unavailable",
                risk_level="LOW",
                observations=["Start a second instrument engine to enable pairs trading"],
            )

        ref_df = all_snaps[ref_iid].chart_data

        if len(df) < self.lookback or len(ref_df) < self.lookback:
            return None

        a = df["Close"].astype(float).values[-self.lookback:]
        b = ref_df["Close"].astype(float).values[-self.lookback:]

        # Normalise both series to zero-mean unit-variance
        a_norm = (a - a.mean()) / (a.std() + 1e-10)
        b_norm = (b - b.mean()) / (b.std() + 1e-10)

        spread = a_norm - b_norm
        z      = float((spread[-1] - spread.mean()) / (spread.std() + 1e-10))

        obs = [
            f"Spread Z-score: {z:.2f}  (threshold ±{self.z_threshold})",
            f"Reference iid:  {ref_iid}",
            f"Lookback:       {self.lookback} candles",
        ]

        if z > self.z_threshold:
            confidence = min(int(50 + abs(z) * 14), 90)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Z-score {z:.2f} — instrument overpriced vs reference "
                    f"(iid={ref_iid}).  Expect mean-reversion down."
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if z < -self.z_threshold:
            confidence = min(int(50 + abs(z) * 14), 90)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Z-score {z:.2f} — instrument underpriced vs reference "
                    f"(iid={ref_iid}).  Expect mean-reversion up."
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=45,
            reasoning=f"Z-score {z:.2f} within normal range — no pairs signal",
            risk_level="LOW",
            observations=obs,
        )
