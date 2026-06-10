"""Rate Arbitrage strategy.

Trades when the instrument's short-term rate of change (momentum) diverges
significantly from its own historical distribution — i.e. when price is moving
much faster or slower than it typically does, suggesting mean-reversion.

Additionally, if a second instrument is running, compares momentum rates between
the two to exploit relative-rate divergences (cross-instrument rate arbitrage).
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal

log = logging.getLogger(__name__)


class RateArbStrategy(Strategy):
    key          = "rate_arb"
    display_name = "Rate Arbitrage"
    description  = (
        "Compares short-term momentum (fast ROC) to its own historical "
        "distribution.  Sells when rate is overextended upward, buys when "
        "overextended downward.  Uses cross-instrument rate comparison when "
        "a second engine is running."
    )

    def __init__(
        self,
        fast_window: int  = 5,
        slow_window: int  = 20,
        threshold:   float = 2.0,
    ) -> None:
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.threshold   = threshold

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        need = self.slow_window + self.fast_window + 2
        if len(df) < need:
            return None

        closes = df["Close"].astype(float)

        # Rate-of-change series
        fast_roc = closes.pct_change(self.fast_window) * 100
        slow_roc = closes.pct_change(self.slow_window) * 100

        # Statistics over the rolling distribution
        window_roc  = fast_roc.iloc[-self.slow_window:]
        curr_fast   = float(fast_roc.iloc[-1])
        mean_fast   = float(window_roc.mean())
        std_fast    = float(window_roc.std()) + 1e-10
        z_self      = (curr_fast - mean_fast) / std_fast

        curr_slow   = float(slow_roc.iloc[-1])
        rate_div    = curr_fast - curr_slow  # positive → fast > slow → accelerating

        # ── Cross-instrument rate comparison (optional) ────────────────────────
        cross_obs: list[str] = []
        z_cross   = 0.0
        try:
            import market_data_hub
            all_snaps = market_data_hub.get_all_snapshots()
            ref_iid   = next((iid for iid in all_snaps if iid != instrument_id), None)
            if ref_iid is not None:
                ref_df      = all_snaps[ref_iid].chart_data
                if len(ref_df) >= need:
                    ref_closes  = ref_df["Close"].astype(float)
                    ref_roc     = ref_closes.pct_change(self.fast_window) * 100
                    ref_curr    = float(ref_roc.iloc[-1])
                    rate_spread = curr_fast - ref_curr
                    # The reference can be on a DIFFERENT interval/instrument, so
                    # aligning fast_roc - ref_roc by index yields NaN wherever the
                    # candle timestamps don't line up.  Drop those before taking
                    # mean/std; if too few overlap, skip the cross-Z (don't emit
                    # NaN, which would otherwise poison the final Z below).
                    ref_spreads = (fast_roc - ref_roc).iloc[-self.slow_window:].dropna()
                    std_cross   = float(ref_spreads.std()) if len(ref_spreads) >= 2 else float("nan")
                    if math.isfinite(rate_spread) and math.isfinite(std_cross) and std_cross > 0:
                        z_cross   = (rate_spread - float(ref_spreads.mean())) / std_cross
                        cross_obs = [f"Rate vs ref ({ref_iid}): {rate_spread:+.3f}%  Z={z_cross:.2f}"]
                    else:
                        cross_obs = [f"Rate vs ref ({ref_iid}): {rate_spread:+.3f}%  Z=n/a (candles don't align)"]
        except Exception as exc:
            log.warning("Cross-instrument rate comparison failed: %s", exc)

        # Signal off the larger-magnitude Z, ignoring a non-finite cross-Z so a
        # valid self-Z is never poisoned by a misaligned reference.
        z = z_self
        if math.isfinite(z_cross) and abs(z_cross) > abs(z_self):
            z = z_cross

        obs = [
            f"Fast ROC({self.fast_window}): {curr_fast:+.3f}%",
            f"Slow ROC({self.slow_window}): {curr_slow:+.3f}%",
            f"Self Z-score: {z_self:.2f}  (threshold ±{self.threshold})",
            *cross_obs,
        ]

        if z > self.threshold and rate_div > 0:
            confidence = min(int(52 + abs(z) * 12), 88)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Rate overextended upward: fast ROC {curr_fast:+.3f}% "
                    f"vs mean {mean_fast:+.3f}% (Z={z:.2f}) — mean-reversion expected"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if z < -self.threshold and rate_div < 0:
            confidence = min(int(52 + abs(z) * 12), 88)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Rate overextended downward: fast ROC {curr_fast:+.3f}% "
                    f"vs mean {mean_fast:+.3f}% (Z={z:.2f}) — mean-reversion expected"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        return StrategySignal(
            signal="HOLD",
            confidence=45,
            reasoning=f"Rate within normal range (Z={z:.2f})",
            risk_level="LOW",
            observations=obs,
        )
