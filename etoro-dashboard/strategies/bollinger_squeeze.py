"""Bollinger Band Squeeze strategy.

Detects periods of low volatility (squeeze) and trades the subsequent
momentum breakout.

Squeeze detection
-----------------
The current BB width is compared to the 20th-percentile width over the
lookback window.  Using a percentile (rather than the absolute minimum)
makes the detector robust against outlier-narrow candles that would
otherwise trigger a permanent "squeeze" state.

  BUY  — price closes above upper band AND in squeeze
  SELL — price closes below lower band AND in squeeze
  HOLD — in squeeze but no break yet, or not in squeeze
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


class BollingerSqueezeStrategy(Strategy):
    key          = "bollinger_squeeze"
    display_name = "Bollinger Band Squeeze"
    description  = (
        "Detects low-volatility squeezes (BB width ≤ 20th pct of recent widths) "
        "and trades breakouts: BUY on upper-band break, SELL on lower-band break."
    )

    def __init__(
        self,
        period: int   = 20,
        std_dev: float = 2.0,
        squeeze_lookback: int = 10,
        squeeze_pct: float = 0.20,   # percentile threshold for squeeze detection
    ) -> None:
        self.period           = period
        self.std_dev          = std_dev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_pct      = squeeze_pct

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        min_len = self.period + self.squeeze_lookback + 2
        if len(df) < min_len:
            return None

        closes = df["Close"].astype(float)
        sma    = closes.rolling(self.period).mean()
        std    = closes.rolling(self.period).std()
        upper  = sma + self.std_dev * std
        lower  = sma - self.std_dev * std
        width  = upper - lower

        curr_upper = float(upper.iloc[-1])
        curr_lower = float(lower.iloc[-1])
        curr_width = float(width.iloc[-1])
        curr_close = float(closes.iloc[-1])
        curr_sma   = float(sma.iloc[-1])

        hist_widths = width.iloc[-(self.squeeze_lookback + 1):-1].dropna()
        if hist_widths.empty:
            return None

        # Squeeze = current width is at or below the Nth percentile of recent widths
        squeeze_threshold = float(hist_widths.quantile(self.squeeze_pct))
        in_squeeze        = curr_width <= squeeze_threshold
        breaking_up       = curr_close > curr_upper
        breaking_down     = curr_close < curr_lower

        obs = [
            f"BB upper: {curr_upper:.5f}",
            f"BB lower: {curr_lower:.5f}",
            f"BB width: {curr_width:.5f}  (p{int(self.squeeze_pct*100)} threshold: {squeeze_threshold:.5f})",
            f"Squeeze: {'YES' if in_squeeze else 'NO'}",
        ]

        if breaking_up and in_squeeze:
            excess     = curr_close - curr_upper
            raw_conf   = int(65 + excess / max(curr_width, 1e-10) * 100)
            confidence = max(min(raw_conf, 92), 65)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=(
                    f"Breakout above upper BB ({curr_upper:.5f}) after squeeze — "
                    f"momentum expanding upward"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if breaking_down and in_squeeze:
            excess     = curr_lower - curr_close
            raw_conf   = int(65 + excess / max(curr_width, 1e-10) * 100)
            confidence = max(min(raw_conf, 92), 65)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=(
                    f"Breakout below lower BB ({curr_lower:.5f}) after squeeze — "
                    f"momentum expanding downward"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if in_squeeze:
            return StrategySignal(
                signal="HOLD",
                confidence=40,
                reasoning="Bollinger squeeze — waiting for directional breakout",
                risk_level="LOW",
                observations=obs,
            )

        # No squeeze — price relative to mid-band
        side = "above" if curr_close > curr_sma else "below"
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"No squeeze — price {side} mid-band, no breakout setup",
            risk_level="LOW",
            observations=obs,
        )
