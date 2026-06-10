"""Candlestick Pattern Recognition strategy.

The only strategy in the set with zero indicators — purely price action.
Scans the last three candles for named patterns and scores them.

Patterns and their bullish (+) / bearish (−) scores
----------------------------------------------------
  Three White Soldiers   +40   Three Black Crows     −40
  Morning Star           +35   Evening Star          −35
  Bullish Engulfing      +30   Bearish Engulfing     −30
  Hammer                 +20   Shooting Star         −20
  Bullish Pin Bar        +15   Bearish Pin Bar       −15
  Doji (context-free)    ± 5   (sign = trend direction)

Signal fires when |net score| ≥ 20.
Confidence = min(50 + |net_score|, 92).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


def _body(o: float, c: float) -> float:
    return abs(c - o)

def _range(h: float, l: float) -> float:
    return h - l if h > l else 1e-10

def _upper_shadow(o: float, h: float, c: float) -> float:
    return h - max(o, c)

def _lower_shadow(o: float, l: float, c: float) -> float:
    return min(o, c) - l


class CandlestickStrategy(Strategy):
    key          = "candlestick"
    display_name = "Candlestick Patterns"
    description  = (
        "Scans the last three candles for classic patterns (engulfing, hammer, "
        "morning/evening star, soldiers/crows, pin bars, doji) and fires on net score."
    )

    def __init__(self, score_threshold: int = 20) -> None:
        self.score_threshold = score_threshold

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < 4:
            return None

        # Grab last 3 candles
        c1 = df.iloc[-3]   # oldest of the three
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]   # most recent

        o1, h1, l1, cl1 = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])
        o2, h2, l2, cl2 = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])
        o3, h3, l3, cl3 = float(c3["Open"]), float(c3["High"]), float(c3["Low"]), float(c3["Close"])

        body1, body2, body3 = _body(o1, cl1), _body(o2, cl2), _body(o3, cl3)
        rng1,  rng2,  rng3  = _range(h1, l1), _range(h2, l2), _range(h3, l3)

        bull1 = cl1 > o1
        bull2 = cl2 > o2
        bull3 = cl3 > o3

        score: int = 0
        matched: list[str] = []

        # ── 3-candle patterns ─────────────────────────────────────────────────

        # Three White Soldiers: 3 consecutive strong bullish candles, each closes higher
        if (bull1 and bull2 and bull3
                and cl3 > cl2 > cl1
                and body1 > rng1 * 0.5
                and body2 > rng2 * 0.5
                and body3 > rng3 * 0.5):
            score += 40
            matched.append("Three White Soldiers (+40)")

        # Three Black Crows: 3 consecutive strong bearish candles, each closes lower
        if (not bull1 and not bull2 and not bull3
                and cl3 < cl2 < cl1
                and body1 > rng1 * 0.5
                and body2 > rng2 * 0.5
                and body3 > rng3 * 0.5):
            score -= 40
            matched.append("Three Black Crows (−40)")

        # Morning Star: bearish c1, small-body c2 (indecision), bullish c3
        small2 = body2 < rng2 * 0.3
        if (not bull1 and body1 > rng1 * 0.4
                and small2
                and bull3 and body3 > rng3 * 0.4
                and cl3 > (o1 + cl1) / 2):   # closes above c1 midpoint
            score += 35
            matched.append("Morning Star (+35)")

        # Evening Star: bullish c1, small-body c2, bearish c3
        if (bull1 and body1 > rng1 * 0.4
                and small2
                and not bull3 and body3 > rng3 * 0.4
                and cl3 < (o1 + cl1) / 2):   # closes below c1 midpoint
            score -= 35
            matched.append("Evening Star (−35)")

        # ── 2-candle patterns (c2 vs c3) ─────────────────────────────────────

        # Bullish Engulfing: c2 bearish, c3 bullish, c3 body fully wraps c2 body
        if (not bull2 and bull3
                and o3 < cl2 and cl3 > o2):
            score += 30
            matched.append("Bullish Engulfing (+30)")

        # Bearish Engulfing: c2 bullish, c3 bearish, c3 body fully wraps c2 body
        if (bull2 and not bull3
                and o3 > cl2 and cl3 < o2):
            score -= 30
            matched.append("Bearish Engulfing (−30)")

        # ── Single-candle patterns (c3 only) ─────────────────────────────────

        up3    = _upper_shadow(o3, h3, cl3)
        lo3    = _lower_shadow(o3, l3, cl3)

        # Hammer: long lower shadow (≥2× body), tiny upper shadow
        if (lo3 >= body3 * 2 and up3 < body3 * 0.5 and rng3 > 0):
            score += 20
            matched.append("Hammer (+20)")

        # Shooting Star: long upper shadow (≥2× body), tiny lower shadow
        if (up3 >= body3 * 2 and lo3 < body3 * 0.5 and rng3 > 0):
            score -= 20
            matched.append("Shooting Star (−20)")

        # Bullish Pin Bar: close in upper third of range, long lower tail
        if (lo3 > rng3 * 0.55 and cl3 > h3 - rng3 * 0.35):
            score += 15
            matched.append("Bullish Pin Bar (+15)")

        # Bearish Pin Bar: close in lower third of range, long upper tail
        if (up3 > rng3 * 0.55 and cl3 < l3 + rng3 * 0.35):
            score -= 15
            matched.append("Bearish Pin Bar (−15)")

        # Doji: very small body relative to range — bias from trend
        if body3 < rng3 * 0.1:
            # 5-candle trend for context
            if len(df) >= 6:
                trend_chg = float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-6])
                score += 5 if trend_chg < 0 else -5  # bullish doji after down, bearish after up
            matched.append("Doji (±5)")

        # ── Build signal ──────────────────────────────────────────────────────
        obs = [f"Net score: {score:+d}  (threshold ±{self.score_threshold})"]
        obs += matched or ["No patterns matched"]

        if abs(score) < self.score_threshold:
            return StrategySignal(
                signal="HOLD",
                confidence=40,
                reasoning=f"Pattern score {score:+d} below threshold — no actionable setup",
                risk_level="LOW",
                observations=obs,
            )

        confidence = min(50 + abs(score), 92)

        if score > 0:
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=f"Bullish price-action patterns (score {score:+d}): {matched[0]}",
                risk_level="MEDIUM",
                observations=obs,
            )

        return StrategySignal(
            signal="SELL",
            confidence=confidence,
            reasoning=f"Bearish price-action patterns (score {score:+d}): {matched[0]}",
            risk_level="MEDIUM",
            observations=obs,
        )
