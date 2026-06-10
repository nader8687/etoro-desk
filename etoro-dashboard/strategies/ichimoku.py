"""Ichimoku Cloud strategy.

Ichimoku Kinko Hyo is a complete self-contained trading system.  This
implementation scores the four actionable conditions and fires signals
when enough of them align.

Components
----------
  Tenkan-sen  (9)   conversion line: (9-period high + low) / 2
  Kijun-sen   (26)  base line:       (26-period high + low) / 2
  Senkou A          (Tenkan + Kijun) / 2,  projected 26 bars forward
  Senkou B   (52)   (52-period high + low) / 2, projected 26 bars forward
  Chikou           current close, plotted 26 bars back

For real-time analysis the "future" cloud values are read 26 bars back
in the historical series (equivalent to the current projected cloud).

Scoring (bullish / bearish mirror)
-----------------------------------
  +25 / −25   price above / below cloud
  +20 / −20   cloud bullish (A > B) / bearish (A < B)
  +20 / −20   Tenkan above / below Kijun
  +15 / −15   Chikou above / below price 26 bars ago

  BUY  threshold: ≥ 55   SELL threshold: ≤ −55   else HOLD
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


def _donchian_mid(highs: pd.Series, lows: pd.Series, period: int) -> pd.Series:
    return (highs.rolling(period).max() + lows.rolling(period).min()) / 2


class IchimokuStrategy(Strategy):
    key          = "ichimoku"
    display_name = "Ichimoku Cloud"
    description  = (
        "Four-component scoring: price vs cloud, cloud colour, TK cross, "
        "Chikou confirmation.  Fires when enough conditions align."
    )

    def __init__(
        self,
        tenkan:   int = 9,
        kijun:    int = 26,
        senkou_b: int = 52,
        displacement: int = 26,
        buy_threshold:  int = 55,
        sell_threshold: int = -55,
    ) -> None:
        self.tenkan        = tenkan
        self.kijun         = kijun
        self.senkou_b      = senkou_b
        self.displacement  = displacement
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        # Need enough candles for Senkou B + displacement look-back
        need = self.senkou_b + self.displacement + 2
        if len(df) < need:
            return None

        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)
        closes = df["Close"].astype(float)

        tenkan_ser  = _donchian_mid(highs, lows, self.tenkan)
        kijun_ser   = _donchian_mid(highs, lows, self.kijun)
        span_a_ser  = (tenkan_ser + kijun_ser) / 2
        span_b_ser  = _donchian_mid(highs, lows, self.senkou_b)

        # Current values
        tenkan_now = float(tenkan_ser.iloc[-1])
        kijun_now  = float(kijun_ser.iloc[-1])
        close_now  = float(closes.iloc[-1])

        # Cloud at current bar = values from `displacement` bars ago
        d = self.displacement
        span_a_now = float(span_a_ser.iloc[-d]) if len(span_a_ser) >= d else float("nan")
        span_b_now = float(span_b_ser.iloc[-d]) if len(span_b_ser) >= d else float("nan")

        if any(pd.isna(v) for v in (tenkan_now, kijun_now, span_a_now, span_b_now)):
            return None

        cloud_top    = max(span_a_now, span_b_now)
        cloud_bottom = min(span_a_now, span_b_now)
        cloud_bull   = span_a_now > span_b_now   # green cloud

        # Chikou: close now vs close from `displacement` bars ago
        chikou_ref = float(closes.iloc[-d]) if len(closes) >= d else None

        # ── Scoring ──────────────────────────────────────────────────────────
        score   = 0
        reasons = []

        # 1. Price vs cloud (+25 / −25)
        if close_now > cloud_top:
            score += 25
            reasons.append(f"Price above cloud ({cloud_top:.5f})")
        elif close_now < cloud_bottom:
            score -= 25
            reasons.append(f"Price below cloud ({cloud_bottom:.5f})")
        else:
            reasons.append("Price inside cloud (neutral)")

        # 2. Cloud colour (+20 / −20)
        if cloud_bull:
            score += 20
            reasons.append(f"Bullish cloud (A {span_a_now:.5f} > B {span_b_now:.5f})")
        else:
            score -= 20
            reasons.append(f"Bearish cloud (A {span_a_now:.5f} < B {span_b_now:.5f})")

        # 3. Tenkan vs Kijun (+20 / −20)
        if tenkan_now > kijun_now:
            score += 20
            reasons.append(f"Tenkan ({tenkan_now:.5f}) above Kijun ({kijun_now:.5f})")
        else:
            score -= 20
            reasons.append(f"Tenkan ({tenkan_now:.5f}) below Kijun ({kijun_now:.5f})")

        # 4. Chikou (+15 / −15)
        if chikou_ref is not None:
            if close_now > chikou_ref:
                score += 15
                reasons.append(f"Chikou above price {d}b ago ({chikou_ref:.5f})")
            else:
                score -= 15
                reasons.append(f"Chikou below price {d}b ago ({chikou_ref:.5f})")

        obs = [f"Score: {score:+d}  (buy ≥{self.buy_threshold}, sell ≤{self.sell_threshold})"] + reasons

        if score >= self.buy_threshold:
            confidence = min(50 + score, 92)
            return StrategySignal(
                signal="BUY",
                confidence=confidence,
                reasoning=f"Ichimoku bullish confluence (score {score:+d}): {reasons[0]}",
                risk_level="LOW" if score >= 75 else "MEDIUM",
                observations=obs,
            )

        if score <= self.sell_threshold:
            confidence = min(50 + abs(score), 92)
            return StrategySignal(
                signal="SELL",
                confidence=confidence,
                reasoning=f"Ichimoku bearish confluence (score {score:+d}): {reasons[0]}",
                risk_level="LOW" if abs(score) >= 75 else "MEDIUM",
                observations=obs,
            )

        direction = "bullish" if score > 0 else "bearish"
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"Ichimoku {direction} but mixed ({score:+d}) — insufficient confluence",
            risk_level="LOW",
            observations=obs,
        )
