"""Larry Connors RSI-2 mean-reversion strategy.

The most-documented short-term mean-reversion edge in the literature
(Connors & Alvarez, "Short Term Trading Strategies That Work"): stocks and
liquid crypto snap back after 1-2 bar panics.  A LONG 200-bar SMA keeps every
trade on the side of the dominant trend — fading panics WITH the trend, never
against it.

  • Trend filter — close above SMA(200) = longs only, below = shorts only.
  • Trigger     — RSI(2) < 10 in an uptrend → BUY the panic dip.
                  RSI(2) > 90 in a downtrend → SELL the euphoric pop.
  • Exits       — the engine's ATR stop / chandelier trail / class take-profit
                  (mean-revert profile banks the bounce quickly).

Uses Wilder smoothing for RSI, same convention as the rest of the fleet.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Strategy, StrategySignal


def _rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


class RSI2Strategy(Strategy):
    key          = "rsi2"
    display_name = "RSI-2 Connors (200-SMA filter)"
    description  = (
        "Buys 1-2 bar panic dips in an uptrend (RSI(2) < 10 above the 200-SMA) "
        "and sells euphoric pops in a downtrend (RSI(2) > 90 below it).  The "
        "classic Connors mean-reversion edge, always trading WITH the long-term "
        "trend."
    )

    def __init__(
        self,
        rsi_period: int  = 2,
        sma_period: int  = 200,
        buy_below: float = 10.0,
        sell_above: float = 90.0,
        deep_buy: float  = 5.0,
        deep_sell: float = 95.0,
    ) -> None:
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.buy_below  = buy_below
        self.sell_above = sell_above
        self.deep_buy   = deep_buy
        self.deep_sell  = deep_sell

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        if len(df) < self.sma_period + 10:
            return None

        closes = df["Close"].astype(float)
        sma = float(closes.rolling(self.sma_period).mean().iloc[-1])
        price = float(closes.iloc[-1])
        uptrend = price > sma

        rsi = _rsi(closes, self.rsi_period)
        r_curr, r_prev = float(rsi.iloc[-1]), float(rsi.iloc[-2])

        obs = [
            f"SMA({self.sma_period}): {sma:.5f} — price {'ABOVE (longs only)' if uptrend else 'BELOW (shorts only)'}",
            f"RSI({self.rsi_period}): {r_prev:.1f} → {r_curr:.1f}",
        ]

        # Fire on ENTERING the extreme zone (not every bar inside it) so one
        # panic produces one signal, not a stream of them.
        entered_oversold   = r_curr < self.buy_below  and r_prev >= self.buy_below
        entered_overbought = r_curr > self.sell_above and r_prev <= self.sell_above

        if uptrend and entered_oversold:
            deep = r_curr < self.deep_buy
            confidence = 64 + (16 if deep else 0) + min(int((self.buy_below - r_curr)), 10)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 90),
                reasoning=(
                    f"Connors RSI-2 panic dip: RSI(2) {r_curr:.1f} < {self.buy_below:.0f}"
                    f"{' (DEEP <' + str(self.deep_buy) + ')' if deep else ''} with price above "
                    f"the {self.sma_period}-SMA — buying the dip WITH the uptrend"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if not uptrend and entered_overbought:
            deep = r_curr > self.deep_sell
            confidence = 64 + (16 if deep else 0) + min(int(r_curr - self.sell_above), 10)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 90),
                reasoning=(
                    f"Connors RSI-2 euphoric pop: RSI(2) {r_curr:.1f} > {self.sell_above:.0f}"
                    f"{' (DEEP >' + str(self.deep_sell) + ')' if deep else ''} with price below "
                    f"the {self.sma_period}-SMA — selling the pop WITH the downtrend"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        side = "long" if uptrend else "short"
        zone = ("oversold" if r_curr < self.buy_below else
                "overbought" if r_curr > self.sell_above else "neutral")
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"Trend allows {side}s; RSI(2) {r_curr:.1f} ({zone}) — no fresh zone entry",
            risk_level="LOW",
            observations=obs,
        )
