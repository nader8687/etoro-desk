"""TTM Squeeze (John Carter) — volatility compression → directional release.

Canonical thinkorswim construction:
  • Bollinger Bands(20, 2σ) INSIDE Keltner Channels(20, 1.5×ATR) = squeeze ON
    (volatility compressed — the market is coiling).
  • Squeeze FIRES on the first bar the bands re-expand outside the channels.
  • Direction comes from the momentum histogram: a 20-bar least-squares
    regression of close − midline, where midline is the average of the
    Donchian midpoint and the 20-SMA.

Entry only on the FIRING bar (Carter's "first green dot after red dots"),
in the direction of the momentum histogram.  Exits belong to the engine's
ATR ladder (trend profile — ride the release with the chandelier trail).

Differs from the fleet's `bollinger_squeeze` (width-percentile fade): this is
a BREAKOUT strategy that waits for compression to resolve, not a reversion.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import Strategy, StrategySignal


class TTMSqueezeStrategy(Strategy):
    key          = "ttm_squeeze"
    display_name = "TTM Squeeze (20, 2σ/1.5ATR)"
    description  = (
        "Detects volatility compression (Bollinger Bands inside Keltner "
        "Channels) and enters on the release bar in the direction of the "
        "momentum histogram.  John Carter's classic coil-and-break system."
    )

    def __init__(
        self,
        period: int        = 20,
        bb_std: float      = 2.0,
        kc_mult: float     = 1.5,
        min_squeeze: int   = 3,
    ) -> None:
        self.period      = period
        self.bb_std      = bb_std
        self.kc_mult     = kc_mult
        self.min_squeeze = min_squeeze   # bars of compression required before a fire counts

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        n = self.period
        if len(df) < 2 * n + 10:
            return None

        closes = df["Close"].astype(float)
        highs  = df["High"].astype(float)
        lows   = df["Low"].astype(float)

        sma = closes.rolling(n).mean()
        std = closes.rolling(n).std(ddof=0)
        bb_up, bb_dn = sma + self.bb_std * std, sma - self.bb_std * std

        pc = closes.shift(1)
        tr = pd.concat([highs - lows, (highs - pc).abs(), (lows - pc).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
        ema = closes.ewm(span=n, adjust=False).mean()
        kc_up, kc_dn = ema + self.kc_mult * atr, ema - self.kc_mult * atr

        squeeze_on = (bb_up < kc_up) & (bb_dn > kc_dn)
        on_curr = bool(squeeze_on.iloc[-1])
        on_prev = bool(squeeze_on.iloc[-2])
        # How long the squeeze lasted before this bar
        run = 0
        for v in squeeze_on.iloc[-2::-1]:
            if not v:
                break
            run += 1

        # Momentum: 20-bar linear regression of close − midline (ToS formula)
        donch_mid = (highs.rolling(n).max() + lows.rolling(n).min()) / 2.0
        midline = (donch_mid + sma) / 2.0
        delta = (closes - midline).iloc[-n:].to_numpy(dtype=float)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, delta, 1)
        momo = float(slope * (n - 1) + intercept)         # regression value at the current bar
        momo_prev = float(slope * (n - 2) + intercept)
        rising = momo > momo_prev

        fired = on_prev and not on_curr and run >= self.min_squeeze

        obs = [
            f"Squeeze: {'ON (compressing, ' + str(run + 1) + ' bars)' if on_curr else ('FIRED after ' + str(run) + ' bars' if fired else 'off')}",
            f"BB({n},{self.bb_std}σ) width: {float(bb_up.iloc[-1] - bb_dn.iloc[-1]):.5f} · "
            f"KC({n},{self.kc_mult}×ATR) width: {float(kc_up.iloc[-1] - kc_dn.iloc[-1]):.5f}",
            f"Momentum: {momo:+.5f} ({'rising' if rising else 'falling'})",
        ]

        if fired and momo > 0:
            confidence = 62 + min(run * 2, 14) + (14 if rising else 0)
            return StrategySignal(
                signal="BUY",
                confidence=min(confidence, 90),
                reasoning=(
                    f"TTM squeeze FIRED after {run} bars of compression with bullish "
                    f"momentum ({momo:+.5f}{', rising' if rising else ''}) — "
                    "coiled energy releasing upward"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        if fired and momo < 0:
            confidence = 62 + min(run * 2, 14) + (14 if not rising else 0)
            return StrategySignal(
                signal="SELL",
                confidence=min(confidence, 90),
                reasoning=(
                    f"TTM squeeze FIRED after {run} bars of compression with bearish "
                    f"momentum ({momo:+.5f}{', falling' if not rising else ''}) — "
                    "coiled energy releasing downward"
                ),
                risk_level="MEDIUM",
                observations=obs,
            )

        state = (f"squeeze ON ({run + 1} bars) — waiting for the release" if on_curr
                 else "no squeeze — bands outside channels")
        return StrategySignal(
            signal="HOLD",
            confidence=40,
            reasoning=f"TTM: {state}; momentum {momo:+.5f}",
            risk_level="LOW",
            observations=obs,
        )
