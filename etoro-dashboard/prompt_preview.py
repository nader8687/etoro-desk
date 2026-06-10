"""Mirror of visual-bot/signal_engine.py prompts — keep in sync for dashboard preview."""

from __future__ import annotations

from typing import Optional

SYSTEM_PROMPT = (
    "You are a professional short-term trading signal evaluator. "
    "Analyze candlestick chart images and decide what to do NOW based on the last "
    "visible candle only. Do not give hindsight trades except as context for the "
    "current decision. Always respond with valid JSON only — no markdown, no text "
    "outside the JSON."
)

CRYPTO_KEYWORDS = ("XRP", "BTC", "ETH", "DOGE", "SOL", "ADA", "CRYPTO", "LTC", "BNB")


def default_spread_pct(asset: str) -> float:
    upper = asset.upper()
    if any(k in upper for k in CRYPTO_KEYWORDS):
        return 1.0
    return 0.1


def build_trading_eval_prompt(
    asset: str,
    timeframe: str,
    current_price: Optional[float],
    spread_pct: float,
    position_type: str = "NONE",
    entry_price: Optional[float] = None,
) -> str:
    pos = position_type.upper()
    if pos not in ("NONE", "LONG", "SHORT"):
        pos = "NONE"
    entry_val = "null" if entry_price is None else f"{entry_price:.5f}"
    price_val = "unknown" if current_price is None else f"{current_price:.5f}"

    return f"""Analyze the attached candlestick chart image and decide what to do NOW based on the last visible candle only. Do not give hindsight trades except as context for the current decision.

Asset: {asset}
Platform: eToro
Timeframe: {timeframe}
Current price: {price_val}
Estimated spread/fees percent: {spread_pct:.2f}

Current position:
position_type: {pos}
entry_price: {entry_val}

Your allowed final actions are only:

BUY_LONG
SELL_SHORT
HOLD_NO_POSITION
CLOSE_LONG
HOLD_LONG
CLOSE_SHORT
HOLD_SHORT

Analyze:
- current trend direction
- recent momentum
- support and resistance
- breakout or breakdown
- rejection candles
- reversal risk
- whether price is extended
- whether the expected move is large enough after spread/fees

Decision rules:

If position_type is NONE:
- Choose BUY_LONG only if upside potential is clearly larger than spread/fees and momentum supports continuation.
- Choose SELL_SHORT only if downside potential is clearly larger than spread/fees and momentum supports continuation.
- If the signal is weak, late, unclear, or too close to support/resistance, choose HOLD_NO_POSITION.

If position_type is LONG (you profit when price goes UP):
- The spread was already paid at entry. A position near entry is in the spread recovery zone — that is NORMAL.
- HOLD_LONG if upward momentum is intact or loss is within 2× entry spread.
- CLOSE_LONG ONLY on a clear bearish reversal with momentum, or after spread is recovered AND price is profitable.
- NEVER CLOSE_LONG if unrealised loss is within 2× entry spread — you MUST choose HOLD_LONG.

If position_type is SHORT (you profit when price goes DOWN):
- The spread was already paid at entry. A position near entry is in the spread recovery zone — that is NORMAL.
- HOLD_SHORT if downward momentum is intact or loss is within 2× entry spread.
- CLOSE_SHORT ONLY on a clear bullish reversal with momentum, or after spread is recovered AND price is profitable.
- NEVER CLOSE_SHORT if unrealised loss is within 2× entry spread — you MUST choose HOLD_SHORT.

Important:
- Never force a trade.
- If uncertain, choose HOLD_NO_POSITION for no position.
- If already profitable after spread and a CLEAR reversal is visible, you may prefer closing.
- Take spread/fees seriously; small moves are not enough to enter — but a small loss within 2× spread is normal.
- Never claim guaranteed profit.
- This is analysis only, not financial advice.

Return JSON only in this exact format:

{{
  "current_signal": "BUY_LONG | SELL_SHORT | HOLD_NO_POSITION | CLOSE_LONG | HOLD_LONG | CLOSE_SHORT | HOLD_SHORT",
  "confidence": 0,
  "position_type": "NONE | LONG | SHORT",
  "entry_price": null,
  "current_price": null,
  "profitable_before_spread": null,
  "profitable_after_spread": null,
  "nearest_support": "approximate price or unknown",
  "nearest_resistance": "approximate price or unknown",
  "expected_direction_next": "UP | DOWN | SIDEWAYS | UNCLEAR",
  "spread_impact": "brief explanation",
  "reason": "brief explanation",
  "risk_warning": "brief warning"
}}"""
