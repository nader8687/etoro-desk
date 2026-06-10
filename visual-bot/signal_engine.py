"""
Chart analysis engine — supports OpenAI and Groq (OpenAI-compatible API).

Set in .env:
  LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini
  LLM_PROVIDER=groq     GROQ_API_KEY=gsk_...    GROQ_MODEL=llama-3.2-90b-vision-preview
"""
import base64
import json
import logging
import re
from typing import Literal, Optional

from openai import OpenAI

log = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "groq":   "llama-3.2-90b-vision-preview",
}

SYSTEM_PROMPT = (
    "You are a disciplined short-term trading signal evaluator. Your performance is "
    "judged on NET P&L after costs, not on activity: a missed trade costs nothing, a "
    "bad trade costs real money. Most candles deserve no action — HOLD_NO_POSITION is "
    "the correct answer the majority of the time.\n"
    "Core discipline:\n"
    "1. Never enter without a defined price TARGET (next S/R level) and a defined "
    "INVALIDATION level (where the idea is proven wrong). If reward-to-risk between "
    "them is below 2:1 after spread, do not enter.\n"
    "2. Evaluate the bull case and the bear case with equal seriousness before "
    "deciding — do not default to buying.\n"
    "3. Never chase: if the move already happened (large extended candles away from "
    "any base), the edge is gone — wait.\n"
    "4. Confidence must be calibrated: 80+ means textbook multi-signal setup, "
    "60-79 means decent setup with a flaw, below 60 means no trade. If your "
    "confidence in an ENTRY is below 70, output HOLD_NO_POSITION instead.\n"
    "Analyze the chart image and decide what to do NOW based on the last visible "
    "candle only. Do not give hindsight trades except as context for the current "
    "decision. Always respond with valid JSON only — no markdown, no text outside "
    "the JSON."
)

ENTRY_SIGNAL_MAP = {
    "BUY_LONG": "BUY",
    "SELL_SHORT": "SELL",
    "HOLD_NO_POSITION": "HOLD",
}

EXIT_ACTION_MAP = {
    "CLOSE_LONG": "CLOSE",
    "HOLD_LONG": "HOLD",
    "CLOSE_SHORT": "CLOSE",
    "HOLD_SHORT": "HOLD",
}

CRYPTO_KEYWORDS = ("XRP", "BTC", "ETH", "DOGE", "SOL", "ADA", "CRYPTO", "LTC", "BNB")
COMMODITY_KEYWORDS = (
    "GOLD", "XAU", "SILVER", "XAG", "OIL", "WTI", "BRENT", "CRUDE",
    "NATGAS", "NATURAL GAS", "COPPER", "PLATINUM", "PALLADIUM",
    "WHEAT", "CORN", "SUGAR", "COCOA", "COFFEE",
)

ASSET_CLASS_CONTEXT = {
    "crypto": (
        "This is CRYPTO: 24/7 market, high volatility, frequent fake breakouts and "
        "stop-hunt wicks. Demand extra confirmation on breakouts; a single large candle "
        "is often exhaustion, not the start of a move."
    ),
    "commodity": (
        "This is a COMMODITY: moves in volatility bursts around session opens and "
        "inventory/macro news; respects multi-session support/resistance strongly. "
        "Prefer entries at well-tested levels, not mid-range."
    ),
    "stock": (
        "This is a STOCK/EQUITY: session-bound trading with lower per-bar volatility "
        "than crypto. Gaps and open/close auction periods distort candles; mid-session "
        "trends are cleaner. Smaller moves are meaningful — scale expectations down."
    ),
    "etf": (
        "This is an ETF: diversified basket — moves are smoother and smaller than "
        "single stocks; breakouts are rarer and trends more persistent. Scale "
        "expectations down and favour trend-following over reversal bets."
    ),
    "index": (
        "This is an INDEX: very smooth price action, strong mean-reversion intraday, "
        "respects round numbers and prior session highs/lows. Small moves are "
        "meaningful — demand clean levels."
    ),
    "forex": (
        "This is FOREX: lowest per-bar volatility, strong session rhythm "
        "(London/NY opens), respects round numbers heavily. Tiny moves matter; "
        "spread is a large fraction of the expected move — be very selective."
    ),
}


def asset_class(asset: str) -> str:
    upper = (asset or "").upper()
    if any(k in upper for k in CRYPTO_KEYWORDS):
        return "crypto"
    if any(k in upper for k in COMMODITY_KEYWORDS):
        return "commodity"
    return "stock"


def display_asset_name(instrument: str) -> str:
    """eToro labels look like 'Bank of America Corp  (BAC)' — return the name part."""
    text = (instrument or "").strip()
    if not text:
        return "Unknown"
    if "  (" in text and text.endswith(")"):
        return text.rsplit("  (", 1)[0].strip()
    return text


def default_spread_pct(asset: str) -> float:
    """Default estimated round-trip spread/fees percent by asset class."""
    upper = asset.upper()
    if any(k in upper for k in CRYPTO_KEYWORDS):
        return 1.0
    return 0.1


def _build_trading_eval_prompt(
    asset: str,
    timeframe: str,
    current_price: Optional[float],
    spread_pct: float,
    position_type: str = "NONE",
    entry_price: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    in_profit: Optional[bool] = None,
    in_spread_recovery_zone: Optional[bool] = None,
    peak_pnl_pct: Optional[float] = None,
    memory: Optional[str] = None,
    asset_class_hint: Optional[str] = None,
) -> str:
    pos = position_type.upper()
    if pos not in ("NONE", "LONG", "SHORT"):
        pos = "NONE"
    entry_val = "null" if entry_price is None else f"{entry_price:.5f}"
    price_val = "unknown" if current_price is None else f"{current_price:.5f}"

    # Build rich position context block for open trades
    position_block = f"position_type: {pos}\nentry_price: {entry_val}"
    pnl_line = ""

    if pos in ("LONG", "SHORT") and entry_price is not None and current_price is not None:
        direction_word = "LONG (buy)" if pos == "LONG" else "SHORT (sell)"
        profit_dir = "ABOVE" if pos == "LONG" else "BELOW"
        move_pct = (current_price - entry_price) / entry_price * 100
        move_sign = "+" if move_pct >= 0 else ""

        # Plain-English entry / current summary so the LLM knows exactly where we stand
        position_block = (
            f"You opened a {direction_word} at {entry_price:.5f}. "
            f"You profit when price goes {profit_dir} that level.\n"
            f"Current price: {current_price:.5f}  ({move_sign}{move_pct:.2f}% from your entry)"
        )

    if pos in ("LONG", "SHORT") and pnl_pct is not None:
        if in_profit:
            profit_status = "PROFITABLE — above entry after spread"
        elif in_spread_recovery_zone:
            profit_status = (
                "SPREAD RECOVERY ZONE — within 2× entry spread (this is normal; MUST HOLD, never CLOSE)"
            )
        else:
            profit_status = "loss — below spread recovery zone"
        pnl_line = f"\nCurrent P&L: {pnl_pct:+.2f}%  ({profit_status})"

        # Peak P&L and pullback distance — critical for normalization detection
        if peak_pnl_pct is not None and peak_pnl_pct > spread_pct:
            pullback = peak_pnl_pct - pnl_pct if pnl_pct is not None else 0.0
            pnl_line += f"\nBest profit reached this trade: {peak_pnl_pct:+.2f}%"
            if pullback > 0.1:
                pnl_line += (
                    f"  → price has since pulled back {pullback:.2f}% from that peak"
                    f" (normalization in progress)"
                )

    if in_spread_recovery_zone:
        pnl_line += (
            "\n*** MANDATORY: loss is within 2× entry spread — choose HOLD_LONG or HOLD_SHORT only. "
            "CLOSE_LONG / CLOSE_SHORT is FORBIDDEN. ***"
        )

    # Determine if we are in "good profit" territory for the spike-normalization rules
    good_profit = (
        in_profit
        and peak_pnl_pct is not None
        and peak_pnl_pct >= spread_pct * 2
    )

    memory_block = ""
    if memory:
        memory_block = (
            "\nYour recent track record on this asset (learn from it — this is "
            "your own past performance, not a prediction):\n"
            f"{memory}\n"
        )

    # Prefer the authoritative class from eToro metadata (passed by the caller);
    # keyword detection is only the fallback.
    klass = (asset_class_hint or "").strip().lower()
    if klass not in ASSET_CLASS_CONTEXT:
        klass = asset_class(asset)
    class_context = ASSET_CLASS_CONTEXT.get(klass, "")

    return f"""Analyze the attached candlestick chart image and decide what to do NOW based on the last visible candle only. Do not give hindsight trades except as context for the current decision.

Asset: {asset}  (asset class: {klass})
{class_context}
Platform: eToro
Timeframe: {timeframe}
Current market price: {price_val}
Estimated spread/fees percent: {spread_pct:.2f}%
{memory_block}
Current position:
{position_block}{pnl_line}

Your allowed final actions are only:

BUY_LONG
SELL_SHORT
HOLD_NO_POSITION
CLOSE_LONG
HOLD_LONG
CLOSE_SHORT
HOLD_SHORT

Analyze IN THIS ORDER (reason through each step before deciding):
1. Trend: direction and structure (higher highs/lows or lower highs/lows?).
2. Momentum: accelerating or fading? Is the current move extended (several
   consecutive large candles away from a base = late, do not chase)?
3. Levels: nearest support and resistance — these define your target and
   invalidation prices.
4. Candles: rejection signals (wicks, doji, shooting star, pin bar, engulfing).
5. The BEAR case AND the BULL case — state the strongest argument for each side
   before picking a direction. Do not default to buying.
6. Costs: is the expected move to target clearly larger than {spread_pct:.2f}% spread/fees?

Decision rules:

If position_type is NONE:
- An entry REQUIRES all of:
  (a) a defined target_price at the next meaningful S/R level,
  (b) a defined invalidation_price where the setup is wrong,
  (c) reward-to-risk = |target − price| / |price − invalidation| ≥ 2.0 after spread,
  (d) trend and momentum agree with the direction (no counter-trend entries),
  (e) entry confidence ≥ 70.
- Choose BUY_LONG only when all five hold for the long side.
- Choose SELL_SHORT only when all five hold for the short side — shorts and longs
  are evaluated with identical strictness; in a downtrend the short side is the
  trend-following side.
- DO NOT enter when: the move already happened (extended/parabolic candles), price
  is mid-range between support and resistance (no edge), the last candle is a huge
  spike (wait for the retest), or the chart is choppy/sideways noise.
- If anything is weak, late, unclear — HOLD_NO_POSITION. Missing a trade costs
  nothing; a bad trade costs real money.

If position_type is LONG (you are LONG — you profit when price goes UP):
- The spread was already paid at entry. A position near entry is in the spread recovery zone — that is NORMAL.
- HOLD_LONG if: upward momentum is intact, the candle is rising, or price is still within ±{spread_pct:.2f}% of entry.
- CLOSE_LONG if any of the following apply:
  (1) A clear bearish reversal candle with momentum appears (bearish engulfing, shooting star, strong red candle after a run).
  (2) Price strongly rejects at a key resistance level.
  (3) The uptrend has definitively broken down (lower lows forming).
  (4) PRICE NORMALIZATION AFTER A SPIKE: price made a meaningful gain above entry AND momentum is now clearly exhausting — the candle is forming a long wick, a doji, or a reversal body, OR price has visibly started retracing toward entry after the spike. DO NOT wait for a full reversal — lock in gains while still clearly profitable.
  (5) LOSING POSITION WITH NO RECOVERY CASE: the loss is beyond the spread recovery zone AND the chart structure no longer supports recovery (trend against you, price below broken support). Do not hold a loser on hope — holding requires an ACTIVE reason visible on the chart (intact support below, bullish structure forming). State that reason; if you cannot, CLOSE_LONG.
- NEVER recommend CLOSE_LONG on a strongly rising candle with no reversal sign.
- NEVER close just because the P&L shows a small negative — that is the spread recovery zone.
- NEVER CLOSE_LONG if unrealised loss is within 2× the entry spread (in dollars or %). That is NOT a real loss — you MUST choose HOLD_LONG.

If position_type is SHORT (you are SHORT — you profit when price goes DOWN):
- The spread was already paid at entry. A position near entry is in the spread recovery zone — that is NORMAL.
- HOLD_SHORT if: downward momentum is intact, the candle is falling, or price is still within ±{spread_pct:.2f}% of entry.
- CLOSE_SHORT if any of the following apply:
  (1) A clear bullish reversal candle with upward momentum appears (bullish engulfing, hammer, strong green candle after a drop).
  (2) Price bounces firmly off a key support level.
  (3) The downtrend has definitively reversed upward (higher lows forming).
  (4) PRICE NORMALIZATION AFTER A SPIKE DOWN: price made a meaningful drop below entry AND momentum is now clearly exhausting — the candle is forming a long lower wick, a doji, or a reversal body, OR price has visibly started bouncing back toward entry after the spike down. DO NOT wait for a full reversal — lock in gains while still clearly profitable.
  (5) LOSING POSITION WITH NO RECOVERY CASE: the loss is beyond the spread recovery zone AND the chart structure no longer supports recovery (trend against you, price above broken resistance). Do not hold a loser on hope — holding requires an ACTIVE reason visible on the chart (intact resistance above, bearish structure forming). State that reason; if you cannot, CLOSE_SHORT.
- NEVER recommend CLOSE_SHORT on a strongly falling candle with no reversal sign.
- NEVER close just because the P&L shows a small negative — that is the spread recovery zone.
- NEVER CLOSE_SHORT if unrealised loss is within 2× the entry spread (in dollars or %). That is NOT a real loss — you MUST choose HOLD_SHORT.

Important:
- Never force a trade.
- If uncertain, choose HOLD_NO_POSITION for no position.
- PROTECT PROFITABLE POSITIONS: if you are sitting on a good profit (P&L ≥ 2× spread/fees) and the chart shows the move is exhausting or price is normalizing back toward entry, prefer CLOSE. A profit given back is worse than a modest closed gain — do not hold through a full reversal just to squeeze more.{"" if not good_profit else chr(10) + "- *** This position is currently in good profit and has pulled back from its peak — pay special attention to normalization signals and close if momentum has clearly turned. ***"}
- If already profitable after spread and a CLEAR reversal is visible, you may prefer closing.
- Take spread/fees seriously; small moves are not enough to enter — but a small loss while in the right direction is fine.
- Never claim guaranteed profit.
- This is analysis only, not financial advice.

Return JSON only in this exact format:

{{
  "current_signal": "BUY_LONG | SELL_SHORT | HOLD_NO_POSITION | CLOSE_LONG | HOLD_LONG | CLOSE_SHORT | HOLD_SHORT",
  "confidence": 75,
  "position_type": "NONE | LONG | SHORT",
  "entry_price": null,
  "current_price": null,
  "target_price": "price level you expect the move to reach, or null for HOLD/CLOSE",
  "invalidation_price": "price level where the trade idea is proven wrong, or null",
  "risk_reward": "numeric reward-to-risk ratio for an entry, or null",
  "bull_case": "strongest argument for upside, one sentence",
  "bear_case": "strongest argument for downside, one sentence",
  "profitable_before_spread": null,
  "profitable_after_spread": null,
  "nearest_support": "approximate price or unknown",
  "nearest_resistance": "approximate price or unknown",
  "expected_direction_next": "UP | DOWN | SIDEWAYS | UNCLEAR",
  "spread_impact": "brief explanation",
  "reason": "brief explanation",
  "risk_warning": "brief warning"
}}"""


def _build_client(provider: str, api_key: str) -> OpenAI:
    if provider == "groq":
        return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    return OpenAI(api_key=api_key)


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _normalize_current_signal(value: str) -> str:
    sig = str(value or "HOLD_NO_POSITION").upper().strip()
    sig = sig.replace(" ", "_").replace("-", "_")
    aliases = {
        "BUY": "BUY_LONG",
        "SELL": "SELL_SHORT",
        "HOLD": "HOLD_NO_POSITION",
        "CLOSE": "CLOSE_LONG",
    }
    return aliases.get(sig, sig)


def _parse_price_level(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in ("unknown", "null", "none", "n/a", ""):
        return None
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def _risk_level_from_eval(result: dict) -> str:
    warning = str(result.get("risk_warning", "")).lower()
    direction = str(result.get("expected_direction_next", "")).upper()
    if any(w in warning for w in ("high", "severe", "significant", "large")):
        return "HIGH"
    if direction == "UNCLEAR" or "uncertain" in warning:
        return "HIGH"
    if direction == "SIDEWAYS":
        return "MEDIUM"
    return "MEDIUM"


def _parse_eval_result(result: dict) -> dict:
    current_signal = _normalize_current_signal(result.get("current_signal", "HOLD_NO_POSITION"))
    raw_conf = result.get("confidence", 50)
    # LLMs sometimes return 0–1 float instead of 0–100 integer — normalise both
    if isinstance(raw_conf, float) and raw_conf <= 1.0:
        raw_conf = raw_conf * 100
    confidence = max(1, min(100, int(raw_conf)))

    support = result.get("nearest_support")
    resistance = result.get("nearest_resistance")
    support_f = _parse_price_level(support)
    resistance_f = _parse_price_level(resistance)
    key_level = support_f or resistance_f

    reason = str(result.get("reason", "")).strip()
    risk_warning = str(result.get("risk_warning", "")).strip()
    spread_impact = str(result.get("spread_impact", "")).strip()
    reasoning_parts = [p for p in (reason, spread_impact, risk_warning) if p]
    reasoning = " ".join(reasoning_parts)

    observations = [
        f"Expected next move: {result.get('expected_direction_next', 'UNCLEAR')}",
        f"Support: {support}",
        f"Resistance: {resistance}",
    ]
    if result.get("target_price") not in (None, "", "null"):
        observations.append(f"Target: {result.get('target_price')}")
    if result.get("invalidation_price") not in (None, "", "null"):
        observations.append(f"Invalidation: {result.get('invalidation_price')}")
    if result.get("risk_reward") not in (None, "", "null"):
        observations.append(f"R:R: {result.get('risk_reward')}")
    if result.get("bull_case"):
        observations.append(f"Bull case: {result.get('bull_case')}")
    if result.get("bear_case"):
        observations.append(f"Bear case: {result.get('bear_case')}")
    if spread_impact:
        observations.append(f"Spread impact: {spread_impact}")

    entry_signal = ENTRY_SIGNAL_MAP.get(current_signal, "HOLD")
    exit_action = EXIT_ACTION_MAP.get(current_signal, "HOLD")

    # ── Mechanical guards on ENTRIES (prompt asks for these; enforce them too) ──
    # 1. Risk:reward floor — an entry whose own stated R:R is under 1.5 is vetoed.
    rr = _parse_price_level(result.get("risk_reward"))
    if entry_signal in ("BUY", "SELL") and rr is not None and rr < 1.5:
        log.info("Entry %s vetoed: stated risk_reward %.2f < 1.5", current_signal, rr)
        entry_signal = "HOLD"
        current_signal = "HOLD_NO_POSITION"
    # 2. Confidence floor — calibrated confidence below 70 means no entry.
    if entry_signal in ("BUY", "SELL") and confidence < 70:
        log.info("Entry %s vetoed: confidence %d < 70", current_signal, confidence)
        entry_signal = "HOLD"
        current_signal = "HOLD_NO_POSITION"

    target_price = _parse_price_level(result.get("target_price"))
    invalidation_price = _parse_price_level(result.get("invalidation_price"))
    # Derive trend_strength from the LLM's expected next direction — NOT from its
    # recommended action.  Basing it on the action was circular: CLOSE_SHORT always
    # set trend_strength = "REVERSING", which always let should_llm_close() pass,
    # even when the market was still moving in the position's favour.
    direction_next = str(result.get("expected_direction_next", "UNCLEAR")).upper()
    if direction_next in ("UP", "DOWN"):
        trend_strength = "STRONG"
    elif direction_next == "SIDEWAYS":
        trend_strength = "WEAKENING"
    else:
        trend_strength = "WEAKENING"

    parsed = {
        "current_signal": current_signal,
        "signal": entry_signal,
        "action": exit_action,
        "confidence": confidence,
        "reasoning": reasoning,
        "reason": reason,
        "risk_warning": risk_warning,
        "spread_impact": spread_impact,
        "observations": observations,
        "risk_level": _risk_level_from_eval(result),
        "key_level": key_level,
        "expected_direction_next": str(result.get("expected_direction_next", "UNCLEAR")).upper(),
        "nearest_support": support,
        "nearest_resistance": resistance,
        "profitable_before_spread": result.get("profitable_before_spread"),
        "profitable_after_spread": result.get("profitable_after_spread"),
        "trend_strength": trend_strength,
        "target_price": target_price,
        "invalidation_price": invalidation_price,
        "risk_reward": rr,
        "bull_case": str(result.get("bull_case", "") or ""),
        "bear_case": str(result.get("bear_case", "") or ""),
    }
    return parsed


def _call_vision_llm(
    img_bytes: bytes,
    user_prompt: str,
    api_key: str,
    model: str,
    provider: str,
    *,
    max_tokens: int = 900,
) -> dict:
    client = _build_client(provider, api_key)
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    use_json_format = provider == "openai"

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high" if provider == "openai" else "auto",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    raw = _strip_json_fences(response.choices[0].message.content)
    log.info("%s (%s) response: %s", provider, model, raw[:240])
    return json.loads(raw)


def analyse_chart(
    img_bytes: bytes,
    api_key: str,
    *,
    asset: str = "Unknown",
    timeframe: str = "1 Minute",
    current_price: Optional[float] = None,
    spread_pct: Optional[float] = None,
    position_type: str = "NONE",
    entry_price: Optional[float] = None,
    pnl_pct: Optional[float] = None,
    in_profit: Optional[bool] = None,
    in_spread_recovery_zone: Optional[bool] = None,
    peak_pnl_pct: Optional[float] = None,
    memory: Optional[str] = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    asset_class_hint: Optional[str] = None,
) -> dict:
    """
    Returns legacy fields {signal, confidence, reasoning, observations, risk_level, key_level}
    plus extended evaluator fields from the unified prompt.
    """
    asset_name = display_asset_name(asset)
    spread = spread_pct if spread_pct is not None else default_spread_pct(asset_name)
    prompt = _build_trading_eval_prompt(
        asset=asset_name,
        timeframe=timeframe,
        current_price=current_price,
        spread_pct=spread,
        position_type=position_type,
        entry_price=entry_price,
        pnl_pct=pnl_pct,
        in_profit=in_profit,
        in_spread_recovery_zone=in_spread_recovery_zone,
        peak_pnl_pct=peak_pnl_pct,
        memory=memory,
        asset_class_hint=asset_class_hint,
    )
    raw = _call_vision_llm(img_bytes, prompt, api_key, model, provider)
    return _parse_eval_result(raw)


def analyse_exit(
    img_bytes: bytes,
    api_key: str,
    position: dict,
    *,
    asset: str = "Unknown",
    timeframe: str = "1 Minute",
    spread_pct: Optional[float] = None,
    memory: Optional[str] = None,
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    asset_class_hint: Optional[str] = None,
) -> dict:
    """
    Returns {action, confidence, reasoning, observations, trend_strength}.
    Uses the same unified evaluator prompt with an open position.
    """
    direction = str(position.get("direction", "LONG")).upper()

    # Extract live P&L fields so the LLM knows whether the position has
    # recovered its entry spread or is genuinely at a loss.
    raw_pnl = position.get("pnl_pct")
    pnl_pct: Optional[float] = float(raw_pnl) if raw_pnl is not None else None
    in_profit: Optional[bool] = bool(position.get("in_profit")) if "in_profit" in position else None
    in_recovery = position.get("in_spread_recovery_zone")
    if in_recovery is None:
        spread_cost = float(position.get("spread_cost") or 0)
        unrealised = float(position.get("unrealised_pnl") or 0)
        if spread_cost > 0:
            in_recovery = (
                unrealised <= spread_cost
                and unrealised > -(2.0 * spread_cost)
            )

    # Compute peak P&L % so the LLM knows how much the position gained at its best
    # and how far price has pulled back from that peak — key for normalization detection.
    peak_pnl_pct: Optional[float] = None
    raw_peak = position.get("peak_pnl")
    amount = float(position.get("amount_invested") or 0)
    if raw_peak is not None and amount > 0:
        peak_pnl_pct = float(raw_peak) / amount * 100

    result = analyse_chart(
        img_bytes,
        api_key,
        asset=display_asset_name(asset),
        timeframe=timeframe,
        current_price=position.get("current_price"),
        spread_pct=spread_pct,
        position_type=direction,
        entry_price=position.get("entry_price"),
        pnl_pct=pnl_pct,
        in_profit=in_profit,
        in_spread_recovery_zone=in_recovery,
        peak_pnl_pct=peak_pnl_pct,
        memory=memory,
        model=model,
        provider=provider,
        asset_class_hint=asset_class_hint,
    )
    action = result.get("action", "HOLD")
    if action not in ("HOLD", "CLOSE"):
        action = "HOLD"
    if in_recovery and action == "CLOSE":
        action = "HOLD"
    return {
        "action": action,
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "observations": result["observations"],
        "trend_strength": result.get("trend_strength", "WEAKENING"),
        "current_signal": result.get("current_signal"),
        "risk_warning": result.get("risk_warning", ""),
    }
