"""
Per-strategy exit profiles — the single source of truth for how each strategy
banks/protects a position once it's open.

Rationale (see docs/position-exit-logic.md):
  • Trend / momentum strategies want to RIDE winners → trailing stop, no hard cap.
  • Mean-reverting / oscillator strategies give profits back if held → bank the
    move with a hard take-profit (plus a tight trailing backstop once in profit).
  • Arbitrage strategies harvest tiny, frequent edges → small hard take-profit.
  • The LLM decides its own exit each candle → keep a trailing stop as a safety net.

The hard stop-loss (computed in trade_manager.compute_stop_loss_price) is the
universal downside guard; its floor % is also set per strategy here.

ASSET-CLASS ADAPTATION: the profile percentages are calibrated on CRYPTO
(XRP/BTC journal data — the most volatile class we trade).  Other classes get
scaled by typical relative volatility (research: stocks need ~1.5–2.5×ATR stops
vs crypto's 2–5×ATR — roughly half the distance):
    crypto    ×1.0   stocks ×0.5   commodity ×0.7
This keeps every exit at the same "amount of normal noise" across assets.

These are starting points — tune the numbers freely; they take effect on the
next bot start / strategy change (no schema migration needed).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class ExitProfile:
    trailing_stop_pct: float   # % pullback from peak (0 = disabled)
    take_profit_pct:   float   # hard +% target on entry (0 = disabled)
    stop_loss_min_pct: float   # hard stop distance as % of entry (floor; spread mult may widen)
    kind:              str      # "trend" | "mean_revert" | "arb" | "llm"


# ── Profiles by behaviour class ───────────────────────────────────────────────
# Stop-loss rationale: risk per trade should be proportional to the profit
# target so the payoff ratio matches the strategy's win rate.
#   • trend (58% win observed): wide 2.5% stop — trend trades need room; a
#     2.0% hard take-profit banks winners and frees concurrent slots (trailing
#     still protects sub-target peaks that reverse before TP).
#   • mean_revert (38% win observed): the journal showed losers riding to the
#     full −2.5% stop (p90 loss −2.0%) while wins averaged +0.33%.  A 1.0% stop
#     vs the 1.2% take-profit makes the payoff ≈1.2:1 instead of 1:2.
#   • arb (59% win): harvests ~0.6% edges — risking 2.5% to make 0.6% was a
#     4:1 inverse payoff; 0.5% stop makes it symmetric.
#   • llm: model manages the exit per candle; 2.0% hard TP is a backstop so
#     winning LLM positions don't sit open indefinitely when the model holds.
TREND       = ExitProfile(trailing_stop_pct=2.0, take_profit_pct=2.0, stop_loss_min_pct=2.5, kind="trend")
MEAN_REVERT = ExitProfile(trailing_stop_pct=1.0, take_profit_pct=1.2, stop_loss_min_pct=1.0, kind="mean_revert")
ARB         = ExitProfile(trailing_stop_pct=0.0, take_profit_pct=0.6, stop_loss_min_pct=0.5, kind="arb")
LLM         = ExitProfile(trailing_stop_pct=2.0, take_profit_pct=2.0, stop_loss_min_pct=2.5, kind="llm")

# ── Strategy → profile mapping ────────────────────────────────────────────────
PROFILES: dict[str, ExitProfile] = {
    # Trend / momentum — ride winners with a trailing stop
    "supertrend":        TREND,
    "ma_crossover":      TREND,
    "macd":              TREND,
    "adx":               TREND,
    "ichimoku":          TREND,
    "donchian":          TREND,
    "orb":               TREND,
    # Mean-reverting / oscillator — bank the bounce with a hard take-profit
    "rsi":               MEAN_REVERT,
    "stoch_rsi":         MEAN_REVERT,
    "bollinger_squeeze": MEAN_REVERT,
    "mean_reversion":    MEAN_REVERT,
    "candlestick":       MEAN_REVERT,
    # Arbitrage — small, frequent edges → small hard take-profit
    "stat_arb":          ARB,
    "rate_arb":          ARB,
    # LLM — model decides; trailing as a safety net
    "llm":               LLM,
}

_DEFAULT = LLM

# ── Asset-class volatility scaling ────────────────────────────────────────────
# Profiles above are calibrated on crypto.  Other classes scale by typical
# relative per-bar volatility (stocks ≈ half of crypto, FX majors far less).
CLASS_SCALE: dict[str, float] = {
    "crypto":    1.0,
    "commodity": 0.7,
    "stock":     0.5,
    "etf":       0.5,
    "index":     0.4,
    "forex":     0.3,
}

# ── Authoritative class registry (fed from the eToro API) ─────────────────────
# trading_engine registers each bot's instrument here using eToro's own
# instrumentTypeID (via EToroClient.asset_class_for), so classification never
# depends on label keywords.  The keyword heuristics below remain only as a
# fallback for instruments that were never registered (e.g. manual analysis).
_registered: dict[str, str] = {}   # UPPER(label) → class


def register_asset_class(instrument_label: str, klass: str) -> None:
    if instrument_label and klass in CLASS_SCALE:
        _registered[instrument_label.upper().strip()] = klass


_CRYPTO_TOKENS = (
    "XRP", "BTC", "BITCOIN", "ETH", "ETHEREUM", "DOGE", "SOL", "SOLANA",
    "ADA", "CARDANO", "LTC", "LITECOIN", "BNB", "CRYPTO", "COIN",
)
_COMMODITY_TOKENS = (
    "GOLD", "XAU", "SILVER", "XAG", "OIL", "WTI", "BRENT", "CRUDE",
    "NATGAS", "NATURAL GAS", "COPPER", "PLATINUM", "PALLADIUM",
    "WHEAT", "CORN", "SUGAR", "COCOA", "COFFEE",
)


def asset_class(instrument_label: str) -> str:
    """Asset class for an instrument — API-registered value first, keyword
    heuristics as fallback, 'stock' as the conservative default."""
    upper = (instrument_label or "").upper().strip()
    if upper in _registered:
        return _registered[upper]
    if any(t in upper for t in _CRYPTO_TOKENS):
        return "crypto"
    if any(t in upper for t in _COMMODITY_TOKENS):
        return "commodity"
    return "stock"


def _scaled(p: ExitProfile, instrument_label: str) -> ExitProfile:
    scale = CLASS_SCALE.get(asset_class(instrument_label), 1.0)
    if scale == 1.0:
        return p
    return replace(
        p,
        trailing_stop_pct=round(p.trailing_stop_pct * scale, 3),
        take_profit_pct=round(p.take_profit_pct * scale, 3),
        stop_loss_min_pct=round(p.stop_loss_min_pct * scale, 3),
    )


def _base_profile(strategy_key: str) -> ExitProfile:
    """Class profile with user_settings overlay (Settings tab)."""
    import user_settings

    base = PROFILES.get((strategy_key or "").strip().lower(), _DEFAULT)
    us = user_settings.exit_kind_settings(base.kind)
    return replace(
        base,
        trailing_stop_pct=us.trailing_stop_pct,
        take_profit_pct=us.take_profit_pct,
        stop_loss_min_pct=us.stop_loss_min_pct,
    )


def profile(strategy_key: str, instrument_label: str = "") -> ExitProfile:
    """Exit profile for a strategy, volatility-scaled to the instrument's
    asset class (falls back to the LLM profile / crypto scale if unknown)."""
    base = _base_profile(strategy_key)
    return _scaled(base, instrument_label) if instrument_label else base


def resolve(
    strategy_key: str,
    trailing_override: Optional[float] = None,
    take_profit_override: Optional[float] = None,
    instrument_label: str = "",
    bot_key: str = "",
) -> tuple[float, float]:
    """Return (trailing_stop_pct, take_profit_pct) for a strategy on an asset.

    Priority: instruments.toml per-bot override > Settings per-bot override >
    Settings class profile > code defaults (asset-class scaled).
    """
    import user_settings

    p = profile(strategy_key, instrument_label)
    trailing = p.trailing_stop_pct
    tp = p.take_profit_pct
    if bot_key:
        bt, bp = user_settings.bot_exit_overrides(bot_key)
        if bt is not None:
            trailing = bt
        if bp is not None:
            tp = bp
    if trailing_override is not None:
        trailing = float(trailing_override)
    if take_profit_override is not None:
        tp = float(take_profit_override)
    return trailing, tp


def stop_loss_min_pct(strategy_key: str, instrument_label: str = "") -> float:
    """Hard stop distance (% of entry), volatility-scaled to the asset class."""
    return profile(strategy_key, instrument_label).stop_loss_min_pct


# ── ATR-adaptive (regime-aware) stop distance ─────────────────────────────────
# The fixed per-strategy stop_loss_min_pct above is a calm-market FLOOR.  In a
# volatility spike a fixed % is too tight (noise-stopped); in dead markets it's
# too wide.  When a live ATR% is available, the stop widens to k x ATR% — capped
# so it can never exceed STOP_WIDEN_MAX x the floor — giving every strategy the
# same "amount of normal noise" of room regardless of regime:
#   stop_pct = clamp(k x atr_pct, floor, floor x STOP_WIDEN_MAX)
# k (ATR multiple) by behaviour class — trend needs more room than arb:
ATR_MULT_BY_KIND: dict[str, float] = {
    "trend":       2.5,
    "llm":         2.5,
    "mean_revert": 1.5,
    "arb":         1.0,
}
STOP_WIDEN_MAX = 3.0   # stop may grow up to 3x the floor in a vol spike


def atr_mult(strategy_key: str) -> float:
    return ATR_MULT_BY_KIND.get(profile(strategy_key).kind, 2.0)


def adaptive_stop_pct(
    strategy_key: str,
    instrument_label: str = "",
    atr_pct: Optional[float] = None,
) -> float:
    """Regime-aware stop distance (% of entry).

    Falls back to the fixed floor when no ATR% is supplied, so callers that
    cannot compute ATR keep the exact previous behaviour.
    """
    floor = stop_loss_min_pct(strategy_key, instrument_label)
    if atr_pct is None or atr_pct <= 0:
        return floor
    k = atr_mult(strategy_key)
    return float(min(max(k * atr_pct, floor), floor * STOP_WIDEN_MAX))


# ── Position-size share of the per-trade dollar cap ─────────────────────────────
# Applied AFTER the global MAX_TRADE_USD ceiling in position_sizer — scales
# notional to strategy behaviour (small edges → smaller tickets).
SIZE_PCT_BY_KIND: dict[str, float] = {
    "arb":         50.0,   # harvest tiny edges — half the cap
    "mean_revert": 75.0,   # moderate bounce trades
    "trend":      100.0,   # full cap — needs room for wide stops
    "llm":        100.0,   # model-driven — full discretion up to cap
}


def size_pct(strategy_key: str) -> float:
    """% of the per-trade cap (MAX_TRADE_USD) this strategy may deploy."""
    kind = profile(strategy_key).kind
    return SIZE_PCT_BY_KIND.get(kind, 100.0)
