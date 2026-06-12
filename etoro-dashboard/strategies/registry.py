"""Strategy registry — single source of truth for all available strategies."""
from __future__ import annotations

from .base import Strategy
from .llm import LLMStrategy
from .orb import ORBStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .rsi import RSIStrategy
from .ma_crossover import MACrossoverStrategy
from .stat_arb import StatArbStrategy
from .rate_arb import RateArbStrategy
from .macd import MACDStrategy
from .supertrend import SupertrendStrategy
from .mean_reversion import MeanReversionStrategy
from .candlestick import CandlestickStrategy
from .stoch_rsi import StochRSIStrategy
from .donchian import DonchianStrategy
from .ichimoku import IchimokuStrategy
from .adx import ADXStrategy
from .daviddtech import DaviddTechStrategy

# Ordered for the UI dropdown — LLM first, then alphabetical-ish by category
_ALL: list[Strategy] = [
    # ── AI ────────────────────────────────────────────────────────────────────
    LLMStrategy(),

    # ── Trend-following ───────────────────────────────────────────────────────
    SupertrendStrategy(),
    MACrossoverStrategy(),
    MACDStrategy(),
    ADXStrategy(),
    IchimokuStrategy(),
    DaviddTechStrategy(),

    # ── Breakout ─────────────────────────────────────────────────────────────
    ORBStrategy(),
    DonchianStrategy(),

    # ── Mean reversion / oscillators ─────────────────────────────────────────
    RSIStrategy(),
    StochRSIStrategy(),
    BollingerSqueezeStrategy(),
    MeanReversionStrategy(),

    # ── Arbitrage ─────────────────────────────────────────────────────────────
    StatArbStrategy(),
    RateArbStrategy(),

    # ── Price action ──────────────────────────────────────────────────────────
    CandlestickStrategy(),
]

_REGISTRY: dict[str, Strategy] = {s.key: s for s in _ALL}

DEFAULT_KEY = "llm"


def get(key: str) -> Strategy:
    """Return a strategy by key; falls back to LLM if unknown."""
    return _REGISTRY.get(key, _REGISTRY[DEFAULT_KEY])


def all_strategies() -> list[Strategy]:
    return list(_ALL)


def display_names() -> dict[str, str]:
    """key → display_name mapping for UI dropdowns."""
    return {s.key: s.display_name for s in _ALL}


def keys() -> list[str]:
    return [s.key for s in _ALL]
