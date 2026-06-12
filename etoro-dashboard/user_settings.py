"""User-editable settings persisted on the data volume (survives rebuilds)."""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

SETTINGS_PATH = Path(
    os.environ.get("USER_SETTINGS_PATH", "/app/data/user_settings.json"),
)

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float = -1.0


@dataclass
class ExitKindSettings:
    trailing_stop_pct: float
    take_profit_pct: float
    stop_loss_min_pct: float


@dataclass
class RiskSettings:
    # Defaults tuned for a heavily CRYPTO-CORRELATED fleet (XRP+BTC ≈ 0.8):
    # same-direction stacking caps are deliberately tight — the 5th+ correlated
    # position adds ~full risk and ~zero independent edge.  Heat (4%) sits
    # BELOW the daily halt (5%) so the halt is a true backstop, not a
    # fires-after-the-crash brake.
    enabled: bool = True
    max_concurrent_positions: int = 14
    max_gross_exposure_pct: float = 60.0
    max_portfolio_heat_pct: float = 4.0
    max_cluster_gross_pct: float = 45.0
    max_cluster_net_pct: float = 25.0
    max_same_dir_per_cluster: int = 4
    max_positions_per_asset: int = 4
    block_internal_hedge: bool = False
    daily_drawdown_halt_pct: float = 5.0


@dataclass
class TradingSettings:
    max_trade_usd: float = 1000.0
    demo_trade_amount: float = 1000.0
    min_trade_usd: float = 200.0
    risk_pct_per_trade: float = 0.75
    max_position_pct: float = 6.0
    cash_reserve_pct: float = 10.0
    reserve_hard_pct: float = 5.0
    # Edge-weighted sizing: scale ticket by per-plan OOS evidence (0.5-1.5x)
    edge_sizing: bool = True
    # Skip stock entries in the first/last N minutes of the US session
    # (auction spreads are widest there).  0 = off.  ORB is always exempt.
    avoid_auction_minutes: int = 10


@dataclass
class LearningSettings:
    entry_guidance_enabled: bool = True
    min_bucket_n: int = 8
    # 0.60 (was 0.40): the documented failure mode is high-winrate buckets that
    # lose money — at 0.40 the AND-veto could never catch them.
    lose_winrate_max: float = 0.60
    lose_profit_factor_max: float = 0.75


@dataclass
class DisplaySettings:
    display_tz: str = "Asia/Dubai"


@dataclass
class BehaviorSettings:
    regime_filter_enabled: bool = True
    recovery_exit_enabled: bool = True
    recovery_hold_mult: float = 2.5   # × strategy avg hold while never green
    # True (default): at recovery, set a BREAKEVEN STOP (lock no-loss, let
    # TP/ATR-trail ride a breakout).  False: close immediately at ≥ $0.
    recovery_breakeven_stop: bool = True
    # "Meaningfully green" threshold = this × the position's entry spread cost;
    # also the recovery-zone band for LLM hold decisions.
    spread_recovery_mult: float = 2.0
    # LLM CLOSE on a losing position only executes at ≥ this confidence.
    llm_loss_cut_min_conf: int = 70


@dataclass
class CashFreeingSettings:
    """Cash-liberation guardrails (reserve relax + weakest-position trims)."""
    min_edge_to_free: float = 0.50      # signal edge floor before touching the book
    edge_margin: float = 0.15           # new edge must beat a victim's edge by this
    trim_cooldown_sec: float = 120.0    # min interval between trims of one position
    min_position_age_sec: float = 120.0 # never trim a position younger than this
    max_trim_fraction: float = 0.75     # max share shaved off one position
    keep_min_usd: float = 200.0         # never leave a position smaller than this


@dataclass
class RankingSettings:
    min_trades: int = 13
    pf_flag: float = 0.75
    pf_recover: float = 1.0
    window: int = 40
    review_sec: float = 1800.0
    # When ON, bots flagged bleeding stop opening NEW positions (existing
    # positions stay managed; the flag itself remains advisory/visible).
    bleeding_block_entries: bool = False


@dataclass
class AtrSettings:
    """Volatility (ATR) stop configuration — entry stops + chandelier trailing.

    Entry-stop multipliers are per behaviour class, research-informed: trend
    needs room (2.5–3x ATR), mean-reversion moderate (2x), arb tight (1.5x).
    The TRAIL multiplier is the golden-rule 2x for every class.  noise_floor
    is the minimum ATR-driven stop distance; widen_max caps the stop at
    fixed-floor x widen_max when ATR explodes in a panic (max-loss rule)."""
    stop_mult_trend: float = 2.5
    stop_mult_llm: float = 2.5
    stop_mult_mean_revert: float = 2.0
    stop_mult_arb: float = 1.5
    trail_mult: float = 2.0
    noise_floor_pct: float = 0.10
    widen_max: float = 3.0


def _code_trading_defaults() -> TradingSettings:
    import position_sizer
    return TradingSettings(
        max_trade_usd=position_sizer.MAX_TRADE_USD,
        demo_trade_amount=float(os.environ.get("ETORO_DEMO_TRADE_AMOUNT", "1000")),
        min_trade_usd=position_sizer.MIN_TRADE_USD,
        risk_pct_per_trade=position_sizer.RISK_PCT_PER_TRADE,
        max_position_pct=position_sizer.MAX_POSITION_PCT,
        cash_reserve_pct=position_sizer.CASH_RESERVE_PCT,
        reserve_hard_pct=position_sizer.RESERVE_HARD_PCT,
    )


def _code_learning_defaults() -> LearningSettings:
    import trade_journal
    return LearningSettings(
        min_bucket_n=trade_journal.MIN_BUCKET_N,
        lose_winrate_max=trade_journal.LOSE_WINRATE_MAX,
        lose_profit_factor_max=trade_journal.LOSE_PROFIT_FACTOR_MAX,
    )


def _defaults() -> dict[str, Any]:
    import exit_profiles

    kinds = ("trend", "mean_revert", "arb", "llm")
    profiles = {
        "trend": exit_profiles.TREND,
        "mean_revert": exit_profiles.MEAN_REVERT,
        "arb": exit_profiles.ARB,
        "llm": exit_profiles.LLM,
    }
    return {
        "exit_profiles": {
            k: {
                "trailing_stop_pct": profiles[k].trailing_stop_pct,
                "take_profit_pct": profiles[k].take_profit_pct,
                "stop_loss_min_pct": profiles[k].stop_loss_min_pct,
            }
            for k in kinds
        },
        "risk": asdict(RiskSettings()),
        "trading": asdict(_code_trading_defaults()),
        "learning": asdict(_code_learning_defaults()),
        "display": asdict(DisplaySettings()),
        "behavior": asdict(BehaviorSettings()),
        "ranking": asdict(RankingSettings()),
        "atr": asdict(AtrSettings()),
        "cash_freeing": asdict(CashFreeingSettings()),
        "bot_overrides": {},
    }


def _read_file() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception as exc:
        log.warning("Could not load user settings: %s", exc)
        return {}


def _merge_section(base: dict, saved: dict | None, keys: tuple[str, ...]) -> None:
    if not isinstance(saved, dict):
        return
    for k in keys:
        if k in saved:
            base[k] = saved[k]


def load() -> dict[str, Any]:
    """Merged settings: code defaults ← saved JSON."""
    global _cache, _cache_mtime
    try:
        mtime = SETTINGS_PATH.stat().st_mtime if SETTINGS_PATH.exists() else 0.0
    except OSError:
        mtime = 0.0
    with _lock:
        if _cache is not None and mtime == _cache_mtime:
            return dict(_cache)
    base = _defaults()
    saved = _read_file()
    for kind, vals in (saved.get("exit_profiles") or {}).items():
        if kind in base["exit_profiles"] and isinstance(vals, dict):
            base["exit_profiles"][kind].update(
                {k: vals[k] for k in vals if k in base["exit_profiles"][kind]},
            )
    _merge_section(base["risk"], saved.get("risk"), tuple(RiskSettings().__dataclass_fields__))
    _merge_section(base["trading"], saved.get("trading"), tuple(TradingSettings().__dataclass_fields__))
    _merge_section(base["learning"], saved.get("learning"), tuple(LearningSettings().__dataclass_fields__))
    _merge_section(base["display"], saved.get("display"), tuple(DisplaySettings().__dataclass_fields__))
    _merge_section(base["behavior"], saved.get("behavior"), tuple(BehaviorSettings().__dataclass_fields__))
    _merge_section(base["ranking"], saved.get("ranking"), tuple(RankingSettings().__dataclass_fields__))
    _merge_section(base["atr"], saved.get("atr"), tuple(AtrSettings().__dataclass_fields__))
    _merge_section(base["cash_freeing"], saved.get("cash_freeing"), tuple(CashFreeingSettings().__dataclass_fields__))
    if isinstance(saved.get("bot_overrides"), dict):
        base["bot_overrides"] = {
            str(k): v for k, v in saved["bot_overrides"].items() if isinstance(v, dict)
        }
    with _lock:
        _cache = base
        _cache_mtime = mtime
    return dict(base)


def save(
    *,
    exit_profiles: dict[str, dict] | None = None,
    risk: dict | None = None,
    trading: dict | None = None,
    learning: dict | None = None,
    display: dict | None = None,
    behavior: dict | None = None,
    ranking: dict | None = None,
    atr: dict | None = None,
    cash_freeing: dict | None = None,
    bot_overrides: dict | None = None,
) -> None:
    """Persist user edits; partial updates merge into the saved file."""
    current = _read_file()
    if exit_profiles is not None:
        current["exit_profiles"] = exit_profiles
    if risk is not None:
        current["risk"] = risk
    if trading is not None:
        current["trading"] = trading
    if learning is not None:
        current["learning"] = learning
    if display is not None:
        current["display"] = display
    if behavior is not None:
        current["behavior"] = behavior
    if ranking is not None:
        current["ranking"] = ranking
    if atr is not None:
        current["atr"] = atr
    if cash_freeing is not None:
        current["cash_freeing"] = cash_freeing
    if bot_overrides is not None:
        current["bot_overrides"] = bot_overrides
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)
    except Exception as exc:
        log.warning("Could not save user settings: %s", exc)
        raise
    invalidate()


def has_saved_risk() -> bool:
    return "risk" in _read_file()


def invalidate() -> None:
    global _cache, _cache_mtime
    with _lock:
        _cache = None
        _cache_mtime = -1.0
    try:
        import risk_manager
        risk_manager.invalidate_limits_cache()
    except Exception:
        pass


def exit_kind_settings(kind: str) -> ExitKindSettings:
    data = load()["exit_profiles"].get(kind, {})
    return ExitKindSettings(
        trailing_stop_pct=float(data.get("trailing_stop_pct", 0.0)),
        take_profit_pct=float(data.get("take_profit_pct", 0.0)),
        stop_loss_min_pct=float(data.get("stop_loss_min_pct", 2.5)),
    )


def risk_settings() -> RiskSettings:
    data = load()["risk"]
    return RiskSettings(**{
        f.name: data.get(f.name, getattr(RiskSettings(), f.name))
        for f in fields(RiskSettings())
    })


def trading_settings() -> TradingSettings:
    data = load()["trading"]
    return TradingSettings(**{
        f.name: data.get(f.name, getattr(_code_trading_defaults(), f.name))
        for f in fields(TradingSettings())
    })


def learning_settings() -> LearningSettings:
    data = load()["learning"]
    return LearningSettings(**{
        f.name: data.get(f.name, getattr(_code_learning_defaults(), f.name))
        for f in fields(LearningSettings())
    })


def display_settings() -> DisplaySettings:
    data = load()["display"]
    return DisplaySettings(
        display_tz=str(data.get("display_tz", "Asia/Dubai") or "Asia/Dubai"),
    )


def behavior_settings() -> BehaviorSettings:
    data = load()["behavior"]
    return BehaviorSettings(**{
        f.name: data.get(f.name, getattr(BehaviorSettings(), f.name))
        for f in fields(BehaviorSettings())
    })


def ranking_settings() -> RankingSettings:
    data = load()["ranking"]
    return RankingSettings(**{
        f.name: data.get(f.name, getattr(RankingSettings(), f.name))
        for f in fields(RankingSettings())
    })


def atr_settings() -> AtrSettings:
    data = load()["atr"]
    return AtrSettings(**{
        f.name: data.get(f.name, getattr(AtrSettings(), f.name))
        for f in fields(AtrSettings())
    })


def cash_freeing_settings() -> CashFreeingSettings:
    data = load()["cash_freeing"]
    return CashFreeingSettings(**{
        f.name: data.get(f.name, getattr(CashFreeingSettings(), f.name))
        for f in fields(CashFreeingSettings())
    })


def bot_exit_overrides(bot_key: str) -> tuple[Optional[float], Optional[float]]:
    raw = load().get("bot_overrides", {}).get(bot_key, {})
    if not isinstance(raw, dict):
        return None, None
    trail = raw.get("trailing_stop_pct")
    tp = raw.get("take_profit_pct")
    return (
        float(trail) if trail is not None else None,
        float(tp) if tp is not None else None,
    )
