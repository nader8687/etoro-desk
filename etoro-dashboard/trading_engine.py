"""
Background trading loop — runs in daemon threads independent of Streamlit tabs.

Multi-instrument, multi-timeframe: one EngineState per bot_id (string key),
one thread per bot, one shared supervisor keeps everything alive.

bot_id is the instruments.toml section key (e.g. "btc", "btc_15m").  Multiple
bots can run for the same underlying asset at different candle intervals.

Backwards-compatible: existing callers that pass instrument_id as an int are
routed to the primary (first-registered) bot for that instrument, so the
Trading tab continues to work unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

import bot_registry
import engine_notify
import market_data_hub
import positions_cache
import signal_log
import signal_worker
import tick_manager
import trade_journal
import trade_manager
from etoro_client import EToroClient, get_shared_client

log = logging.getLogger(__name__)

TICK_INTERVAL     = 1.0
# Slower cadence for a FLAT bot (no open position).  A flat bot only acts on
# candle closes, so polling every second wastes wakeups — especially with many
# bots running.  A bot holding a position stays at TICK_INTERVAL so stop-loss /
# trailing-stop / take-profit react promptly.  On 1-minute+ candles the extra
# latency for entry detection is negligible.
FLAT_TICK_INTERVAL = 10.0
IDLE_TICK_INTERVAL = 30.0   # auto-trade OFF and flat — minimal CPU
ADOPT_SUPPRESS_SEC = 45.0
# Grace period before treating a tracked position as "vanished" (closed
# externally).  Protects against the positions-cache lag right after opening.
VANISH_GRACE_SEC  = 15.0
# A tracked position must be ABSENT from a confirmed-fresh, non-empty positions
# cache for this many CONSECUTIVE ticks before we treat it as truly closed.
# Guards against transient/incomplete portfolio fetches wrongly "vanishing" a
# still-open position (which would strip its identity and log a phantom close).
VANISH_MISS_THRESHOLD = 5
HTF_HIST_POLL_SEC = 30.0   # how often 4h/daily bots poll eToro for a new closed bar


# ── Config / snapshot data types ─────────────────────────────────────────────

@dataclass
class EngineConfig:
    instrument_id:   int
    instrument_label: str
    interval_label:  str
    interval_seconds: int
    candle_count:    int
    trading_active:  bool
    demo_amount:     float
    is_demo:         bool
    api_key:         str
    user_key:        str
    strategy_name:    str = "llm"   # key from strategies.registry
    bot_id:           str = ""     # instruments.toml key; empty = derive from iid
    trailing_stop_pct: float = 1.5  # % pullback from peak to trigger trailing close (0 = disabled)
    take_profit_pct:   float = 0.0  # hard take-profit target in % (0 = disabled)


@dataclass
class EngineSnapshot:
    """Trading state only — chart data lives in market_data_hub.ChartSnapshot."""
    instrument_id:   int
    instrument_label: str
    interval_label:  str
    latest_ask:      float
    latest_bid:      float
    tick_count:      int
    last_tick_time:  Optional[datetime]
    trading_active:  bool
    position_open:   bool
    bot_id:          str = ""   # instruments.toml key, e.g. "xrp_15m"
    bot_uuid:        str = ""   # stable UUID from bot_registry, used as signal/trade identifier
    started_at:      datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at:      datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── Per-instrument engine state ───────────────────────────────────────────────

@dataclass
class _EngineState:
    config:              EngineConfig
    client:              EToroClient
    running:             bool = False
    thread:              Optional[threading.Thread] = None
    prev_candle_time:    Optional[pd.Timestamp] = None
    processed_sig_at:    Optional[str] = None
    processed_exit_at:   Optional[str] = None
    snapshot:            Optional[EngineSnapshot] = None
    skip_adopt_until:    float = 0.0  # monotonic
    vanish_misses:       int = 0      # consecutive confirmed "position absent" ticks
    # Set True when auto-trade is toggled ON so the engine fires one immediate
    # signal dispatch instead of waiting for the next candle close.
    pending_dispatch:    bool = False
    last_htf_hist_poll:  float = 0.0   # monotonic; HTF bots poll eToro hist for bar close
    started_at:          datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    bot_uuid:            str = ""   # assigned from bot_registry on start_instrument
    market_closed:       bool = False  # True while the instrument's exchange is closed (stocks)


# ── Module-level registry ─────────────────────────────────────────────────────

_lock = threading.Lock()
_engines: dict[str, _EngineState] = {}    # bot_id → _EngineState
_iid_to_primary: dict[int, str] = {}      # instrument_id → primary bot_id (backward compat)
_active_iid: Optional[int] = None         # Trading-tab instrument (backwards compat)
_active_bot_id: Optional[str] = None      # bot_id of most-recently configured engine
# Bots the user explicitly turned OFF.  This is AUTHORITATIVE: is_auto_trade and
# the engine both treat a disabled bot as OFF, and start_instrument refuses to
# revive it — so neither a per-tab rerun nor a container restart can silently
# turn a bot back on.  Persisted to the data volume so OFF survives restarts.
# set_auto_trade(True) / the global Start button re-enables.
_disabled_bots: set[str] = set()
_DISABLED_PATH = Path(os.environ.get("DISABLED_BOTS_PATH", "/app/data/disabled_bots.json"))


def _save_disabled() -> None:
    """Atomic write (temp + rename) so concurrent toggles can't corrupt the file."""
    try:
        _DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _DISABLED_PATH.with_name(_DISABLED_PATH.name + ".tmp")
        tmp.write_text(json.dumps(sorted(_disabled_bots)), encoding="utf-8")
        os.replace(tmp, _DISABLED_PATH)
    except Exception:
        log.warning("Could not persist disabled-bots set", exc_info=True)


def _load_disabled() -> None:
    try:
        if not _DISABLED_PATH.exists():
            return
        text = _DISABLED_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data, _end = json.JSONDecoder().raw_decode(text)
            log.warning("Disabled-bots set recovered from a corrupted file")
        if isinstance(data, list):
            _disabled_bots.update(str(x) for x in data)
    except Exception:
        log.warning("Could not load disabled-bots set", exc_info=True)


_load_disabled()   # populate from disk at import so OFF survives restarts


def prune_disabled(known_keys: set[str]) -> None:
    """Drop persisted disabled-set entries that are not real configured bots.

    Self-heals stray keys such as an instrument-id "phantom" engine key (e.g.
    "100003") that an empty-bot_id Trading-tab render could create in older
    builds.  Call once after boot has registered every configured engine."""
    stale = {k for k in _disabled_bots if k not in known_keys}
    if stale:
        _disabled_bots.difference_update(stale)
        _save_disabled()
        log.info("Pruned %d stale disabled-bot key(s): %s", len(stale), sorted(stale))

_portfolio_bump: bool = False
_last_closes: dict[int, trade_manager.ClosedTrade] = {}
_trade_errors: dict[int, str] = {}

_desired_live: bool = True
_supervisor_thread: Optional[threading.Thread] = None
_supervisor_started: bool = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _engine_key(config: EngineConfig) -> str:
    """Derive the registry key for an EngineConfig."""
    return config.bot_id if config.bot_id else str(config.instrument_id)


def _primary_for(instrument_id: int) -> Optional[str]:
    """Return the primary bot_id registered for instrument_id (no lock needed for reads)."""
    return _iid_to_primary.get(instrument_id)


def _bump_portfolio() -> None:
    global _portfolio_bump
    _portfolio_bump = True
    positions_cache.invalidate()


def _positions_for_instrument(
    positions: list[dict], instrument_id: int
) -> list[dict]:
    return [p for p in positions if p.get("instrument_id") == instrument_id]


def _sync_hub_config(config: EngineConfig) -> None:
    market_data_hub.configure(
        market_data_hub.HubConfig(
            instrument_id=config.instrument_id,
            interval_label=config.interval_label,
            interval_seconds=config.interval_seconds,
            candle_count=config.candle_count,
            bot_id=config.bot_id,
        ),
    )


def _free_cash_and_resize(
    client: EToroClient,
    config: EngineConfig,
    confidence: float,
    signal: str,
    instrument_id: int,
    decision,
):
    """A signal couldn't be funded from spendable cash.  For a strong-enough
    signal, try to make room WITHOUT a full liquidation, then re-size:
      1. relax the cash-reserve floor (RESERVE_HARD_PCT) — closes nothing;
      2. partial-trim the weakest open positions (cash_manager).
    Returns the re-computed SizeDecision (amount may still be 0 → genuine skip)."""
    import position_sizer, cash_manager
    new_edge = cash_manager.signal_edge(config.strategy_name, confidence)
    if new_edge < cash_manager.min_edge_to_free():
        return decision  # too weak to justify touching the book

    # Only act when CASH is the actual bottleneck.  If spendable (at the normal
    # reserve) already covers a min trade, the skip came from something else
    # (e.g. a tiny risk-budget notional) and freeing cash wouldn't help — so
    # don't churn the book.
    try:
        spendable = max(0.0, float(decision.free_cash) - float(decision.reserve))
    except Exception:
        spendable = 0.0
    if spendable >= position_sizer.min_trade_usd():
        return decision

    # CAPACITY PRECHECK — never pay for cash the risk gate won't let us spend.
    # The amount-independent hard caps (concurrent count, per-asset, cluster
    # crowding, drawdown halt) run AFTER sizing in _maybe_open_trade; without
    # this check we could trim live positions (paying spread) to fund a trade
    # those caps then block regardless of size.
    import risk_manager
    _pre = risk_manager.capacity_precheck(
        direction="LONG" if signal.upper() == "BUY" else "SHORT",
        instrument_id=instrument_id,
        instrument_label=config.instrument_label,
        equity=float(getattr(decision, "equity", 0.0) or 0.0),
    )
    if _pre.blocked:
        log.info(
            "Cash freeing skipped on %s (%s): risk capacity blocked — %s",
            config.instrument_label, config.strategy_name, _pre.reason,
        )
        return decision

    def _resize():
        import order_executor
        return position_sizer.size_trade(
            client,
            strategy=config.strategy_name,
            instrument_label=config.instrument_label,
            is_demo=config.is_demo,
            config_amount=config.demo_amount,
            reserve_pct=position_sizer.reserve_hard_pct(),
            inflight_usd=order_executor.inflight_cash(),
        )

    # Tier 1 — relax the reserve floor for this strong signal (no closes).
    relaxed = _resize()
    if relaxed.amount > 0:
        log.info("Funded %s via reserve relaxation: %s", config.instrument_label, relaxed.summary())
        return relaxed

    # Tier 2 — trim the weakest open positions, then re-size against REAL cash.
    try:
        res = cash_manager.try_free_cash(
            client,
            is_demo=config.is_demo,
            needed_usd=position_sizer.min_trade_usd(),
            new_strategy=config.strategy_name,
            new_confidence=confidence,
        )
    except Exception:
        log.exception("Cash freeing failed for %s", config.instrument_label)
        return decision
    if res.get("freed", 0.0) > 0:
        log.info("Freed $%.0f for %s: %s",
                 res["freed"], config.instrument_label, "; ".join(res["actions"]))
        engine_notify.push(
            "cash_freed",
            f"{config.instrument_label}: freed ${res['freed']:,.0f} for {signal} — "
            f"{'; '.join(res['actions'])}",
            instrument_id=instrument_id,
        )
        return _resize()
    return decision


def _annotate_signal_exec(
    state: _EngineState,
    *,
    trigger_at: str,
    signal_type: str,
    decision: str,
    status: str,
    reason: str = "",
) -> None:
    """Write execution outcome onto the matching Signals-tab log row."""
    if not trigger_at or not state.bot_uuid:
        return
    d = (decision or "").upper()
    if signal_type == "entry":
        if d == "HOLD":
            signal_log.annotate_execution(
                bot_id=state.bot_uuid,
                instrument_id=state.config.instrument_id,
                interval=state.config.interval_label,
                trigger_at=trigger_at,
                signal_type="entry",
                status="not_applicable",
                reason=reason or "HOLD — no entry order",
            )
            return
        if d not in ("BUY", "SELL"):
            return
    elif signal_type == "exit":
        if d == "HOLD":
            signal_log.annotate_execution(
                bot_id=state.bot_uuid,
                instrument_id=state.config.instrument_id,
                interval=state.config.interval_label,
                trigger_at=trigger_at,
                signal_type="exit",
                status="not_applicable",
                reason=reason or "HOLD — no close order",
            )
            return
        if d != "CLOSE":
            return
    signal_log.annotate_execution(
        bot_id=state.bot_uuid,
        instrument_id=state.config.instrument_id,
        interval=state.config.interval_label,
        trigger_at=trigger_at,
        signal_type=signal_type,
        status=status,
        reason=reason,
    )


_regime_memo_lock = threading.Lock()
_regime_memo: dict[tuple, object] = {}   # (instrument_id, last_candle_ts) -> RegimeState


def _classify_regime_memo(instrument_id: int, df):
    """regime.classify computed ONCE per (instrument, latest candle) and shared.

    At a candle close, 8–16 bots on the same asset gate on the SAME DataFrame —
    this collapses their identical ADX/ATR/EMA classifications into one pass.
    A concurrent double-compute is harmless (idempotent result, last write wins)."""
    import regime as _regime
    try:
        key = (instrument_id, str(df["time"].iloc[-1]))
    except Exception:
        return _regime.classify(df)
    with _regime_memo_lock:
        hit = _regime_memo.get(key)
    if hit is not None:
        return hit
    rs = _regime.classify(df)
    with _regime_memo_lock:
        if len(_regime_memo) > 256:   # tiny bound; entries are per-candle
            _regime_memo.clear()
        _regime_memo[key] = rs
    return rs


def _maybe_open_trade(
    config: EngineConfig,
    client: EToroClient,
    instrument_id: int,
    sig_result: dict,
    ask: float,
    bid: float,
    state: _EngineState,
) -> None:
    _signal = (sig_result.get("signal") or "HOLD").upper()
    at = sig_result.get("_at", "")

    if not config.trading_active or not config.is_demo:
        if _signal in ("BUY", "SELL"):
            _annotate_signal_exec(
                state, trigger_at=at, signal_type="entry", decision=_signal,
                status="skipped",
                reason="Auto-trade off" if not config.trading_active else "Demo account required",
            )
        return
    # Authoritative OFF: a user-disabled bot never opens, even if some path left
    # trading_active set.  The disabled set is the single source of truth.
    if config.bot_id and config.bot_id in _disabled_bots:
        if _signal in ("BUY", "SELL"):
            _annotate_signal_exec(
                state, trigger_at=at, signal_type="entry", decision=_signal,
                status="skipped", reason="Bot disabled on Bots tab",
            )
        return
    if sig_result.get("_status") != "done" or "_error" in sig_result:
        return

    if state.processed_sig_at == at:
        return
    state.processed_sig_at = at   # mark processed regardless of viability

    if _signal == "HOLD":
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision="HOLD",
            status="not_applicable",
        )
        return

    # ── Execution quality gate ────────────────────────────────────────────────
    if not sig_result.get("viable", True):
        exec_risk    = sig_result.get("exec_risk", "?")
        net_edge     = sig_result.get("net_edge_pct", 0.0)
        slippage     = sig_result.get("slippage_pct", 0.0)
        skip_reason = (
            f"Execution quality: net edge {net_edge:+.3f}% after "
            f"{slippage:.3f}% slippage ({exec_risk} risk)"
        )
        log.info(
            "Signal skipped (execution quality) %s: sig=%s conf=%s "
            "net_edge=%+.3f%% slippage=%.3f%% risk=%s",
            instrument_id, sig_result.get("signal"), sig_result.get("confidence"),
            net_edge, slippage, exec_risk,
        )
        engine_notify.push(
            "signal_skipped",
            f"{config.instrument_label}: {sig_result.get('signal')} skipped "
            f"— net edge {net_edge:+.3f}% after {slippage:.3f}% slippage "
            f"({exec_risk} risk)",
            instrument_id=instrument_id,
        )
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason=skip_reason,
        )
        return

    if trade_manager.has_open(state.bot_uuid):
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason="Bot already has an open position",
        )
        return

    # ── Evidence-based learning guard ─────────────────────────────────────────
    # Veto setups that historically lost — but ONLY when there is a statistically
    # meaningful sample (entry_guidance is conservative and allows by default, so
    # this never fires on a fresh system and cannot overfit to a few trades).
    _conf     = int(sig_result.get("confidence", 0))
    _exec_risk = sig_result.get("exec_risk", "") or ""
    _net_edge  = float(sig_result.get("net_edge_pct", 0.0) or 0.0)
    _direction = "LONG" if _signal.upper() == "BUY" else "SHORT"
    # Win-rate-shrunk confidence (conservative; == raw until enough samples).
    # Used for cash-freeing edge + analytics, NOT for the viability/guidance gate.
    try:
        _conf_cal = trade_journal.calibrated_confidence(config.strategy_name, _direction, _conf)
    except Exception:
        _conf_cal = _conf
    guidance = trade_journal.entry_guidance(
        instrument_id, config.strategy_name, _direction, _conf, _exec_risk,
    )
    if not guidance.get("allow", True):
        log.info(
            "Signal vetoed by trade journal %s: %s",
            instrument_id, guidance.get("reason", ""),
        )
        engine_notify.push(
            "signal_skipped",
            f"{config.instrument_label}: {_signal} skipped — {guidance.get('reason','')}",
            instrument_id=instrument_id,
        )
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason=guidance.get("reason", "Trade journal veto"),
        )
        return
    if guidance.get("caution"):
        log.info("Trade journal caution %s: %s", instrument_id, guidance["caution"])

    # ── Market-regime filter ──────────────────────────────────────────────────
    # Silence strategy FAMILIES that are in the wrong regime (mean-reversion in a
    # strong trend, trend bots in dead low-vol chop).  Also yields the live ATR%
    # used for regime-aware stop sizing below.  Fails open to "unknown" (allows).
    _atr_pct = None
    _regime_label = ""
    try:
        import user_settings
        _regime_on = user_settings.behavior_settings().regime_filter_enabled
    except Exception:
        _regime_on = True
    try:
        import regime as _regime, market_data_hub
        _snap = market_data_hub.get_snapshot(instrument_id)
        _df = getattr(_snap, "chart_data", None) if _snap else None
        if _df is not None and len(_df) > 0:
            _rs = _classify_regime_memo(instrument_id, _df)
            _regime_label = _rs.label
            _atr_pct = _rs.atr_pct or None
            if _regime_on:
                _allowed, _why = _regime.allows(config.strategy_name, _rs)
                if not _allowed:
                    log.info("Signal suppressed by regime filter %s: %s", instrument_id, _why)
                    engine_notify.push(
                        "signal_skipped",
                        f"{config.instrument_label}: {_signal} skipped — {_why}",
                        instrument_id=instrument_id,
                    )
                    _annotate_signal_exec(
                        state, trigger_at=at, signal_type="entry", decision=_signal,
                        status="skipped", reason=_why,
                    )
                    return
    except Exception:
        log.warning("Regime filter failed — allowing trade", exc_info=True)

    # ── Pacing slot (anti-stampede) ───────────────────────────────────────────
    # All cheap gates passed — this entry is real.  Acquire a paced slot so the
    # fleet's candle-close burst hits eToro a few orders per second instead of
    # all at once (429 storm → sleep-retry → late fills).  Exits never pace.
    import order_executor
    _wait = order_executor.acquire_entry_slot()
    if _wait is None:
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped",
            reason=f"Pacing queue full (> {order_executor.MAX_WAIT_SEC:.0f}s) — entry not chased",
        )
        return

    # ── Re-quote guard (stale-signal protection) ──────────────────────────────
    # The quote that justified this signal is now _wait seconds old (plus any
    # LLM/queue latency).  Re-read the live quote; if price already ran in the
    # trade's direction beyond a fraction of its computed edge, the edge is
    # gone — abandon rather than chase.  A favourable move just improves entry.
    _q = tick_manager.get_latest_quote(instrument_id)
    if _q and _q[0] and _q[1]:
        _new_ask, _new_bid = float(_q[0]), float(_q[1])
        _tol_pct = max(0.05, 0.33 * max(_net_edge, 0.0))
        if _signal == "BUY" and ask > 0:
            _adverse = (_new_ask - ask) / ask * 100.0
        elif _signal == "SELL" and bid > 0:
            _adverse = (bid - _new_bid) / bid * 100.0
        else:
            _adverse = 0.0
        if _adverse > _tol_pct:
            _reason = (
                f"Stale signal: price moved {_adverse:+.3f}% adverse while paced "
                f"(tolerance {_tol_pct:.3f}%) — not chasing"
            )
            log.info("Entry abandoned on %s (%s): %s",
                     config.instrument_label, config.strategy_name, _reason)
            engine_notify.push(
                "signal_skipped",
                f"{config.instrument_label}: {_signal} skipped — {_reason}",
                instrument_id=instrument_id,
            )
            _annotate_signal_exec(
                state, trigger_at=at, signal_type="entry", decision=_signal,
                status="skipped", reason=_reason,
            )
            return
        # Proceed at the CURRENT quote — sizing, stop placement and the order
        # itself should use reality, not the signal-time snapshot.
        ask, bid = _new_ask, _new_bid

    # ── Entry-quality gates (Settings-driven) ─────────────────────────────────
    # 1. Bleeding auto-demote: when the Settings toggle is ON, a bot carrying
    #    the advisory BLEEDING flag stops opening NEW positions (existing ones
    #    stay fully managed; the flag stays visible either way).
    try:
        import user_settings as _us_gate
        if getattr(_us_gate.ranking_settings(), "bleeding_block_entries", False):
            import bot_ranking as _br_gate
            if _br_gate.is_bleeding(bot_id=state.bot_uuid):
                _reason = ("bot is flagged BLEEDING and Settings blocks new "
                           "entries while bleeding (existing positions still managed)")
                log.info("Entry blocked on %s (%s): %s",
                         config.instrument_label, config.strategy_name, _reason)
                engine_notify.push(
                    "signal_skipped",
                    f"{config.instrument_label}: {_signal} skipped — {_reason}",
                    instrument_id=instrument_id,
                )
                _annotate_signal_exec(
                    state, trigger_at=at, signal_type="entry", decision=_signal,
                    status="skipped", reason=_reason,
                )
                return
    except Exception:
        log.debug("bleeding entry gate failed open", exc_info=True)
    # 2. Auction-window avoidance: exchange-traded instruments (eToro stock ids
    #    are small; crypto lives at >= 100000) skip entries in the first/last N
    #    minutes of the US session, where spreads are widest.  ORB is exempt —
    #    its entire edge IS the opening range.
    try:
        _avoid_min = int(getattr(_us_gate.trading_settings(), "avoid_auction_minutes", 0))
    except Exception:
        _avoid_min = 0
    if _avoid_min > 0 and instrument_id < 100000 and config.strategy_name != "orb":
        try:
            import market_calendar as _mc_gate
            _sess_edge = _mc_gate.us_session_edge_minutes()
        except Exception:
            _sess_edge = None
        if _sess_edge is not None and (
            _sess_edge[0] < _avoid_min or _sess_edge[1] < _avoid_min
        ):
            _reason = (
                f"auction window — {'first' if _sess_edge[0] < _avoid_min else 'last'} "
                f"{_avoid_min} min of the US session has the widest spreads"
            )
            log.info("Entry skipped on %s (%s): %s",
                     config.instrument_label, config.strategy_name, _reason)
            _annotate_signal_exec(
                state, trigger_at=at, signal_type="entry", decision=_signal,
                status="skipped", reason=_reason,
            )
            return

    # ── Dynamic position sizing ───────────────────────────────────────────────
    # Risk-based, performance-adaptive, account-aware (see position_sizer).
    # config.demo_amount is only the fallback when the account API is down.
    # inflight_usd: $ reserved by other bots' orders still in flight this same
    # burst — prevents N concurrent entries from collectively overshooting the
    # cash reserve they each individually fit.
    import position_sizer
    decision = position_sizer.size_trade(
        client,
        strategy=config.strategy_name,
        instrument_label=config.instrument_label,
        is_demo=config.is_demo,
        config_amount=config.demo_amount,
        inflight_usd=order_executor.inflight_cash(),
        bot_key=config.bot_id,
    )
    if decision.amount <= 0:
        # Not enough spendable cash.  For a strong-enough signal, try to free
        # room before giving up — relax the reserve (no closes), then trim the
        # weakest open positions — then re-size against the real account.
        decision = _free_cash_and_resize(
            client, config, _conf_cal, _signal, instrument_id, decision,
        )
    if decision.amount <= 0:
        log.info(
            "Trade skipped by position sizer on %s (%s): %s",
            config.instrument_label, config.strategy_name, decision.reason,
        )
        engine_notify.push(
            "signal_skipped",
            f"{config.instrument_label}: {_signal} skipped — {decision.reason}",
            instrument_id=instrument_id,
        )
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason=decision.reason,
        )
        return
    log.info(
        "Position size for %s (%s): %s",
        config.instrument_label, config.strategy_name, decision.summary(),
    )

    # ── Portfolio-level risk gate ─────────────────────────────────────────────
    # Inspect the COMBINED book before opening: caps gross/cluster/net exposure,
    # portfolio heat, concurrent count, correlated stacking, internal hedging,
    # and a daily-drawdown kill-switch.  Default-on but conservative; can SHRINK
    # the size or BLOCK the trade.  Fails open on internal error.
    import risk_manager
    rdecision = risk_manager.check_new_trade(
        direction=_direction,
        amount=decision.amount,
        risk_dollars=decision.risk_dollars,
        equity=decision.equity,
        instrument_id=instrument_id,
        instrument_label=config.instrument_label,
        strategy=config.strategy_name,
        bot_id=state.bot_uuid,
    )
    if rdecision.blocked:
        log.info(
            "Trade blocked by risk manager on %s (%s): %s",
            config.instrument_label, config.strategy_name, rdecision.reason,
        )
        engine_notify.push(
            "signal_skipped",
            f"{config.instrument_label}: {_signal} blocked — {rdecision.reason}",
            instrument_id=instrument_id,
        )
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason=rdecision.reason,
        )
        return
    if rdecision.shrunk:
        log.info(
            "Risk manager shrank %s (%s): $%.0f → $%.0f (%s)",
            config.instrument_label, config.strategy_name,
            decision.amount, rdecision.amount, rdecision.capped_by,
        )
    open_amount = rdecision.amount

    # Reserve the in-flight cash while the order travels — released as soon as
    # the call returns (success or failure); by then the account-cache
    # invalidation makes the real balance visible to the next sizer call.
    order_executor.reserve_cash(state.bot_uuid, open_amount)
    try:
        opened = trade_manager.open_trade(
            instrument_id,
            config.instrument_label,
            _signal,
            _conf,
            ask,
            bid,
            client=client,
            demo_amount=open_amount,
            bot_id=state.bot_uuid,
            bot_key=config.bot_id,
            strategy=config.strategy_name,
            exec_risk=_exec_risk,
            net_edge_pct=_net_edge,
            atr_pct=_atr_pct,
            interval=config.interval_label,
            entry_reason=str(sig_result.get("reasoning", "") or "")[:300],
            regime=_regime_label,
            confidence_calibrated=_conf_cal,
        )
    finally:
        order_executor.release_cash(state.bot_uuid)
    if opened:
        position_sizer.invalidate_account_cache()
        _bump_portfolio()
        engine_notify.push(
            "trade_open",
            f"Opened {config.instrument_label}",
            instrument_id=instrument_id,
        )
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="executed", reason="Order opened on eToro demo",
        )
    elif err := trade_manager.get_last_error():
        with _lock:
            _trade_errors[instrument_id] = err
        engine_notify.push("trade_error", err, instrument_id=instrument_id)
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="entry", decision=_signal,
            status="skipped", reason=err,
        )


def _maybe_process_exit(
    instrument_id: int,
    exit_result: dict,
    ask: float,
    bid: float,
    client: EToroClient,
    state: _EngineState,
) -> Optional[trade_manager.ClosedTrade]:
    if exit_result.get("_status") != "done" or "_error" in exit_result:
        return None

    at = exit_result.get("_at", "")
    if state.processed_exit_at == at:
        return None
    state.processed_exit_at = at
    action = exit_result.get("action", "HOLD").upper()

    if action != "CLOSE":
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="exit", decision=action,
            status="not_applicable",
        )
        return None

    trade = trade_manager.get_open(state.bot_uuid)
    if not trade:
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="exit", decision="CLOSE",
            status="skipped", reason="No open position tracked for this bot",
        )
        return None

    veto = trade_manager.llm_close_veto_reason(
        trade, ask, bid, exit_result=exit_result,
    )
    if veto:
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="exit", decision="CLOSE",
            status="skipped", reason=veto,
        )
        return None

    closed = trade_manager.close_llm(
        state.bot_uuid, ask, bid, client,
        reasoning=exit_result.get("reasoning"),
        observations=exit_result.get("observations"),
    )
    if closed:
        _annotate_signal_exec(
            state, trigger_at=at, signal_type="exit", decision="CLOSE",
            status="executed", reason="Position closed on eToro demo",
        )
    return closed


def _strategy_exit_check(
    config: EngineConfig,
    state: "_EngineState",
    chart_data: pd.DataFrame,
    instrument_id: int,
    ask: float,
    bid: float,
    trigger_at: str,
    client: "EToroClient",
    bot_id: str = "",
    is_primary_bot: bool = False,
) -> "Optional[trade_manager.ClosedTrade]":
    """Run the rule-based strategy and close the position if signal reverses.

    For a LONG position: a SELL signal triggers a close.
    For a SHORT position: a BUY signal triggers a close.
    In both cases a HOLD keeps the position alive.

    Ownership guard: only closes the trade when this bot opened it.  A primary
    bot may receive position_open=True for a sibling-owned trade; the signal is
    still logged but no close is issued.  The signal is also stored in
    signal_worker so the Bots page can display it.
    """
    import strategies as _strats
    from strategies.execution_quality import assess as _eq_assess

    strategy = _strats.get(config.strategy_name)
    try:
        sig = strategy.generate(chart_data, ask, bid, instrument_id)
    except Exception:
        log.exception("Strategy %r exit-check failed", config.strategy_name)
        return None

    if sig is None:
        return None

    eq = _eq_assess(
        chart_data, ask, bid,
        strategy_key=config.strategy_name,
        confidence=sig.confidence,
        signal=sig.signal,
    )
    result = sig.to_result_dict(trigger_at)
    result["strategy"] = config.strategy_name
    result.update(eq.to_dict())

    # Store signal for display regardless of whether we close
    signal_worker.set_result_direct(
        instrument_id,
        config.interval_label,
        result,
        instrument_label=config.instrument_label,
        trigger_at=trigger_at,
        bot_id=state.bot_uuid,
    )
    log.info(
        "Strategy %r exit-check: signal=%s conf=%s for %s",
        config.strategy_name, sig.signal, sig.confidence, config.instrument_label,
    )

    trade = trade_manager.get_open(state.bot_uuid)
    if not trade:
        return None

    direction = trade.direction.upper()   # "LONG" or "SHORT"
    signal    = sig.signal.upper()        # "BUY" / "SELL" / "HOLD"

    should_close = (
        (direction == "LONG"  and signal == "SELL")
        or (direction == "SHORT" and signal == "BUY")
    )
    if not should_close:
        # Logged for the Signals tab but no entry/exit order — annotate so the
        # UI does not stay "pending" forever (common: BUY while already LONG).
        if signal == "HOLD":
            _annotate_signal_exec(
                state, trigger_at=trigger_at, signal_type="entry", decision=signal,
                status="not_applicable", reason="HOLD — no order while position open",
            )
        elif signal in ("BUY", "SELL"):
            _annotate_signal_exec(
                state, trigger_at=trigger_at, signal_type="entry", decision=signal,
                status="not_applicable",
                reason=(
                    f"No new order — open {direction} position; "
                    f"{signal} agrees with the current side"
                ),
            )
        return None

    # Profit gate: only realise a strategy-reversal exit when the position is in
    # GAIN above trading friction.  unrealised_pnl is the realisable profit per
    # price unit (already net of the exit-side spread); require it to also clear
    # the entry slippage and current spread.  If the position is flat or in loss
    # we HOLD on the reversal and let the hard stop-loss govern the downside —
    # we never close a reversal into a loss.
    pnl_unit = trade_manager.unrealised_pnl(trade, ask, bid)
    cushion  = max(float(trade.entry_spread or 0.0), float(ask - bid), 0.0)
    if pnl_unit <= cushion:
        log.info(
            "Strategy %r reversal %s suppressed on %s %s — not in gain above costs "
            "(pnl/unit=%.6f ≤ cushion=%.6f); holding, stop-loss governs downside",
            config.strategy_name, signal, direction, config.instrument_label,
            pnl_unit, cushion,
        )
        _annotate_signal_exec(
            state, trigger_at=trigger_at, signal_type="entry", decision=signal,
            status="skipped",
            reason=(
                "Strategy reversal suppressed — position not in gain above "
                f"spread costs (pnl/unit {pnl_unit:.6f} ≤ {cushion:.6f})"
            ),
        )
        return None

    log.info(
        "Strategy %r closing %s %s position (signal=%s, pnl/unit=%.6f > cushion=%.6f)",
        config.strategy_name, direction, config.instrument_label, signal,
        pnl_unit, cushion,
    )
    closed = trade_manager.close_llm(
        state.bot_uuid, ask, bid, client,
        reasoning=f"{config.strategy_name} generated {signal} while {direction} — strategy-driven exit in profit",
        observations=[f"Strategy: {config.strategy_name}", f"Signal: {signal}", f"Confidence: {sig.confidence}%"],
    )
    if closed:
        _annotate_signal_exec(
            state, trigger_at=trigger_at, signal_type="entry", decision=signal,
            status="executed", reason="Strategy reversal close executed on eToro demo",
        )
    return closed


def _dispatch_strategy(
    config: EngineConfig,
    chart_data: pd.DataFrame,
    instrument_id: int,
    ask: float,
    bid: float,
    trigger_at: str,
    bot_uuid: str = "",
) -> None:
    """Route the candle-close signal request to the active strategy."""
    import strategies  # lazy import — avoids circular dep at module load

    strategy = strategies.get(config.strategy_name)

    if strategy.is_async:
        # LLM and other async strategies fire a background job; result comes
        # back later via signal_worker.get_result().  Execution quality is
        # assessed inside signal_worker after the result arrives.
        strategy.generate(
            chart_data, ask, bid, instrument_id,
            instrument_label=config.instrument_label,
            interval_label=config.interval_label,
            trigger_at=trigger_at,
            bot_id=bot_uuid,
        )
    else:
        # Synchronous strategies return a StrategySignal immediately.
        try:
            sig = strategy.generate(chart_data, ask, bid, instrument_id)
        except Exception:
            log.exception(
                "Strategy %r failed for instrument %s", config.strategy_name, instrument_id
            )
            return
        if sig is not None:
            # Assess execution quality at signal generation time (age ≈ 0 s).
            from strategies.execution_quality import assess as _eq_assess
            eq = _eq_assess(
                chart_data, ask, bid,
                strategy_key=config.strategy_name,
                confidence=sig.confidence,
                signal=sig.signal,
            )
            result = sig.to_result_dict(trigger_at)
            result["strategy"] = config.strategy_name   # logged to signal_log + shown in Signals tab
            result.update(eq.to_dict())
            log.info(
                "Strategy %r signal %s conf=%s viable=%s net_edge=%+.3f%% risk=%s",
                config.strategy_name, sig.signal, sig.confidence,
                eq.viable, eq.net_edge_pct, eq.exec_risk,
            )
            signal_worker.set_result_direct(
                instrument_id,
                config.interval_label,
                result,
                instrument_label=config.instrument_label,
                trigger_at=trigger_at,
                bot_id=bot_uuid,
            )


def _bot_engine_active(config: EngineConfig, bot_key: str) -> bool:
    """True when the user has auto-trade enabled (not manually disabled)."""
    return bool(config.trading_active and bot_key not in _disabled_bots)


def is_user_auto_trade_enabled(instrument_id: int, bot_id: Optional[str] = None) -> bool:
    """User intent: auto-trade ON and not in the persisted OFF set."""
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        return bool(state and _bot_engine_active(state.config, key))


def _market_open_for_state(state: _EngineState) -> bool:
    """True when this instrument's market session allows trading (cached ~60s)."""
    try:
        return bool(state.client and state.client.is_market_open(state.config.instrument_id))
    except Exception:
        return True


def _sync_bot_market_hours(bot_key: str) -> None:
    """Stop bots while their market is closed; resume when it reopens if still enabled."""
    with _lock:
        state = _engines.get(bot_key)
        if state is None or not _bot_engine_active(state.config, bot_key):
            return
        wants_run = True
        running = state.running
        client = state.client
        iid = state.config.instrument_id
        label = state.config.instrument_label
    if not client:
        return
    try:
        open_ = client.is_market_open(iid)
    except Exception:
        return
    with _lock:
        state = _engines.get(bot_key)
        if state is None:
            return
        was_closed = state.market_closed
        state.market_closed = not open_
    if not open_ and running:
        stop_bot(bot_key)
        log.info("Market closed for %s — bot %s stopped", label, bot_key)
    elif open_ and was_closed and wants_run:
        _ensure_engine_thread(bot_key)
        log.info("Market reopened for %s — bot %s resumed", label, bot_key)


def _sync_all_market_hours() -> None:
    with _lock:
        keys = list(_engines.keys())
    for k in keys:
        _sync_bot_market_hours(k)


def _ensure_engine_thread(bot_key: str) -> None:
    """Start (or restart) the daemon thread + feeds for one registered bot."""
    with _lock:
        state = _engines.get(bot_key)
        if state is None:
            return
        if not _bot_engine_active(state.config, bot_key):
            return
        if not _market_open_for_state(state):
            state.running = False
            state.market_closed = True
            log.info(
                "Market closed for %s — bot %s idle until session opens",
                state.config.instrument_label, bot_key,
            )
            return
        state.market_closed = False
        state.running = True
        if state.thread is not None and state.thread.is_alive():
            return
        config = state.config
        client = state.client
        iid = config.instrument_id

    _sync_hub_config(config)

    t = threading.Thread(
        target=_instrument_loop, args=(bot_key,), daemon=True,
        name=f"engine-{bot_key}",
    )
    with _lock:
        state.thread = t
    t.start()
    log.info("Engine thread started for bot %s (%s)", bot_key, config.instrument_label)

    threading.Thread(
        target=_preload_hist, args=(bot_key, iid, config, client),
        daemon=True, name=f"hist-{bot_key}",
    ).start()

    tick_manager.start(iid, config.api_key, config.user_key)
    market_data_hub.set_desired_active(True)
    positions_cache.start_background_poller(client, config.is_demo)
    # Evidence-based bot ranking: periodically flag BLEEDING strategies whose
    # rolling profit factor shows negative expectancy; clear the flag when
    # evidence recovers.  Advisory only — never gates trading.  Idempotent.
    import bot_ranking
    bot_ranking.ensure_reviewer()
    ensure_supervisor()


def _run_tick(bot_id: str, state: _EngineState) -> None:
    """One tick iteration for a single bot."""
    config = state.config
    client = state.client
    instrument_id = config.instrument_id

    # Idle bots (auto-trade OFF, no position) skip all work — thread may still
    # exist briefly while shutting down or managing an orphaned position.
    if not _bot_engine_active(config, bot_id):
        if not trade_manager.has_open(state.bot_uuid):
            return

    chart = market_data_hub.get_snapshot(bot_id=bot_id)
    if chart is None or chart.instrument_id != instrument_id:
        return

    chart_data         = chart.chart_data      # committed + forming (live snapshot)
    committed_data     = chart.committed       # closed candles only (for candle-close signals)
    latest_ask    = chart.latest_ask
    latest_bid    = chart.latest_bid
    ticks_count   = chart.tick_count
    last_tick_time = chart.last_tick_time
    latest = (
        {"ask": latest_ask, "bid": latest_bid}
        if latest_ask or latest_bid
        else None
    )

    # Positions are refreshed by a dedicated background poller (started in
    # start_instrument), so the tick loop NEVER blocks on a REST round-trip.
    etoro_positions = positions_cache.get_positions()

    instrument_positions = _positions_for_instrument(etoro_positions, instrument_id)
    is_primary_bot = (_iid_to_primary.get(instrument_id) == bot_id)

    # ── Market hours (stocks) ─────────────────────────────────────────────────
    # Stocks only trade while their exchange is open; eToro rejects (or parks as
    # pending) orders sent outside the session.  When the market is closed we
    # keep monitoring/adoption/reconcile but issue NO orders and dispatch NO
    # signals (saves LLM calls overnight).  Crypto is 24/7 → always open.
    # is_market_open is cached (60s/instrument) so this never adds tick latency.
    market_open = client.is_market_open(instrument_id)
    if not market_open:
        if not state.market_closed:
            log.info(
                "Market closed for %s — stopping bot %s",
                config.instrument_label, bot_id,
            )
        state.market_closed = True
        stop_bot(bot_id)
        return
    if state.market_closed:
        log.info("Market reopened for %s — resuming bot %s",
                 config.instrument_label, bot_id)
        state.market_closed = False

    # ── Adoption (per bot) ────────────────────────────────────────────────────
    # Each bot re-adopts ITS OWN orphaned eToro position (matched via the
    # persisted position→owner map).  The primary bot additionally adopts an
    # UNCLAIMED position (no recorded owner, not held by any other bot) so
    # manual / pre-existing positions don't go unmanaged.
    if (
        latest
        and instrument_positions
        and not trade_manager.has_open(state.bot_uuid)
        and time.monotonic() >= state.skip_adopt_until
    ):
        held = trade_manager.held_position_ids()
        my_pos = None
        for p in instrument_positions:
            pid = p.get("position_id")
            if pid is not None and pid not in held and \
                    trade_manager.owner_of_position(pid) == state.bot_uuid:
                my_pos = p
                break
        if my_pos is None and is_primary_bot:
            for p in instrument_positions:
                pid = p.get("position_id")
                if pid is not None and pid not in held and \
                        trade_manager.owner_of_position(pid) is None:
                    my_pos = p
                    break
        if my_pos is not None:
            trade_manager.adopt_etoro_position(
                instrument_id, config.instrument_label, my_pos,
                latest["ask"], latest["bid"], bot_id=state.bot_uuid,
                strategy=config.strategy_name,
            )

    open_trade = trade_manager.get_open(state.bot_uuid)

    # The eToro position record for THIS bot's trade (matched by position id).
    etoro_pos = None
    if open_trade is not None and open_trade.etoro_position_id is not None:
        etoro_pos = next(
            (p for p in instrument_positions
             if str(p.get("position_id")) == str(open_trade.etoro_position_id)),
            None,
        )

    # Our tracked position vanished from eToro (closed in the app / server stop):
    # release it locally so the bot can trade again.  Debounced HARD to avoid a
    # false close: the position must be absent from a CONFIRMED-FRESH, non-empty
    # positions cache for VANISH_MISS_THRESHOLD consecutive ticks AND past the
    # grace period.  An empty cache (failed/partial poll) is NOT counted as a
    # miss, so a transient portfolio hiccup can never strip a live position.
    if (
        open_trade is not None
        and open_trade.etoro_position_id is not None
        and etoro_pos is None
        and bool(etoro_positions)            # cache returned data → trust absence
    ):
        state.vanish_misses += 1
    else:
        state.vanish_misses = 0

    if (
        open_trade is not None
        and latest
        and open_trade.etoro_position_id is not None
        and etoro_pos is None
        and state.vanish_misses >= VANISH_MISS_THRESHOLD
        and (datetime.now(tz=timezone.utc) - open_trade.entry_time).total_seconds() > VANISH_GRACE_SEC
    ):
        # FINAL confirmation against the live API, not the shared cache: a
        # partial/stale poll can hide a live position for several ticks, and a
        # false "external" close both journals phantom P&L and releases the bot
        # to re-adopt its own position (the duplicate-records loop).
        really_gone = False
        try:
            live_ids = client.position_ids_for_instrument(
                instrument_id, open_trade.direction == "LONG"
            )
            really_gone = open_trade.etoro_position_id not in live_ids
        except Exception as exc:
            log.warning(
                "Vanish confirmation lookup failed for %s — keeping position: %s",
                open_trade.etoro_position_id, exc,
            )
        if not really_gone:
            # The cache lied — position is alive.  Reset and keep managing it.
            state.vanish_misses = 0
        else:
            # The position is gone from eToro but THIS bot never issued the close —
            # it was closed on eToro's side (its own SL/TP, the eToro app, or a merge).
            # Record it as "external", not "manual", so the P&L view is honest.
            cleared = trade_manager.close_manual(
                state.bot_uuid, latest["ask"], latest["bid"], client=None,
                reason="external",
            )
            if cleared:
                with _lock:
                    _last_closes[instrument_id] = cleared
                _bump_portfolio()
            open_trade = None
            state.vanish_misses = 0

    if open_trade and latest:
        trade_manager.update_peak_pnl(
            open_trade, latest["ask"], latest["bid"], etoro_pos=etoro_pos,
        )

    # get_open is bot-keyed, so the returned trade is always THIS bot's.
    trade_owned_by_us = open_trade is not None

    # Reconcile this bot's trade — position ID + entry_time — from eToro's real
    # order data.  Cheap & idempotent: short-circuits once both are synced.
    if (
        open_trade is not None
        and etoro_pos is not None
        and (open_trade.etoro_position_id is None or not open_trade.etoro_open_time_synced)
    ):
        trade_manager.reconcile_from_etoro(state.bot_uuid, etoro_pos)

    if latest and trade_owned_by_us and market_open:
        # ── Protective-exit ladder ────────────────────────────────────────────
        # SEQUENTIAL by priority — each check runs unless an earlier one already
        # closed the trade.  (A previous elif-chain meant a bot with BOTH a
        # take-profit and a trailing stop — the whole mean-revert family — never
        # ran its trailing check at all: TP's branch masked it every tick.)
        #   1. hard stop-loss (incl. breakeven floor)   — downside, always first
        #   2. recovery exit / breakeven-floor arming   — overstayed-red rescue
        #   3. take-profit target                       — banks the move up
        #   4. chandelier ATR trailing stop             — protects the peak
        ask_, bid_ = latest["ask"], latest["bid"]
        closed = trade_manager.check_stop_loss(state.bot_uuid, ask_, bid_, client)
        if not closed:
            try:
                import user_settings
                _beh = user_settings.behavior_settings()
                _rec_on = _beh.recovery_exit_enabled
                _rec_mult = float(_beh.recovery_hold_mult)
                _rec_be = bool(getattr(_beh, "recovery_breakeven_stop", True))
            except Exception:
                _rec_on, _rec_mult, _rec_be = True, 2.5, True
            closed = trade_manager.check_recovery_exit(
                state.bot_uuid, ask_, bid_,
                config.strategy_name, client,
                enabled=_rec_on, hold_mult=_rec_mult,
                breakeven_stop=_rec_be,
            )
        if not closed and config.take_profit_pct > 0:
            closed = trade_manager.check_take_profit(
                state.bot_uuid, ask_, bid_, config.take_profit_pct, client,
            )
        if not closed and config.trailing_stop_pct > 0:
            # Chandelier ATR trail (golden rule 2xATR).  Live ATR% comes from
            # the per-candle regime memo — zero extra cost per tick; without
            # ATR the legacy %-from-peak trail applies unchanged.
            _atr_now = None
            try:
                if chart_data is not None and len(chart_data) > 0:
                    _atr_now = _classify_regime_memo(instrument_id, chart_data).atr_pct or None
            except Exception:
                _atr_now = None
            import exit_profiles as _ep
            closed = trade_manager.check_trailing_stop(
                state.bot_uuid, ask_, bid_,
                config.trailing_stop_pct, client,
                atr_pct=_atr_now,
                atr_mult=_ep.atr_trail_mult(config.strategy_name, config.bot_id),
            )
        if closed:
            with _lock:
                _last_closes[instrument_id] = closed
            _bump_portfolio()
            engine_notify.push(
                "trade_close",
                f"Closed {config.instrument_label} ({closed.reason or 'close'})"
                f" — ${closed.pnl_dollars:+.2f} ({closed.profit:+.5f} px)",
                instrument_id=instrument_id,
            )

    last_committed_time = chart.last_committed_time
    new_candle_closed = False
    htf_signal_data: Optional[pd.DataFrame] = None

    if config.interval_seconds >= market_data_hub.HTF_HIST_COMMITTED_SEC:
        # HTF committed bars come from eToro hist, not ticks — poll the API for
        # a new closed bar instead of relying on tick-bucket rollover.
        now_mono = time.monotonic()
        if now_mono - state.last_htf_hist_poll >= HTF_HIST_POLL_SEC:
            state.last_htf_hist_poll = now_mono
            htf_signal_data = _refresh_htf_hist_at_close(bot_id, config, client)
            if htf_signal_data is not None and not htf_signal_data.empty:
                hist_last = htf_signal_data["time"].iloc[-1]
                if state.prev_candle_time is None:
                    state.prev_candle_time = hist_last
                elif hist_last != state.prev_candle_time:
                    new_candle_closed = True
                    last_committed_time = hist_last
                    state.prev_candle_time = hist_last
    else:
        new_candle_closed = (
            last_committed_time is not None
            and state.prev_candle_time is not None
            and last_committed_time != state.prev_candle_time
        )
        if last_committed_time is not None and state.prev_candle_time is None:
            state.prev_candle_time = last_committed_time
        elif new_candle_closed:
            state.prev_candle_time = last_committed_time

    # position_open: this bot's own trade (the store is bot-keyed, so get_open
    # never returns a sibling's position).
    _open_trade = trade_manager.get_open(state.bot_uuid)
    position_open = _open_trade is not None

    if new_candle_closed and latest and market_open:
        trigger_at = last_committed_time.strftime("%H:%M:%S")
        # Use committed-only candles so strategies analyse the just-CLOSED candle
        # as df.iloc[-1], not the newly forming (incomplete) candle.
        if htf_signal_data is not None and not htf_signal_data.empty:
            signal_data = htf_signal_data
        else:
            signal_data = committed_data if not committed_data.empty else chart_data
        import strategies as _strats
        _strat = _strats.get(config.strategy_name)
        if position_open:
            if _strat.is_async:
                # LLM: delegate exit decision to visual-bot
                pos_ctx = trade_manager.build_exit_position_context(
                    latest["ask"],
                    latest["bid"],
                    trade=trade_manager.get_open(state.bot_uuid),
                    etoro_pos=etoro_pos,
                )
                if pos_ctx:
                    signal_worker.request_exit_signal(
                        signal_data,
                        instrument_id,
                        config.instrument_label,
                        config.interval_label,
                        pos_ctx,
                        trigger_at=trigger_at,
                        bot_id=state.bot_uuid,
                    )
            else:
                # Rule-based strategy: re-run the strategy and close if the
                # signal reverses direction relative to the open position.
                _closed = _strategy_exit_check(
                    config, state, signal_data, instrument_id,
                    latest["ask"], latest["bid"], trigger_at, client,
                    bot_id=bot_id, is_primary_bot=is_primary_bot,
                )
                if _closed:
                    with _lock:
                        _last_closes[instrument_id] = _closed
                    _bump_portfolio()
                    engine_notify.push(
                        "trade_close",
                        f"Closed {config.instrument_label} — ${_closed.pnl_dollars:+.2f} ({_closed.profit:+.5f} px)",
                        instrument_id=instrument_id,
                    )
        elif config.trading_active:
            state.pending_dispatch = False   # candle close satisfies it
            _dispatch_strategy(
                config, signal_data, instrument_id,
                latest["ask"], latest["bid"], trigger_at,
                bot_uuid=state.bot_uuid,
            )

    # Immediate dispatch when auto-trade is first enabled (don't wait for candle close)
    if (
        state.pending_dispatch
        and config.trading_active
        and not position_open
        and latest
        and market_open
        and chart_data is not None
        and not chart_data.empty
    ):
        state.pending_dispatch = False
        trigger_at = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        log.info("Immediate dispatch on auto-trade enable for instrument %s", instrument_id)
        _dispatch_strategy(
            config, chart_data, instrument_id,
            latest["ask"], latest["bid"], trigger_at,
            bot_uuid=state.bot_uuid,
        )

    sig_result = signal_worker.get_result(instrument_id, config.interval_label, state.bot_uuid)
    if sig_result and latest and not position_open and market_open:
        _maybe_open_trade(config, client, instrument_id, sig_result,
                          latest["ask"], latest["bid"], state)
        position_open = trade_manager.get_open(state.bot_uuid) is not None

    # LLM-driven exit: only relevant for async (LLM) strategies.
    # Rule-based strategies handle exit inside _strategy_exit_check above.
    import strategies as _strats_exit
    if _strats_exit.get(config.strategy_name).is_async:
        exit_result = signal_worker.get_exit_result(instrument_id, config.interval_label, state.bot_uuid)
        if exit_result and latest and position_open and market_open:
            closed = _maybe_process_exit(
                instrument_id, exit_result, latest["ask"], latest["bid"], client, state
            )
            if closed:
                with _lock:
                    _last_closes[instrument_id] = closed
                _bump_portfolio()
                engine_notify.push(
                    "trade_close",
                    f"Closed {config.instrument_label} — ${closed.pnl_dollars:+.2f} ({closed.profit:+.5f} px)",
                    instrument_id=instrument_id,
                )

    snap = EngineSnapshot(
        instrument_id=instrument_id,
        instrument_label=config.instrument_label,
        interval_label=config.interval_label,
        latest_ask=latest_ask,
        latest_bid=latest_bid,
        tick_count=ticks_count,
        last_tick_time=last_tick_time,
        trading_active=config.trading_active,
        position_open=position_open,
        bot_id=bot_id,
        bot_uuid=state.bot_uuid,
        started_at=state.started_at,
    )
    with _lock:
        if bot_id in _engines:
            _engines[bot_id].snapshot = snap


def _instrument_loop(bot_id: str) -> None:
    """Tick loop for one bot — runs in its own daemon thread."""
    while True:
        with _lock:
            state = _engines.get(bot_id)
            if state is None or not state.running:
                break
        try:
            _run_tick(bot_id, state)
        except Exception:
            log.exception("Engine tick failed for bot %s", bot_id)
        # Adaptive cadence: hold 1s granularity while a position is open (exits
        # must react fast) or a dispatch is pending (immediate entry on enable);
        # otherwise poll slower since a flat bot only trades on candle closes.
        try:
            active = _bot_engine_active(state.config, bot_id)
            has_pos = trade_manager.has_open(state.bot_uuid)
            busy = state.pending_dispatch or has_pos
        except Exception:
            active, busy = True, True
        if not active and not busy:
            time.sleep(IDLE_TICK_INTERVAL)
        else:
            time.sleep(TICK_INTERVAL if busy else FLAT_TICK_INTERVAL)


# ── Public API — multi-instrument ─────────────────────────────────────────────

def start_instrument(
    spec,                        # InstrumentSpec from instrument_config
    api_key: Optional[str] = None,
    user_key: Optional[str] = None,
    hist_df: Optional[pd.DataFrame] = None,
    *,
    is_demo: bool = True,
) -> None:
    """
    Start (or restart) a background engine.

    Accepts either an InstrumentSpec (boot-time) or an EngineConfig (UI call).
    Multiple bots for the same instrument_id can run at different intervals;
    each is identified by its unique bot_id (= instruments.toml section key).
    """
    import exit_profiles
    from instrument_config import InstrumentSpec as _Spec
    # Per-bot toml overrides for exit params (None = follow the strategy profile)
    _exit_overrides: tuple[Optional[float], Optional[float]] = (None, None)
    if isinstance(spec, _Spec):
        _exit_overrides = (spec.trailing_stop_pct, spec.take_profit_pct)
        config = EngineConfig(
            instrument_id=spec.instrument_id,
            instrument_label=spec.label,
            interval_label=spec.interval,
            interval_seconds=spec.interval_secs,
            candle_count=spec.candle_count,
            trading_active=spec.auto_trade,
            demo_amount=spec.demo_amount,
            is_demo=is_demo,
            api_key=api_key or "",
            user_key=user_key or "",
            strategy_name=spec.strategy,
            bot_id=spec.key,
        )
    else:
        config = spec
        api_key  = api_key  or config.api_key
        user_key = user_key or config.user_key

    iid    = config.instrument_id
    bot_key = _engine_key(config)

    # Shared client + stable UUID are needed for BOTH the running and the
    # disabled-but-registered paths below, so resolve them up front.  Both are
    # cheap / cached and start no feed.
    client = get_shared_client(config.api_key, config.user_key)
    bot_uuid = bot_registry.get_or_create(bot_key)

    # Register the AUTHORITATIVE asset class from eToro's instrument metadata
    # (instrumentTypeID) so exit sizing and the LLM prompt never rely on label
    # keywords.  Cached in etoro_client, so this is one API call per instrument.
    _klass = client.asset_class_for(iid)
    if _klass:
        exit_profiles.register_asset_class(config.instrument_label, _klass)

    # Resolve exit params AFTER class registration so the volatility scaling
    # uses the API-provided class (toml overrides still win when set).
    _trail, _tp = exit_profiles.resolve(
        config.strategy_name, *_exit_overrides,
        instrument_label=config.instrument_label,
        bot_key=bot_key,
    )
    config = replace(config, trailing_stop_pct=_trail, take_profit_pct=_tp)

    # AUTHORITATIVE OFF: a bot the user turned off must not be revived by a
    # background rerun (sync_background_engine → update_from_ui → start_instrument
    # runs on every tab switch).  We STILL register the engine — so its real
    # strategy / interval / UUID are known to the UI (otherwise get_strategy()
    # returns its "llm" default and every off bot wrongly shows LLM) and a later
    # turn-ON can revive it cleanly — but we leave it stopped and never start its
    # thread or feed.  The Bots-tab toggle calls set_auto_trade(True) first, which
    # clears the disable, so an explicit turn-ON starts the bot normally.
    with _lock:
        if bot_key in _disabled_bots:
            existing = _engines.get(bot_key)
            if existing is not None:
                existing.config = replace(
                    config,
                    trading_active=False,
                    strategy_name=existing.config.strategy_name,
                )
                existing.running = False
                existing.client = client
                existing.bot_uuid = bot_uuid
            else:
                stopped = _EngineState(
                    config=replace(config, trading_active=False),
                    client=client,
                    running=False,
                )
                stopped.bot_uuid = bot_uuid
                _engines[bot_key] = stopped
                if iid not in _iid_to_primary:
                    _iid_to_primary[iid] = bot_key
            return

    _sync_hub_config(config)

    if hist_df is not None and not hist_df.empty:
        market_data_hub.set_hist(iid, hist_df, bot_id=bot_key)

    with _lock:
        existing = _engines.get(bot_key)
        if existing is not None:
            # Preserve fields managed exclusively by set_auto_trade / set_strategy.
            config = replace(
                config,
                trading_active=existing.config.trading_active,
                strategy_name=existing.config.strategy_name,
            )
            existing.config = config
            existing.client = client
            existing.bot_uuid = bot_uuid
            if (
                existing.running
                and existing.thread
                and existing.thread.is_alive()
                and _bot_engine_active(config, bot_key)
            ):
                return
            state = existing
        else:
            new_state = _EngineState(
                config=config,
                client=client,
                running=False,
            )
            new_state.bot_uuid = bot_uuid
            _engines[bot_key] = new_state
            if iid not in _iid_to_primary:
                _iid_to_primary[iid] = bot_key
            state = new_state

    if not _bot_engine_active(config, bot_key):
        with _lock:
            state.running = False
        market_data_hub.stop(bot_id=bot_key)
        log.info("Engine registered (idle) for bot %s / %s", bot_key, config.instrument_label)
        return

    _ensure_engine_thread(bot_key)
    log.info("Engine started for bot %s / instrument %s (%s)", bot_key, iid, config.instrument_label)


def _preload_hist(bot_key: str, iid: int, config: EngineConfig, client: EToroClient) -> None:
    """Load historical candles in a background thread and feed to hub."""
    try:
        # Shared cache: all bots on this (instrument, interval) reuse one fetch
        # instead of each hitting eToro on boot.
        df = client.get_hist_candles_cached(iid, config.interval_seconds, config.candle_count + 20)
        if df is not None and not df.empty:
            market_data_hub.set_hist(iid, df, bot_id=bot_key)
            log.info("Hist loaded for %s (%s): %d candles", config.instrument_label, bot_key, len(df))
    except Exception as exc:
        log.warning("Hist preload failed for %s (%s): %s", config.instrument_label, bot_key, exc)


def _refresh_htf_hist_at_close(
    bot_key: str,
    config: EngineConfig,
    client: EToroClient,
) -> Optional[pd.DataFrame]:
    """Fetch fresh eToro OHLC at HTF candle close; return committed bars for signals."""
    try:
        count = config.candle_count + 20
        df = client.get_hist_candles_cached(
            config.instrument_id,
            config.interval_seconds,
            count,
            ttl=0.0,
        )
        if df is None or df.empty:
            log.warning(
                "HTF hist refresh empty for %s (%s)", bot_key, config.interval_label,
            )
            return None
        snap = market_data_hub.refresh_stream_hist(
            config.instrument_id,
            config.interval_seconds,
            df,
        )
        if snap is not None and not snap.committed.empty:
            log.info(
                "HTF hist refreshed at candle close for %s (%s) — %d committed bars",
                bot_key, config.interval_label, len(snap.committed),
            )
            return snap.committed
        trimmed = df.tail(int(config.candle_count))
        log.info(
            "HTF hist refreshed (hist-only) for %s (%s) — %d bars",
            bot_key, config.interval_label, len(trimmed),
        )
        return trimmed
    except Exception:
        log.warning(
            "HTF hist refresh failed for %s (%s)",
            bot_key, config.interval_label, exc_info=True,
        )
        return None


def stop_instrument(instrument_id: int) -> None:
    """Stop all engines for one instrument (by iid)."""
    with _lock:
        bot_keys = [k for k, s in _engines.items() if s.config.instrument_id == instrument_id]
        for k in bot_keys:
            _engines[k].running = False
    # Only stop tick stream if no other bots need it
    tick_manager.stop(instrument_id)
    market_data_hub.stop(instrument_id)
    log.info("Engine stopped for instrument %s", instrument_id)


def stop_bot(bot_id: str) -> None:
    """Stop a specific bot by bot_id."""
    iid: Optional[int] = None
    with _lock:
        state = _engines.get(bot_id)
        if state:
            iid = state.config.instrument_id
            state.running = False
    if iid is not None:
        # Stop tick stream only if no other bots for this iid are still running
        with _lock:
            others_running = any(
                s.running for k, s in _engines.items()
                if k != bot_id and s.config.instrument_id == iid
            )
        if not others_running:
            tick_manager.stop(iid)
    market_data_hub.stop(bot_id=bot_id)
    log.info("Bot %s stopped", bot_id)


def delete_bot(bot_id: str) -> tuple[bool, str]:
    """Permanently remove a bot from the registry.

    Returns (True, "") on success, or (False, reason) when the bot cannot be
    deleted because it is still active or owns an open position.
    """
    with _lock:
        state = _engines.get(bot_id)
        if state is None:
            return False, "Bot not found."
        if state.config.trading_active:
            return False, "Disable auto-trade first."
        iid = state.config.instrument_id
        if trade_manager.get_open(state.bot_uuid) is not None:
            return False, "Close the open position first."
        # Safe — stop and remove
        state.running = False
        del _engines[bot_id]
        _disabled_bots.discard(bot_id)
        # Re-assign primary for this iid if needed
        if _iid_to_primary.get(iid) == bot_id:
            new_primary = next(
                (k for k, s in _engines.items() if s.config.instrument_id == iid), None
            )
            if new_primary:
                _iid_to_primary[iid] = new_primary
            else:
                _iid_to_primary.pop(iid, None)

    _save_disabled()
    market_data_hub.stop(bot_id=bot_id)
    market_data_hub.remove(iid, bot_id=bot_id)
    bot_registry.remove(bot_id)
    log.info("Bot %s (UUID %s) deleted", bot_id, state.bot_uuid)
    return True, ""


def stop_all() -> None:
    """Stop all running engines."""
    with _lock:
        bot_keys = list(_engines.keys())
    for k in bot_keys:
        with _lock:
            state = _engines.get(k)
            if state:
                state.running = False
    tick_manager.stop_all()
    market_data_hub.stop()
    log.info("All engines stopped")


def configured_instruments() -> dict[str, int]:
    """label → instrument_id for every registered engine (Backtest page)."""
    with _lock:
        return {
            s.config.instrument_label: s.config.instrument_id
            for s in _engines.values()
            if s.config.instrument_id and s.config.instrument_label
        }


def _resolve_bot_key(instrument_id: int, bot_id: Optional[str] = None) -> Optional[str]:
    """Return the engine key to use: explicit bot_id wins, else primary for iid."""
    if bot_id is not None:
        return bot_id
    return _iid_to_primary.get(instrument_id)


def _default_bot_for(iid: int, label: str = "") -> Optional[str]:
    """First configured bot key for an instrument (matched by id, else label).

    Used by the UI entry points when no engine exists yet for the instrument
    (e.g. a Trading-tab sync or Start-button callback firing before app boot).
    Binding to a real configured bot instead of minting a str(iid)-keyed legacy
    engine keeps trades attributed and keeps stray engines from shadowing the
    configured fleet."""
    try:
        import instrument_config
        for spec in instrument_config.load_specs():
            if (spec.instrument_id and spec.instrument_id == iid) or (
                label and spec.label == label
            ):
                return spec.key
    except Exception:
        pass
    return None


def get_snapshot(
    instrument_id: Optional[int] = None,
    bot_id: Optional[str] = None,
) -> Optional[EngineSnapshot]:
    """Return trading snapshot by bot_id, instrument_id, or the active bot."""
    with _lock:
        if bot_id is not None:
            state = _engines.get(bot_id)
            return state.snapshot if state else None
        iid = instrument_id if instrument_id is not None else _active_iid
        if iid is None:
            return None
        key = _iid_to_primary.get(iid)
        state = _engines.get(key) if key else None
        return state.snapshot if state else None


def get_snapshot_by_uuid(bot_uuid: str) -> Optional[EngineSnapshot]:
    """Return the snapshot for a bot identified by its UUID (not the TOML key)."""
    with _lock:
        for state in _engines.values():
            if state.bot_uuid == bot_uuid and state.snapshot is not None:
                return state.snapshot
    return None


def get_all_snapshots() -> dict[str, EngineSnapshot]:
    """Return snapshots for every running bot (bot_id → EngineSnapshot)."""
    with _lock:
        return {
            bot_key: s.snapshot
            for bot_key, s in _engines.items()
            if s.running and s.snapshot is not None
        }


def get_config(
    instrument_id: Optional[int] = None,
    bot_id: Optional[str] = None,
) -> Optional[EngineConfig]:
    with _lock:
        if bot_id is not None:
            state = _engines.get(bot_id)
            return state.config if state else None
        iid = instrument_id if instrument_id is not None else _active_iid
        if iid is None:
            return None
        key = _iid_to_primary.get(iid)
        state = _engines.get(key) if key else None
        return state.config if state else None


def set_auto_trade(instrument_id: int, active: bool, bot_id: Optional[str] = None) -> None:
    """Toggle auto-trade for one bot (or the primary bot for instrument_id)."""
    key: Optional[str] = None
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        if state:
            was_active = state.config.trading_active
            state.config = replace(state.config, trading_active=active)
            if active and not was_active:
                state.pending_dispatch = True
        if key:
            if active:
                _disabled_bots.discard(key)
            else:
                _disabled_bots.add(key)
    _save_disabled()
    if key:
        if active:
            _ensure_engine_thread(key)
        else:
            stop_bot(key)
    log.info("Auto-trade %s for bot %s (iid=%s)", "ON" if active else "OFF", bot_id or key, instrument_id)


def is_auto_trade(instrument_id: int, bot_id: Optional[str] = None) -> bool:
    """Effective auto-trade: user enabled AND the instrument's market is open."""
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        if not (state and _bot_engine_active(state.config, key)):
            return False
    return _market_open_for_state(state)


def refresh_all_exit_params() -> int:
    """Re-apply exit profiles / Settings overrides on every running bot."""
    import exit_profiles
    import instrument_config

    updated = 0
    with _lock:
        keys = list(_engines.keys())
    spec_by_key = {s.key: s for s in instrument_config.load_specs()}
    for key in keys:
        with _lock:
            state = _engines.get(key)
            if not state:
                continue
            cfg = state.config
        spec = spec_by_key.get(key)
        _trail, _tp = exit_profiles.resolve(
            cfg.strategy_name,
            trailing_override=spec.trailing_stop_pct if spec else None,
            take_profit_override=spec.take_profit_pct if spec else None,
            instrument_label=cfg.instrument_label,
            bot_key=key,
        )
        with _lock:
            state = _engines.get(key)
            if state:
                state.config = replace(
                    state.config, trailing_stop_pct=_trail, take_profit_pct=_tp,
                )
                updated += 1
    if updated:
        log.info("Refreshed exit params on %d bot(s)", updated)
    return updated


def set_strategy(instrument_id: int, strategy_name: str, bot_id: Optional[str] = None) -> None:
    """Change the active strategy for a running bot without restarting it.

    Also re-aligns the exit params (trailing stop / take-profit) to the new
    strategy's profile so the exit behaviour matches the strategy in use."""
    import exit_profiles
    _trail = _tp = float("nan")
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        if state:
            _trail, _tp = exit_profiles.resolve(
                strategy_name,
                instrument_label=state.config.instrument_label,
                bot_key=key,
            )
            state.config = replace(
                state.config, strategy_name=strategy_name,
                trailing_stop_pct=_trail, take_profit_pct=_tp,
            )
    log.info(
        "Strategy set to %r for bot %s (iid=%s) — exits: trail=%.2f%% tp=%.2f%%",
        strategy_name, bot_id or key, instrument_id, _trail, _tp,
    )


def get_strategy(instrument_id: int, bot_id: Optional[str] = None) -> str:
    """Return the active strategy key for a bot."""
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        return state.config.strategy_name if state else "llm"


def engine_count() -> int:
    """Number of registered engines (bots).  Lives in this module — which is
    imported once and persists for the whole process — so callers can use it as
    a reliable 'already booted' guard that survives Streamlit's per-rerun
    re-execution of the main script (where module-level flags get reset)."""
    with _lock:
        return len(_engines)


def engine_keys() -> list[str]:
    """Keys of all registered engines (configured bot keys, or legacy iid-strings)."""
    with _lock:
        return list(_engines)


# True only after app boot has registered the full instruments.toml fleet in
# THIS process.  engine_count()>0 is NOT a valid substitute: a single stray
# engine created by a UI path before boot (e.g. an instrument-id-keyed legacy
# engine) once masked the boot guard and left all configured bots unstarted
# for hours with no errors anywhere.
_fleet_booted: bool = False


def fleet_booted() -> bool:
    return _fleet_booted


def mark_fleet_booted() -> None:
    global _fleet_booted
    _fleet_booted = True


def auto_trade_count() -> int:
    """Bots effectively trading (user ON and market open)."""
    with _lock:
        items = list(_engines.items())
    return sum(
        1 for k, s in items
        if _bot_engine_active(s.config, k) and _market_open_for_state(s)
    )


def active_engine_count() -> int:
    """Bots with a live tick loop (auto-trade ON or holding a position)."""
    with _lock:
        return sum(1 for s in _engines.values() if s.running)


def set_all_auto_trade(active: bool) -> None:
    """Enable or disable auto-trade for every registered bot at once."""
    with _lock:
        keys = list(_engines.keys())
    for k in keys:
        with _lock:
            state = _engines.get(k)
            if state:
                was = state.config.trading_active
                state.config = replace(state.config, trading_active=active)
                if active and not was:
                    state.pending_dispatch = True
    with _lock:
        # Global enable clears all per-bot disables; global disable marks all off.
        if active:
            _disabled_bots.clear()
        else:
            _disabled_bots.update(keys)
    _save_disabled()
    for k in keys:
        if active:
            _ensure_engine_thread(k)
        else:
            stop_bot(k)
    log.info("All auto-trade set to %s (%d bots)", "ON" if active else "OFF", len(keys))


def restore_auto_trade(active: bool) -> None:
    """Boot-time restore that RESPECTS the persisted per-bot OFF set.

    Used instead of set_all_auto_trade(True) at startup so a bot the user
    individually turned off (and which we persisted) stays off across restarts,
    even when the session's global auto-trade flag was on."""
    with _lock:
        items = list(_engines.items())
    for k, _ in items:
        on = active and (k not in _disabled_bots)
        was = False
        with _lock:
            s = _engines.get(k)
            if s:
                was = s.config.trading_active
                s.config = replace(s.config, trading_active=on)
                if on and not was:
                    s.pending_dispatch = True
        if on and not was:
            _ensure_engine_thread(k)
        elif not on and was:
            stop_bot(k)
    log.info(
        "Boot restore: auto-trade=%s, honoring %d persisted-OFF bot(s) %s",
        active, len(_disabled_bots), sorted(_disabled_bots),
    )


def suppress_adopt(instrument_id: int, seconds: float = ADOPT_SUPPRESS_SEC, bot_id: Optional[str] = None) -> None:
    """After a manual close, avoid re-adopting a stale cached eToro position."""
    with _lock:
        key = _resolve_bot_key(instrument_id, bot_id)
        state = _engines.get(key) if key else None
        if state:
            state.skip_adopt_until = time.monotonic() + seconds
    log.info("Suppress adopt for bot %s / iid %s (%.0fs)", bot_id or key, instrument_id, seconds)


# ── Shared notifications / P&L ────────────────────────────────────────────────

def consume_portfolio_bump() -> bool:
    global _portfolio_bump
    with _lock:
        if _portfolio_bump:
            _portfolio_bump = False
            return True
        return False


def pop_last_close(instrument_id: int) -> Optional[trade_manager.ClosedTrade]:
    with _lock:
        return _last_closes.pop(instrument_id, None)


def get_trade_error(instrument_id: int) -> Optional[str]:
    with _lock:
        return _trade_errors.get(instrument_id)


def clear_trade_error(instrument_id: int) -> None:
    with _lock:
        _trade_errors.pop(instrument_id, None)


# ── Supervisor (shared across all instruments) ────────────────────────────────

_watchdog_alert_at: float = 0.0


def _signal_watchdog() -> None:
    """Silent-fleet detector — the Jun-11 failure mode (engines registered,
    zero signal evaluation for nine hours, no errors anywhere) must never be
    silent again.  If any engine thread is RUNNING but the signal log has not
    been written for ~2.5x the smallest running interval (min 35 min), raise
    an engine notification + error log, repeating at most hourly."""
    global _watchdog_alert_at
    try:
        with _lock:
            running_secs = [
                s.config.interval_seconds for s in _engines.values()
                if s.running and s.config.trading_active
            ]
        if not running_secs:
            return
        import signal_log as _slog
        try:
            age = time.time() - _slog.LOG_PATH.stat().st_mtime
        except OSError:
            age = float("inf")
        threshold = max(2100.0, 2.5 * float(min(running_secs)))
        if age < threshold:
            return
        now = time.time()
        if now - _watchdog_alert_at < 3600.0:
            return
        _watchdog_alert_at = now
        msg = (
            f"WATCHDOG: {len(running_secs)} bot(s) running but no signal has "
            f"been written for {age / 60.0:.0f} min — signal evaluation may "
            f"be stalled.  Check engine threads / restart the container."
        )
        log.error(msg)
        engine_notify.push("watchdog", msg)
    except Exception:
        log.debug("watchdog check failed open", exc_info=True)


def _supervisor_loop() -> None:
    while True:
        try:
            _sync_all_market_hours()
            _signal_watchdog()
            if _desired_live:
                with _lock:
                    items = list(_engines.items())
                for bot_key, state in items:
                    if not state.running:
                        continue
                    iid = state.config.instrument_id
                    tick_manager.start(iid, state.config.api_key, state.config.user_key)
                    if state.thread is None or not state.thread.is_alive():
                        log.info("Supervisor restarting engine for bot %s", bot_key)
                        t = threading.Thread(
                            target=_instrument_loop, args=(bot_key,),
                            daemon=True, name=f"engine-{bot_key}",
                        )
                        with _lock:
                            state.thread = t
                        t.start()
            else:
                with _lock:
                    any_running = any(s.running for s in _engines.values())
                if any_running:
                    stop_all()
        except Exception:
            log.exception("Engine supervisor failed")
        time.sleep(10)


def ensure_supervisor() -> None:
    global _supervisor_thread, _supervisor_started
    if _supervisor_started and _supervisor_thread and _supervisor_thread.is_alive():
        return
    _supervisor_started = True
    _supervisor_thread = threading.Thread(
        target=_supervisor_loop, daemon=True, name="engine-supervisor",
    )
    _supervisor_thread.start()
    log.info("Engine supervisor started")


# ── Backwards-compatible API (Trading tab — operates on _active_iid) ──────────

def ensure_running(config: EngineConfig, hist_df: Optional[pd.DataFrame] = None) -> None:
    """Start or update the engine (Trading-tab entry point).

    When config.bot_id is empty, the call is routed to the primary bot for
    config.instrument_id so Trading-tab reruns don't create duplicate engines.
    """
    global _active_iid, _active_bot_id
    iid = config.instrument_id
    # Resolve bot_id: if not set, reuse the primary bot registered for this iid;
    # before boot (no engines yet) fall back to the first configured bot so we
    # never mint a legacy str(iid)-keyed engine for a configured instrument.
    if not config.bot_id:
        with _lock:
            primary = _iid_to_primary.get(iid)
        if not primary:
            primary = _default_bot_for(iid, config.instrument_label)
        if primary:
            config = replace(config, bot_id=primary)
    with _lock:
        _active_iid = iid
        _active_bot_id = config.bot_id or str(iid)
    start_instrument(config, hist_df=hist_df)


def update_from_ui(config: EngineConfig, hist_df: Optional[pd.DataFrame] = None) -> None:
    """Called from Streamlit to push latest user settings to the background engine."""
    global _active_iid, _active_bot_id
    iid = config.instrument_id
    # Resolve bot_id before configuring the hub so both point to the same key.
    # Same fallback chain as ensure_running: primary, else first configured bot.
    if not config.bot_id:
        with _lock:
            primary = _iid_to_primary.get(iid)
        if not primary:
            primary = _default_bot_for(iid, config.instrument_label)
        if primary:
            config = replace(config, bot_id=primary)
    with _lock:
        _active_iid = iid
        _active_bot_id = config.bot_id or str(iid)
    market_data_hub.configure(
        market_data_hub.HubConfig(
            instrument_id=iid,
            interval_label=config.interval_label,
            interval_seconds=config.interval_seconds,
            candle_count=config.candle_count,
            bot_id=config.bot_id,
        ),
        hist_df=hist_df,
    )
    # Ticks are per-instrument — start the feed for the viewed bot's chart even
    # when auto-trade is OFF (hub build + live chart still need ticks).
    if config.api_key and config.user_key:
        tick_manager.start(iid, config.api_key, config.user_key)
    ensure_running(config, hist_df)


def set_desired_live(enabled: bool) -> None:
    global _desired_live
    _desired_live = enabled
    market_data_hub.set_desired_active(enabled)
    if not enabled:
        stop_all()
    ensure_supervisor()


def is_desired_live() -> bool:
    return _desired_live


def set_desired_trading_active(active: bool) -> None:
    """Set auto-trade for the currently active (Trading-tab) instrument."""
    with _lock:
        iid = _active_iid
    if iid is not None:
        set_auto_trade(iid, active)
    log.info("Auto-trade desired (active instrument): %s", "ON" if active else "OFF")


def is_desired_trading_active() -> bool:
    with _lock:
        iid = _active_iid
    if iid is not None:
        return is_auto_trade(iid)
    return False


def is_trading_active() -> bool:
    return is_desired_trading_active()


def is_running(instrument_id: Optional[int] = None) -> bool:
    with _lock:
        if instrument_id is not None:
            key = _iid_to_primary.get(instrument_id)
            state = _engines.get(key) if key else None
            return bool(state and state.running and state.thread and state.thread.is_alive())
        iid = _active_iid
        if iid is not None:
            key = _iid_to_primary.get(iid)
            state = _engines.get(key) if key else None
            return bool(state and state.running and state.thread and state.thread.is_alive())
        return len(_engines) > 0


def thread_alive(instrument_id: Optional[int] = None) -> bool:
    with _lock:
        iid = instrument_id if instrument_id is not None else _active_iid
        if iid is None:
            return False
        key = _iid_to_primary.get(iid)
        state = _engines.get(key) if key else None
        return bool(state and state.thread and state.thread.is_alive())


def set_hist(
    df: pd.DataFrame,
    instrument_id: Optional[int] = None,
    bot_id: Optional[str] = None,
) -> None:
    """Update historical candles for one bot (or all hubs on an instrument)."""
    with _lock:
        iid = instrument_id if instrument_id is not None else _active_iid
        bid = bot_id if bot_id is not None else _active_bot_id
    if iid is not None and df is not None and not df.empty:
        market_data_hub.set_hist(iid, df, bot_id=bid or None)


# ── Shutdown handlers ─────────────────────────────────────────────────────────

def _shutdown(*_args) -> None:
    log.info("Shutdown signal — stopping all engines and WebSockets")
    stop_all()
    tick_manager.stop_all()


def _register_shutdown_handlers() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
    except ValueError:
        log.debug("Shutdown handlers not registered in this interpreter")


_register_shutdown_handlers()
ensure_supervisor()
