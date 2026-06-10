"""
Background chart builder — independent of Streamlit and the trading engine.

Ticks flow:  tick_manager → market_data_hub (candle aggregation) → consumers
              ├─ trading_engine  (signals, stop-loss, entries)
              └─ Streamlit UI    (Plotly render only)

Multi-instrument, multi-timeframe: one HubState per bot_id (string key),
one shared build thread iterates all active hubs every BUILD_INTERVAL seconds.

bot_id is the instruments.toml section key (e.g. "btc", "btc_15m").
When callers omit bot_id the registry falls back to the primary bot for that
instrument_id so all existing iid-based callers continue to work unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

import tick_manager

log = logging.getLogger(__name__)

BUILD_INTERVAL = 2.0
# Bar length at or above this: committed OHLC comes from eToro hist only.  The
# tick buffer (~2 h) cannot represent a full 4h/daily bar; ticks still build the
# forming candle for live chart display.
HTF_HIST_COMMITTED_SEC = 14400


# ── Public data types ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HubConfig:
    instrument_id: int
    interval_label: str
    interval_seconds: int
    candle_count: int
    bot_id: str = ""   # instruments.toml section key; empty = derive from iid


@dataclass(frozen=True)
class ChartSnapshot:
    instrument_id: int
    interval_label: str
    interval_seconds: int
    committed: pd.DataFrame
    forming: pd.DataFrame
    chart_data: pd.DataFrame
    latest_ask: float
    latest_bid: float
    tick_count: int
    last_tick_time: Optional[datetime]
    last_committed_time: Optional[pd.Timestamp]
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


# ── Per-instrument hub state ──────────────────────────────────────────────────

@dataclass
class _HubState:
    config: HubConfig
    hist_df: pd.DataFrame
    snapshot: Optional[ChartSnapshot] = None
    running: bool = False
    # Last tick-sequence value we built a snapshot for.  Using the monotonic
    # tick sequence (not buffer length) means saturating the bounded tick buffer
    # never makes us falsely skip a rebuild and freeze the live chart.
    last_tick_seq: int = -1


# ── Module-level registry ─────────────────────────────────────────────────────

_lock = threading.Lock()
_hubs: dict[str, _HubState] = {}          # bot_id → _HubState
_iid_to_primary: dict[int, str] = {}      # instrument_id → primary bot_id (backward compat)
_active_key: Optional[str] = None         # bot_id of most-recently configured hub
_desired_active: bool = False

_build_thread: Optional[threading.Thread] = None
_build_running: bool = False

_supervisor_thread: Optional[threading.Thread] = None
_supervisor_started: bool = False


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hub_key(config: HubConfig) -> str:
    """Derive the registry key for a HubConfig."""
    return config.bot_id if config.bot_id else f"{config.instrument_id}_{config.interval_seconds}"


def _build_candles(
    instrument_id: int,
    interval_seconds: int,
    hist_df: pd.DataFrame,
    candle_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.Timestamp]]:
    live_df = tick_manager.ticks_to_candles(instrument_id, interval_seconds)

    # HTF: eToro hist is authoritative for closed bars; ticks only for forming.
    if interval_seconds >= HTF_HIST_COMMITTED_SEC and not hist_df.empty:
        committed_all = hist_df.tail(int(candle_count)).copy()
        if "time" in committed_all.columns:
            committed_all["time"] = pd.to_datetime(committed_all["time"], utc=True)
        live_forming = (
            live_df[live_df["time"] == live_df["time"].max()].copy()
            if not live_df.empty else pd.DataFrame()
        )
        chart_data = (
            pd.concat([committed_all, live_forming], ignore_index=True)
            if not live_forming.empty
            else committed_all
        )
        last_committed = (
            committed_all["time"].iloc[-1] if not committed_all.empty else None
        )
        return committed_all, live_forming, chart_data, last_committed

    if not live_df.empty:
        forming_time    = live_df["time"].max()
        live_committed  = live_df[live_df["time"] < forming_time]
        live_forming    = live_df[live_df["time"] == forming_time]

        if not hist_df.empty:
            cutoff = live_df["time"].min()
            base   = hist_df[hist_df["time"] < cutoff].copy()
            base["time"] = base["time"].dt.tz_convert("UTC")
            committed_all = pd.concat([base, live_committed], ignore_index=True)
        else:
            committed_all = live_committed

        committed_all = committed_all.sort_values("time").tail(int(candle_count))
    else:
        committed_all = (
            hist_df.tail(int(candle_count)) if not hist_df.empty else pd.DataFrame()
        )
        live_forming  = pd.DataFrame()

    chart_data = (
        pd.concat([committed_all, live_forming], ignore_index=True)
        if not live_forming.empty
        else committed_all
    )
    last_committed = (
        committed_all["time"].iloc[-1] if not committed_all.empty else None
    )
    return committed_all, live_forming, chart_data, last_committed


def _build_group(
    iid: int,
    interval_seconds: int,
    members: list[tuple[str, _HubState]],
) -> None:
    """Build ONE chart snapshot for an (instrument, interval) stream and share it
    across every bot subscribed to it.

    Many bots differ only by strategy, so on the same asset+timeframe they consume
    identical candles.  Building once per stream — instead of once per bot — removes
    the redundant tick→candle assembly that previously scaled with bot count (e.g.
    15 BTC bots rebuilding the same 1-minute candles on every tick)."""
    cur_seq = tick_manager.get_tick_seq(iid)

    # Change-detect for the whole group (cheap, O(1)).  If every member already
    # holds a snapshot at the current tick sequence, nothing changed → skip.
    # The monotonic tick sequence (not buffer length) means rebuilds never freeze
    # once the bounded tick buffer saturates.
    if all(s.last_tick_seq == cur_seq and s.snapshot is not None for _, s in members):
        return

    buf            = tick_manager.get_buffer(iid)
    latest         = buf[-1] if buf else None     # O(1) — no full-deque copy
    tick_count     = len(buf)
    last_tick_time = tick_manager.get_last_tick_time(iid)

    # Members share an interval ⇒ identical committed candles.  Build with the
    # largest candle_count any member wants (a shorter-window strategy just
    # ignores the extra history) and the first non-empty hist (they match within
    # a stream — every bot preloads the same instrument/interval candles).
    candle_count   = max(s.config.candle_count for _, s in members)
    interval_label = members[0][1].config.interval_label
    hist_ref       = next((s.hist_df for _, s in members if not s.hist_df.empty), pd.DataFrame())
    hist_df        = hist_ref.copy() if not hist_ref.empty else pd.DataFrame()

    committed, forming, chart_data, last_committed = _build_candles(
        iid, interval_seconds, hist_df, candle_count,
    )

    snap = ChartSnapshot(
        instrument_id=iid,
        interval_label=interval_label,
        interval_seconds=interval_seconds,
        committed=committed,
        forming=forming,
        chart_data=chart_data,
        latest_ask=latest["ask"] if latest else 0.0,
        latest_bid=latest["bid"] if latest else 0.0,
        tick_count=tick_count,
        last_tick_time=last_tick_time,
        last_committed_time=last_committed,
    )
    # ChartSnapshot is frozen/immutable, so the SAME object can be shared safely
    # across every member of the stream.
    with _lock:
        for key, _ in members:
            st = _hubs.get(key)
            if st is not None:
                st.snapshot = snap
                st.last_tick_seq = cur_seq


# ── Build loop (shared across all instruments) ────────────────────────────────

def _build_loop() -> None:
    global _build_running
    while _build_running:
        with _lock:
            active = [(key, s) for key, s in _hubs.items() if s.running]
        # Collapse bots onto the market-data stream they share: one build per
        # (instrument, interval) instead of one per bot.
        groups: dict[tuple[int, int], list[tuple[str, _HubState]]] = {}
        for key, s in active:
            gk = (s.config.instrument_id, s.config.interval_seconds)
            groups.setdefault(gk, []).append((key, s))
        for (iid, interval_seconds), members in groups.items():
            try:
                _build_group(iid, interval_seconds, members)
            except Exception:
                log.exception(
                    "Hub build failed for stream %s/%ss", iid, interval_seconds,
                )
        time.sleep(BUILD_INTERVAL)


def _start_build_thread() -> None:
    global _build_thread, _build_running
    if _build_thread is not None and _build_thread.is_alive():
        return
    _build_running = True
    _build_thread = threading.Thread(
        target=_build_loop, daemon=True, name="market-data-hub-build",
    )
    _build_thread.start()
    log.info("Market data hub build thread started")


# ── Public API ────────────────────────────────────────────────────────────────

def configure(config: HubConfig, hist_df: Optional[pd.DataFrame] = None) -> None:
    """Create or update the hub for config.instrument_id / config.bot_id.

    When bot_id is empty, the call is treated as a Trading-tab update and is
    routed to the primary hub already registered for that instrument_id (if
    any), so interval changes are applied in-place rather than creating a
    duplicate hub.
    """
    global _active_key
    iid = config.instrument_id

    with _lock:
        # Resolve bot_id: if empty, check whether a primary hub already exists
        # for this iid and re-use its key (supports interval changes from UI).
        if not config.bot_id and iid in _iid_to_primary:
            primary = _iid_to_primary[iid]
            if primary in _hubs:
                config = replace(config, bot_id=primary)

        key = _hub_key(config)

        if key in _hubs:
            prev = _hubs[key]
            if prev.config.interval_seconds != config.interval_seconds:
                tick_manager.clear_candle_cache(iid)
                prev.last_tick_seq = -1
            prev.config = config
            prev.running = True
            if hist_df is not None and not hist_df.empty:
                prev.hist_df = hist_df.copy()
        else:
            _hubs[key] = _HubState(
                config=config,
                hist_df=hist_df.copy() if hist_df is not None and not hist_df.empty else pd.DataFrame(),
                running=True,
            )
            # Register as primary for this iid only if none exists yet
            if iid not in _iid_to_primary:
                _iid_to_primary[iid] = key

        _active_key = key
    if _desired_active:
        _start_build_thread()


def set_hist(instrument_id: int, df: pd.DataFrame, bot_id: Optional[str] = None) -> None:
    """Replace historical candles.

    If bot_id is given, update only that specific hub.
    Otherwise update all hubs registered for instrument_id (backward compat).
    """
    with _lock:
        if bot_id is not None:
            if bot_id in _hubs:
                _hubs[bot_id].hist_df = df.copy() if df is not None and not df.empty else pd.DataFrame()
                _hubs[bot_id].last_tick_seq = -1
        else:
            for state in _hubs.values():
                if state.config.instrument_id == instrument_id:
                    state.hist_df = df.copy() if df is not None and not df.empty else pd.DataFrame()
                    state.last_tick_seq = -1


def refresh_stream_hist(
    instrument_id: int,
    interval_seconds: int,
    hist_df: pd.DataFrame,
) -> Optional[ChartSnapshot]:
    """Replace history for every hub on (instrument, interval) and rebuild immediately.

    Used at HTF candle close so committed OHLC comes from eToro's API (full bar)
    instead of the tick buffer, which cannot cover a whole 4h/daily period.
    """
    if hist_df is None or hist_df.empty:
        return None
    with _lock:
        members = [
            (key, s)
            for key, s in _hubs.items()
            if s.config.instrument_id == instrument_id
            and s.config.interval_seconds == interval_seconds
        ]
        if not members:
            return None
        hist_copy = hist_df.copy()
        for key, _ in members:
            st = _hubs.get(key)
            if st is not None:
                st.hist_df = hist_copy.copy()
                st.last_tick_seq = -1
    try:
        _build_group(instrument_id, interval_seconds, members)
    except Exception:
        log.exception(
            "refresh_stream_hist build failed for %s/%ss", instrument_id, interval_seconds,
        )
        return None
    with _lock:
        for key, _ in members:
            st = _hubs.get(key)
            if st is not None and st.snapshot is not None:
                return st.snapshot
    return None


def get_snapshot(
    instrument_id: Optional[int] = None,
    bot_id: Optional[str] = None,
) -> Optional[ChartSnapshot]:
    """Return latest snapshot by bot_id, instrument_id, or the active hub."""
    with _lock:
        if bot_id is not None:
            state = _hubs.get(bot_id)
            return state.snapshot if state else None
        if instrument_id is not None:
            primary = _iid_to_primary.get(instrument_id)
            if primary:
                state = _hubs.get(primary)
                return state.snapshot if state else None
            return None
        if _active_key is not None:
            state = _hubs.get(_active_key)
            return state.snapshot if state else None
        return None


def get_all_snapshots() -> dict[str, ChartSnapshot]:
    """Return snapshots for all running bots (bot_id → ChartSnapshot)."""
    with _lock:
        return {
            key: s.snapshot
            for key, s in _hubs.items()
            if s.running and s.snapshot is not None
        }


def get_hist_df(
    *,
    bot_id: Optional[str] = None,
    instrument_id: Optional[int] = None,
    interval_seconds: Optional[int] = None,
) -> pd.DataFrame:
    """Return raw preloaded historical candles from the hub (no REST).

    Only returns ``hist_df`` — never ``snapshot.committed``, which is already
    merged with live ticks and must not be fed back in as base history."""
    with _lock:
        state: Optional[_HubState] = None
        if bot_id is not None:
            state = _hubs.get(bot_id)
        elif instrument_id is not None:
            primary = _iid_to_primary.get(instrument_id)
            if primary:
                state = _hubs.get(primary)
        elif _active_key:
            state = _hubs.get(_active_key)
        if state is None or state.hist_df.empty:
            return pd.DataFrame()
        if (
            interval_seconds is not None
            and state.config.interval_seconds != interval_seconds
        ):
            return pd.DataFrame()
        return state.hist_df.copy()


def get_config(
    instrument_id: Optional[int] = None,
    bot_id: Optional[str] = None,
) -> Optional[HubConfig]:
    with _lock:
        if bot_id is not None:
            state = _hubs.get(bot_id)
            return state.config if state else None
        if instrument_id is not None:
            primary = _iid_to_primary.get(instrument_id)
            if primary:
                state = _hubs.get(primary)
                return state.config if state else None
        if _active_key:
            state = _hubs.get(_active_key)
            return state.config if state else None
        return None


def set_desired_active(enabled: bool) -> None:
    """Global live-feed toggle — applies to all hubs."""
    global _desired_active
    _desired_active = enabled
    if not enabled:
        stop()
    ensure_supervisor()


def is_desired_active() -> bool:
    return _desired_active


def start(instrument_id: Optional[int] = None) -> None:
    """Mark hub(s) as running and start build thread."""
    with _lock:
        if instrument_id is not None:
            for state in _hubs.values():
                if state.config.instrument_id == instrument_id:
                    state.running = True
        else:
            for s in _hubs.values():
                s.running = True
    _start_build_thread()


def stop(instrument_id: Optional[int] = None, bot_id: Optional[str] = None) -> None:
    """Stop one hub or all hubs."""
    global _build_running
    with _lock:
        if bot_id is not None:
            if bot_id in _hubs:
                _hubs[bot_id].running = False
                _hubs[bot_id].snapshot = None
        elif instrument_id is not None:
            for state in _hubs.values():
                if state.config.instrument_id == instrument_id:
                    state.running = False
                    state.snapshot = None
        else:
            for s in _hubs.values():
                s.running = False
                s.snapshot = None
            _build_running = False
    log.info("Market data hub stopped (instrument=%s, bot=%s)", instrument_id or "all", bot_id)


def stop_all() -> None:
    stop()


def remove(instrument_id: int, bot_id: Optional[str] = None) -> None:
    """Remove hub(s) from the registry."""
    with _lock:
        if bot_id is not None:
            _hubs.pop(bot_id, None)
        else:
            keys = [k for k, s in _hubs.items() if s.config.instrument_id == instrument_id]
            for k in keys:
                _hubs.pop(k, None)
            _iid_to_primary.pop(instrument_id, None)


def is_running(instrument_id: Optional[int] = None) -> bool:
    with _lock:
        if instrument_id is not None:
            return any(
                s.running for s in _hubs.values()
                if s.config.instrument_id == instrument_id
            )
        return any(s.running for s in _hubs.values())


def thread_alive() -> bool:
    return _build_thread is not None and _build_thread.is_alive()


# ── Supervisor ────────────────────────────────────────────────────────────────

def _supervisor_loop() -> None:
    while True:
        try:
            if _desired_active:
                if not thread_alive():
                    _start_build_thread()
            elif is_running():
                stop()
        except Exception:
            log.exception("Market data hub supervisor failed")
        time.sleep(10)


def ensure_supervisor() -> None:
    global _supervisor_thread, _supervisor_started
    if _supervisor_started and _supervisor_thread and _supervisor_thread.is_alive():
        return
    _supervisor_started = True
    _supervisor_thread = threading.Thread(
        target=_supervisor_loop, daemon=True, name="hub-supervisor",
    )
    _supervisor_thread.start()
    log.info("Market data hub supervisor started")


ensure_supervisor()
