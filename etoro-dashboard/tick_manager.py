"""
Robust WebSocket tick manager — auto-reconnects, exposes connection state.
Module-level stores survive Streamlit script reruns.
"""
import itertools
import json
import logging
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd
import websocket

log = logging.getLogger(__name__)

WS_URL    = "wss://ws.etoro.com/ws"
MAX_TICKS = 7200          # rolling 2-hour tick buffer per instrument
STALE_SEC = 45            # seconds without a tick before health-watchdog forces reconnect
LIVE_SEC  = 30            # UI badge: green if last tick within this many seconds
FIRST_TICK_SEC = 25       # reconnect if connected but no tick received in this window
MAX_BACKOFF = 30          # max reconnect delay (seconds)


class State(str, Enum):
    IDLE         = "idle"
    CONNECTING   = "connecting"
    CONNECTED    = "connected"
    RECONNECTING = "reconnecting"
    STOPPED      = "stopped"


# ── Module-level stores (survive Streamlit reruns) ───────────────────────────
_buffers:    dict[int, deque]                    = {}
_states:     dict[int, State]                    = {}
_last_tick:  dict[int, Optional[datetime]]       = {}
_stop_flags: dict[int, bool]                     = {}
_ws_conns:   dict[int, websocket.WebSocketApp]   = {}
_threads:    dict[int, threading.Thread]         = {}
_watchdogs:  dict[int, threading.Thread]         = {}
_creds:      dict[int, tuple[str, str]]          = {}   # instrument_id -> (api_key, user_key)
_candle_cache: dict[tuple[int, int], dict]       = {}   # (iid, interval_sec) -> snapshot
# Monotonic per-instrument tick counter — increments on EVERY tick and never
# resets while the process lives.  Consumers use it as the change-detector
# instead of len(buffer): the buffer is a bounded deque whose length plateaus at
# MAX_TICKS once full, which would otherwise make "len unchanged" falsely report
# "no new ticks" and freeze candle/chart updates after ~2 hours of streaming.
_tick_seq:   dict[int, int]                       = {}
_lock = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def get_buffer(instrument_id: int) -> deque:
    with _lock:
        if instrument_id not in _buffers:
            _buffers[instrument_id] = deque(maxlen=MAX_TICKS)
        return _buffers[instrument_id]


def get_state(instrument_id: int) -> State:
    with _lock:
        return _states.get(instrument_id, State.IDLE)


def get_last_tick_time(instrument_id: int) -> Optional[datetime]:
    with _lock:
        return _last_tick.get(instrument_id)


def get_tick_seq(instrument_id: int) -> int:
    """Monotonic count of ticks ever appended for this instrument.

    Use this — not len(get_buffer(...)) — to detect new ticks: the buffer is a
    bounded deque whose length stops growing once full, but this counter keeps
    incrementing so consumers never falsely conclude the feed is idle."""
    with _lock:
        return _tick_seq.get(instrument_id, 0)


def get_latest_quote(instrument_id: int) -> tuple[float, float] | None:
    """Latest (ask, bid) from the tick buffer, or None if empty."""
    buf = get_buffer(instrument_id)
    if not buf:
        return None
    t = buf[-1]
    return float(t["ask"]), float(t["bid"])


def get_tick_price_change(instrument_id: int) -> tuple[float | None, float | None]:
    """Price change between the last two buffered ticks (abs, %)."""
    ticks = list(get_buffer(instrument_id))
    if len(ticks) < 2:
        return None, None
    prev = float(ticks[-2]["last"])
    curr = float(ticks[-1]["last"])
    if not prev:
        return None, None
    ch = curr - prev
    return ch, (ch / prev) * 100


def is_running(instrument_id: int) -> bool:
    with _lock:
        t = _threads.get(instrument_id)
        return t is not None and t.is_alive()


def clear_buffer(instrument_id: int) -> None:
    buf = get_buffer(instrument_id)
    buf.clear()
    with _lock:
        for key in list(_candle_cache):
            if key[0] == instrument_id:
                _candle_cache.pop(key, None)


def clear_candle_cache(instrument_id: int | None = None) -> None:
    with _lock:
        if instrument_id is None:
            _candle_cache.clear()
        else:
            for key in list(_candle_cache):
                if key[0] == instrument_id:
                    _candle_cache.pop(key, None)


def start(instrument_id: int, api_key: str, user_key: str) -> None:
    """Start streaming for instrument_id. No-op if already running.

    Thread-safe: the run/start decision is made under _lock so two concurrent
    callers can't race past the is_running check and spawn duplicate WebSocket
    threads for the same instrument.
    """
    with _lock:
        existing = _threads.get(instrument_id)
        if existing is not None and existing.is_alive():
            # Refresh credentials in case they changed; otherwise no-op.
            _creds[instrument_id] = (api_key, user_key)
            return
        _stop_flags[instrument_id] = False
        _creds[instrument_id] = (api_key, user_key)
        _states[instrument_id] = State.CONNECTING
        t = threading.Thread(
            target=_run_loop, args=(instrument_id,),
            daemon=True, name=f"ws-{instrument_id}",
        )
        _threads[instrument_id] = t
        existing_wd = _watchdogs.get(instrument_id)
        if existing_wd is None or not existing_wd.is_alive():
            w = threading.Thread(
                target=_watchdog, args=(instrument_id,),
                daemon=True, name=f"wd-{instrument_id}",
            )
            _watchdogs[instrument_id] = w
        else:
            w = None

    # Instant quote via REST while the WebSocket handshake runs
    threading.Thread(
        target=_seed_rest_quote,
        args=(instrument_id, api_key, user_key),
        daemon=True,
        name=f"seed-{instrument_id}",
    ).start()
    t.start()
    if w is not None:
        w.start()


def stop(instrument_id: int) -> None:
    """Gracefully stop streaming."""
    with _lock:
        _stop_flags[instrument_id] = True
        _states[instrument_id] = State.STOPPED
        ws = _ws_conns.pop(instrument_id, None)
    if ws:
        try:
            ws.close()
        except Exception:
            pass


def stop_all() -> None:
    for iid in list(_threads.keys()):
        stop(iid)


_EMPTY_CANDLES = pd.DataFrame(columns=["time", "Open", "High", "Low", "Close"])


def _full_candle_build(ticks: list[dict], interval_seconds: int) -> pd.DataFrame:
    df = pd.DataFrame(ticks)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["bucket"] = df["time"].dt.floor(f"{interval_seconds}s")
    return (
        df.groupby("bucket", sort=True)
        .agg(
            Open=("last", "first"),
            High=("last", "max"),
            Low=("last", "min"),
            Close=("last", "last"),
        )
        .reset_index()
        .rename(columns={"bucket": "time"})
    )


def ticks_to_candles(instrument_id: int, interval_seconds: int) -> pd.DataFrame:
    """Aggregate buffered ticks into OHLC candles (incremental when possible).

    Change-detection uses the monotonic tick sequence (_tick_seq), NOT len(buf),
    so updates never stall once the bounded tick buffer saturates.  The
    incremental path folds EVERY tick that arrived since the last build (not just
    the newest), so intra-interval High/Low extremes are never dropped.
    """
    buf = get_buffer(instrument_id)
    if not buf:
        return _EMPTY_CANDLES.copy()

    cur_seq   = get_tick_seq(instrument_id)
    cache_key = (instrument_id, interval_seconds)
    interval  = f"{interval_seconds}s"

    with _lock:
        cached = _candle_cache.get(cache_key)

    # Up to date — nothing new since last build.
    if cached is not None and cached.get("seq") == cur_seq and not cached["df"].empty:
        return cached["df"].copy()

    # Incremental — fold only the ticks added since the cached sequence.
    if (
        cached is not None
        and cached.get("seq") is not None
        and cur_seq > cached["seq"]
        and not cached["df"].empty
    ):
        new_count = min(cur_seq - cached["seq"], len(buf))
        # Read only the last new_count ticks WITHOUT copying the whole deque
        # (reversed() walks from the right in O(new_count); list(buf) would be
        # O(buffer)=O(7200) every build and starve the WS threads of the GIL).
        if new_count > 0:
            new_ticks = list(itertools.islice(reversed(buf), new_count))
            new_ticks.reverse()
        else:
            new_ticks = []
        df = cached["df"].copy()
        for t in new_ticks:
            price  = float(t["last"])
            bucket = pd.Timestamp(t["time"]).tz_convert("UTC").floor(interval)
            last_bucket = df["time"].iloc[-1]
            if bucket == last_bucket:
                idx = df.index[-1]
                if price > float(df.loc[idx, "High"]):
                    df.loc[idx, "High"] = price
                if price < float(df.loc[idx, "Low"]):
                    df.loc[idx, "Low"] = price
                df.loc[idx, "Close"] = price
            elif bucket > last_bucket:
                df = pd.concat(
                    [df, pd.DataFrame([{
                        "time": bucket, "Open": price,
                        "High": price, "Low": price, "Close": price,
                    }])],
                    ignore_index=True,
                )
            # bucket < last_bucket → late/out-of-order tick; skip.
        with _lock:
            _candle_cache[cache_key] = {"df": df, "seq": cur_seq, "buf_len": len(buf)}
        return df.copy()

    # Full rebuild — first call, cache miss, or sequence reset.
    ticks = list(buf)
    candles = _full_candle_build(ticks, interval_seconds)
    with _lock:
        _candle_cache[cache_key] = {"df": candles, "seq": cur_seq, "buf_len": len(buf)}
    return candles.copy()


# ── Internal: reconnect loop ──────────────────────────────────────────────────

def _run_loop(instrument_id: int) -> None:
    backoff = 1
    while not _stop_flags.get(instrument_id, True):
        api_key, user_key = _creds[instrument_id]
        _set_state(instrument_id, State.CONNECTING)

        connected_at = time.monotonic()
        try:
            _connect(instrument_id, api_key, user_key)  # blocks until closed
        except Exception as exc:
            log.warning("WS error for %s: %s", instrument_id, exc)

        if _stop_flags.get(instrument_id, True):
            break

        # Reset the backoff after a session that stayed up a while: it was a
        # genuinely healthy connection, so a later transient drop should
        # reconnect FAST (1 s).  The old loop only ever doubled backoff and never
        # reset it, so after a few flaps every reconnect waited the full
        # MAX_BACKOFF (~30 s) — turning brief drops into repeated 30 s stale gaps.
        if time.monotonic() - connected_at >= 30:
            backoff = 1

        _set_state(instrument_id, State.RECONNECTING)
        log.info("Reconnecting instrument %s in %ss", instrument_id, backoff)
        for _ in range(int(backoff * 10)):          # sleep in 100ms slices so stop() is responsive
            if _stop_flags.get(instrument_id, True):
                return
            time.sleep(0.1)
        backoff = min(backoff * 2, MAX_BACKOFF)

    _set_state(instrument_id, State.STOPPED)


def _append_tick(buf: deque, instrument_id: int, content: dict) -> bool:
    """Parse eToro tick payload; return True if a quote was stored."""
    ask = content.get("Ask") if content.get("Ask") is not None else content.get("ask")
    bid = content.get("Bid") if content.get("Bid") is not None else content.get("bid")
    if ask is None or bid is None:
        return False

    last = content.get("LastExecution") or content.get("lastExecution")
    date = content.get("Date") or content.get("date")
    market_ts = datetime.now(tz=timezone.utc)
    if date:
        try:
            market_ts = datetime.fromisoformat(str(date).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            pass

    buf.append({
        "time": market_ts,
        "ask":  float(ask),
        "bid":  float(bid),
        "last": float(last) if last else (float(ask) + float(bid)) / 2,
    })
    # Wall-clock time for staleness checks (market Date can be hours old on snapshots).
    with _lock:
        _last_tick[instrument_id] = datetime.now(tz=timezone.utc)
        _tick_seq[instrument_id] = _tick_seq.get(instrument_id, 0) + 1
    return True


def _connect(instrument_id: int, api_key: str, user_key: str) -> None:
    buf = get_buffer(instrument_id)
    authenticated = threading.Event()

    def on_open(ws):
        ws.send(json.dumps({
            "id": str(uuid.uuid4()),
            "operation": "Authenticate",
            "data": {"userKey": user_key, "apiKey": api_key},
        }))

    def on_message(ws, msg):
        if not isinstance(msg, str) or not msg.strip():
            return
        try:
            outer = json.loads(msg)
        except Exception:
            return

        op = outer.get("operation", "")
        if op == "Authenticate":
            if outer.get("success"):
                authenticated.set()
                ws.send(json.dumps({
                    "id": str(uuid.uuid4()),
                    "operation": "Subscribe",
                    "data": {"topics": [f"instrument:{instrument_id}"], "snapshot": True},
                }))
                _set_state(instrument_id, State.CONNECTED)
            else:
                log.error("WS auth failed for instrument %s", instrument_id)
                ws.close()
            return

        if op == "Subscribe":
            return  # subscribe ack, no data needed

        for message in outer.get("messages", []):
            raw = message.get("content", "")
            try:
                content = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if isinstance(content, dict):
                _append_tick(buf, instrument_id, content)

    def on_error(ws, err):
        log.warning("WS error instrument %s: %s", instrument_id, err)

    def on_close(ws, code, msg):
        with _lock:
            _ws_conns.pop(instrument_id, None)
        if get_state(instrument_id) not in (State.STOPPED, State.RECONNECTING):
            _set_state(instrument_id, State.RECONNECTING)

    app = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    with _lock:
        _ws_conns[instrument_id] = app
    app.run_forever(ping_interval=20, ping_timeout=10)


def _seed_rest_quote(instrument_id: int, api_key: str, user_key: str) -> None:
    """Bootstrap bid/ask from REST so the UI is not empty during WS connect."""
    try:
        import requests as _req

        resp = _req.get(
            "https://public-api.etoro.com/api/v1/market-data/instruments/rates",
            headers={
                "x-api-key": api_key,
                "x-user-key": user_key,
                "x-request-id": str(uuid.uuid4()),
                "Accept": "application/json",
            },
            params=[("instrumentIds", instrument_id)],
            timeout=8,
        )
        resp.raise_for_status()
        for row in resp.json().get("rates", []):
            rid = row.get("instrumentID")
            if rid is None or int(rid) != instrument_id:
                continue
            ask, bid = row.get("ask"), row.get("bid")
            if not ask or not bid:
                continue
            now = datetime.now(tz=timezone.utc)
            buf = get_buffer(instrument_id)
            buf.append({
                "time": now,
                "ask":  float(ask),
                "bid":  float(bid),
                "last": float(row.get("lastExecution") or (float(ask) + float(bid)) / 2),
            })
            with _lock:
                _last_tick[instrument_id] = now
                _tick_seq[instrument_id] = _tick_seq.get(instrument_id, 0) + 1
            log.info("REST seed quote for instrument %s", instrument_id)
            return
    except Exception as exc:
        log.debug("REST seed failed for %s: %s", instrument_id, exc)


def _watchdog(instrument_id: int) -> None:
    """Force-reconnect if no tick received in STALE_SEC while supposedly connected."""
    connected_since: Optional[float] = None
    while not _stop_flags.get(instrument_id, True):
        time.sleep(5)
        if _stop_flags.get(instrument_id, True):
            break
        state = get_state(instrument_id)
        if state != State.CONNECTED:
            connected_since = None
            continue
        if connected_since is None:
            connected_since = time.monotonic()
        last = get_last_tick_time(instrument_id)
        now = datetime.now(tz=timezone.utc)
        if last is None:
            if time.monotonic() - connected_since > FIRST_TICK_SEC:
                log.warning(
                    "Instrument %s connected but no tick in %ss — forcing reconnect",
                    instrument_id, FIRST_TICK_SEC,
                )
                _force_reconnect(instrument_id)
                connected_since = None
            continue
        stale_for = (now - last).total_seconds()
        if stale_for > STALE_SEC:
            log.warning("Instrument %s stale for %.0fs — forcing reconnect", instrument_id, stale_for)
            _force_reconnect(instrument_id)
            connected_since = None


def _force_reconnect(instrument_id: int) -> None:
    with _lock:
        ws = _ws_conns.pop(instrument_id, None)
    if ws:
        try:
            ws.close()
        except Exception:
            pass


def _set_state(instrument_id: int, state: State) -> None:
    with _lock:
        _states[instrument_id] = state
