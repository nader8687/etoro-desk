"""
Shared open-positions cache — single REST poller, many readers.

A dedicated background thread (`start_background_poller`) keeps the cache
fresh.  All engine ticks and UI fragments read the cache via `get_positions()`
without blocking on REST — the poller absorbs that latency once per interval
instead of on every reader's call.

For explicit refreshes (e.g. after a manual close) callers may still invoke
`refresh_if_stale(..., force=True)` to fetch synchronously.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from etoro_client import EToroClient

log = logging.getLogger(__name__)

# Portfolio REST + parallel rates enrichment is expensive (~500 ms – 2 s).
# Background poller fires every POLL_INTERVAL; consumers should treat the
# cache as authoritative.  Positions don't change sub-second.
#
# 4 s is a good balance: a Portfolio fragment refresh every 4 s always sees
# fresh data, and a 25% REST duty cycle is acceptable for typical < 10
# position counts.  Tune POLL_INTERVAL up if eToro rate-limits.
POLL_INTERVAL = 8.0
REFRESH_SEC   = 6.0   # legacy TTL used by refresh_if_stale() for synchronous callers

_lock = threading.Lock()
_positions: list[dict] = []
_fetched_at: float = 0.0
_demo: Optional[bool] = None

# ── Background poller state ───────────────────────────────────────────────────
_poller_thread:   Optional[threading.Thread] = None
_poller_client:   Optional["EToroClient"]    = None
_poller_demo:     bool                       = True
_poller_stop:     threading.Event            = threading.Event()
_poller_interval: float                      = POLL_INTERVAL


def get_positions() -> list[dict]:
    with _lock:
        return list(_positions)


def set_positions(positions: list[dict], demo: bool) -> None:
    """Overwrite the cache with a freshly-fetched eToro position list.

    Used by callers that fetch directly from eToro (e.g. the Portfolio tab) so
    the shared cache stays in sync without a second REST round-trip."""
    global _positions, _fetched_at, _demo
    with _lock:
        _positions = list(positions)
        _fetched_at = time.monotonic()
        _demo = demo


def invalidate() -> None:
    """Mark the cache stale (next refresh refetches) but KEEP last-known data.

    This fires after EVERY trade open/close (constantly with a large fleet).
    Blanking the list here made every reader show 'Loading positions…' until
    the poller refilled it — which under eToro rate-limiting takes seconds —
    i.e. the Portfolio tab 'hang'.  Readers must always get the last-known
    list instantly; freshness is the background poller's job."""
    global _fetched_at
    with _lock:
        _fetched_at = 0.0


def remove_position(position_id: int) -> None:
    """Drop one closed position from the cache without blanking the rest.

    Used after a successful manual close so the Portfolio tab updates instantly
    instead of going empty while a synchronous REST refresh blocks the server."""
    global _positions, _fetched_at
    with _lock:
        _positions = [
            p for p in _positions if int(p.get("position_id") or -1) != int(position_id)
        ]
        _fetched_at = time.monotonic()


def refresh_if_stale(
    client: "EToroClient",
    demo: bool,
    *,
    force: bool = False,
    ttl: float = REFRESH_SEC,
) -> list[dict]:
    """Synchronous refresh — used by code paths that need a fresh read RIGHT NOW
    (manual close, explicit Refresh button).  Normal readers should use
    get_positions() and rely on the background poller."""
    global _positions, _fetched_at, _demo

    now = time.monotonic()
    with _lock:
        if (
            not force
            and _demo == demo
            and now - _fetched_at < ttl
            and _positions
        ):
            return list(_positions)

    try:
        fresh = client.get_open_positions(demo=demo)
    except Exception as exc:
        log.debug("Position refresh failed: %s", exc)
        with _lock:
            return list(_positions)

    with _lock:
        _positions = fresh
        _fetched_at = now
        _demo = demo
    return list(fresh)


# ── Background poller ────────────────────────────────────────────────────────

def start_background_poller(
    client: "EToroClient",
    demo: bool,
    *,
    interval: float = POLL_INTERVAL,
) -> None:
    """Start (or update) the background positions poller.

    Idempotent — safe to call from every engine boot and every UI rerun.
    The poller refreshes the cache every *interval* seconds in its own thread
    so engine ticks and UI fragments can read get_positions() instantly.
    """
    global _poller_thread, _poller_client, _poller_demo, _poller_interval
    _poller_client = client
    _poller_demo = demo
    _poller_interval = max(1.0, interval)

    if _poller_thread is not None and _poller_thread.is_alive():
        return
    _poller_stop.clear()
    _poller_thread = threading.Thread(
        target=_poller_loop, daemon=True, name="positions-poller",
    )
    _poller_thread.start()
    log.info("Positions background poller started (interval %.1fs)", _poller_interval)


def stop_background_poller() -> None:
    _poller_stop.set()


def _poller_loop() -> None:
    # Fetch immediately on start so consumers see fresh data without waiting
    # a full interval.
    while not _poller_stop.is_set():
        client = _poller_client
        demo = _poller_demo
        if client is not None:
            try:
                refresh_if_stale(client, demo, ttl=0)
            except Exception:
                log.exception("Positions poller iteration failed")
        if _poller_stop.wait(_poller_interval):
            return
