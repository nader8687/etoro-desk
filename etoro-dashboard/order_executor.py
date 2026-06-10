"""
Paced order executor — kills the candle-close stampede.

At every candle boundary, dozens of bots fire simultaneously and their order
REST calls hit eToro's per-key rate limit in one burst: 429 → sleep-retry →
the herd takes seconds to drain, entries fill late (edge decays), and exits
can queue behind entry traffic.  This module serialises that burst:

  • ENTRIES acquire a pacing slot (FIFO token spacing, ~ORDER_ENTRY_RATE/sec).
    Each waiting bot blocks only ITS OWN thread, so the fleet self-staggers.
    A slot that would exceed MAX_WAIT is declined → the engine skips the entry
    (better no trade than a badly late one).
  • EXITS never wait.  Close paths call note_priority_order(), which pushes the
    next entry slot back instead — exits always have headroom under the limit.
  • IN-FLIGHT CASH ledger: entries reserve their dollar amount while the order
    is in flight, and the position sizer subtracts the in-flight total from
    spendable — closing the race where N bots sized against the same free-cash
    snapshot in the same second could collectively overshoot the cash reserve.

The caller pairs the slot with a RE-QUOTE GUARD (trading_engine): after the
wait, the live quote is compared to the signal-time quote and the entry is
abandoned as stale if price already ran beyond a fraction of its edge.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)

# Entries per second across the whole fleet (token spacing = 1/rate).  eToro's
# per-key limit comfortably absorbs this plus background polling.
ORDER_ENTRY_RATE   = float(os.environ.get("ORDER_ENTRY_RATE", "2.0"))
ENTRY_SPACING_SEC  = 1.0 / max(0.25, ORDER_ENTRY_RATE)
MAX_WAIT_SEC       = float(os.environ.get("ORDER_ENTRY_MAX_WAIT", "8.0"))

_lock = threading.Lock()
_next_slot_at: float = 0.0          # monotonic time the next entry may fire
_inflight: dict[str, float] = {}    # bot_uuid -> reserved $ while order in flight


def acquire_entry_slot(timeout: float = MAX_WAIT_SEC) -> Optional[float]:
    """Block until this thread may place an ENTRY order.

    Returns the seconds waited (0.0 = immediate), or None when the queue is so
    deep the slot would exceed *timeout* — callers must then SKIP the entry.
    FIFO by arrival; each caller blocks only its own bot thread."""
    global _next_slot_at
    with _lock:
        now  = time.monotonic()
        slot = max(now, _next_slot_at)
        wait = slot - now
        if wait > timeout:
            return None
        _next_slot_at = slot + ENTRY_SPACING_SEC
    if wait > 0:
        time.sleep(wait)
    return wait


def note_priority_order() -> None:
    """An exit/close REST call is about to fire.  Exits never wait — instead the
    next ENTRY slot is pushed back one spacing so the combined order stream
    stays under the per-key rate limit."""
    global _next_slot_at
    with _lock:
        _next_slot_at = max(_next_slot_at, time.monotonic()) + ENTRY_SPACING_SEC


# ── In-flight cash ledger ─────────────────────────────────────────────────────

def reserve_cash(bot_uuid: str, amount: float) -> None:
    if not bot_uuid:
        return
    with _lock:
        _inflight[bot_uuid] = float(amount)


def release_cash(bot_uuid: str) -> None:
    if not bot_uuid:
        return
    with _lock:
        _inflight.pop(bot_uuid, None)


def inflight_cash() -> float:
    """Total $ reserved by entries currently in flight — the position sizer
    subtracts this from spendable so concurrent entries can't overshoot the
    cash reserve."""
    with _lock:
        return float(sum(_inflight.values()))
