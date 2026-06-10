"""Thread-safe notification queue from background engine → Streamlit UI."""
from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Literal

Kind = Literal["trade_open", "trade_close", "trade_error", "info"]

# queue.Queue is already thread-safe — no extra lock needed around put/get.
_queue: queue.Queue = queue.Queue(maxsize=200)


@dataclass(frozen=True)
class EngineNotification:
    kind: Kind
    message: str
    instrument_id: int | None = None


def push(
    kind: Kind,
    message: str,
    *,
    instrument_id: int | None = None,
) -> None:
    try:
        _queue.put_nowait(EngineNotification(kind, message, instrument_id))
    except queue.Full:
        pass


def drain(*, max_items: int = 20) -> list[EngineNotification]:
    out: list[EngineNotification] = []
    while len(out) < max_items:
        try:
            out.append(_queue.get_nowait())
        except queue.Empty:
            break
    return out
