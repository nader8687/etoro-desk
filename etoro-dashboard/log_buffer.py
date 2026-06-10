"""In-memory ring buffer of log records, surfaced in the dashboard 'Logs' tab.

A bounded handler is attached to the root logger ONCE, so every module's logging
(trading_engine, trade_manager, signal_worker, tick_manager, …, which all
propagate to root) is captured live — no docker socket or log-file mounts needed.
The records also keep flowing to stdout, so `docker logs` is unaffected.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

_MAX_RECORDS = 4000
_buffer: deque = deque(maxlen=_MAX_RECORDS)
_lock = threading.Lock()

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class _RingBufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        with _lock:
            _buffer.append({
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc),
                "level": record.levelname,
                "logger": record.name,
                "msg": msg,
            })


_installed = False
_install_lock = threading.Lock()


def install(level: int = logging.INFO) -> None:
    """Attach the ring-buffer handler to the root logger (idempotent)."""
    global _installed
    with _install_lock:
        if _installed:
            return
        handler = _RingBufferHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > level:
            root.setLevel(level)
        _installed = True


def get_records(
    level: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 1000,
) -> list[dict]:
    """Newest-last records, optionally filtered by min level and a text query."""
    with _lock:
        records = list(_buffer)
    if level and level != "ALL":
        thr = _LEVEL_ORDER.get(level, 0)
        records = [r for r in records if _LEVEL_ORDER.get(r["level"], 0) >= thr]
    if query:
        q = query.lower()
        records = [
            r for r in records
            if q in r["msg"].lower() or q in r["logger"].lower()
        ]
    return records[-limit:]


def format_lines(records: list[dict]) -> str:
    """Render records as plain log lines: 'HH:MM:SS LEVEL logger: message'."""
    return "\n".join(
        f"{r['ts'].astimezone().strftime('%H:%M:%S')} "
        f"{r['level']:<7} {r['logger']}: {r['msg']}"
        for r in records
    )


def clear() -> None:
    with _lock:
        _buffer.clear()


def count() -> int:
    with _lock:
        return len(_buffer)
