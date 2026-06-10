"""In-memory ring buffer of log records for the visual-bot, exposed via GET /logs
so the dashboard's 'Logs' tab can show both containers in one place.

A bounded handler is attached to the root logger once; records still flow to
stdout, so `docker logs visual-bot` is unaffected.
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
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": msg,
            })


_installed = False
_install_lock = threading.Lock()


def install(level: int = logging.INFO) -> None:
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


def get_records(level: Optional[str] = None, limit: int = 1000) -> list[dict]:
    with _lock:
        records = list(_buffer)
    if level and level != "ALL":
        thr = _LEVEL_ORDER.get(level, 0)
        records = [r for r in records if _LEVEL_ORDER.get(r["level"], 0) >= thr]
    return records[-limit:]
