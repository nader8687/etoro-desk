"""Persist trading runtime settings across container restarts."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PERSIST_PATH = Path(
    os.environ.get("ETORO_RUNTIME_STATE", "/tmp/etoro_runtime.json"),
)

PERSIST_KEYS = (
    "auto_trade_active",        # global auto-trade on/off — persists across restarts
    "live_feed",
    "engine_instrument_id",
    "engine_selected_label",
    "engine_interval_label",
    "engine_interval_seconds",
    "engine_candle_count",
    "demo_trade_amount",
    "display_tz",               # user-selected display timezone (Trading page)
)

# Keys whose change warrants an immediate save (skip debounce).
_CRITICAL_KEYS = {"engine_instrument_id", "live_feed", "auto_trade_active"}

# Debounce: write at most once every N seconds unless a critical key changed.
_SAVE_INTERVAL = 30.0

_last_save_at: float = 0.0
_last_save_data: dict[str, Any] = {}


def load() -> dict[str, Any]:
    try:
        if PERSIST_PATH.exists():
            data = json.loads(PERSIST_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        log.warning("Could not load runtime state: %s", exc)
    return {}


def save(session: dict[str, Any]) -> None:
    global _last_save_at, _last_save_data

    data = {k: session[k] for k in PERSIST_KEYS if k in session}

    now = time.monotonic()
    elapsed = now - _last_save_at

    # Check whether any critical key changed value.
    critical_changed = any(
        data.get(k) != _last_save_data.get(k)
        for k in _CRITICAL_KEYS
    )

    # Skip write if nothing critical changed and interval not elapsed.
    if not critical_changed and elapsed < _SAVE_INTERVAL and _last_save_data:
        return

    try:
        PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PERSIST_PATH.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        _last_save_at = now
        _last_save_data = dict(data)
    except Exception as exc:
        log.warning("Could not save runtime state: %s", exc)
