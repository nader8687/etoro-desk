"""
Persistent bot UUID registry.

Each bot_key (instruments.toml section name, e.g. "xrp", "xrp_15m") is
assigned a random UUID v4 on first use.  The mapping is persisted in
/app/data/bot_ids.json so UUIDs survive container restarts.

Use `get_or_create(bot_key)` to get the stable UUID for a bot.
Use `label(bot_key)` to get the short display form (first 8 hex chars).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

_REGISTRY_PATH = Path(os.environ.get("BOT_REGISTRY_PATH", "/app/data/bot_ids.json"))
_lock: threading.Lock = threading.Lock()
_cache: dict[str, str] = {}


def _load() -> None:
    if not _REGISTRY_PATH.exists():
        return
    try:
        text = _REGISTRY_PATH.read_text(encoding="utf-8").strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Salvage a file corrupted by a previous non-atomic write: recover the
            # first valid JSON object.  Losing these UUIDs would regenerate NEW
            # ids for every bot and orphan every owner-map entry, so never wipe.
            data, _end = json.JSONDecoder().raw_decode(text)
            log.warning("Bot registry recovered from a corrupted file")
        if isinstance(data, dict):
            _cache.update({str(k): str(v) for k, v in data.items()})
            _save()   # rewrite cleanly
        log.debug("Bot registry loaded: %d entries", len(_cache))
    except Exception as exc:
        log.warning("Bot registry load failed: %s", exc)


def _save() -> None:
    """Atomic write (temp + rename) so a crash or concurrent writer can never
    leave a half-written / corrupted registry file."""
    try:
        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _REGISTRY_PATH.with_name(_REGISTRY_PATH.name + ".tmp")
        tmp.write_text(json.dumps(_cache, indent=2), encoding="utf-8")
        os.replace(tmp, _REGISTRY_PATH)
    except Exception as exc:
        log.warning("Bot registry save failed: %s", exc)


# Load on module import
_load()


def get_or_create(bot_key: str) -> str:
    """Return the stable UUID for *bot_key*, creating one if needed."""
    with _lock:
        if bot_key not in _cache:
            _cache[bot_key] = str(uuid.uuid4())
            _save()
            log.info("New bot UUID for %r: %s", bot_key, _cache[bot_key])
        return _cache[bot_key]


def get(bot_key: str) -> str | None:
    """Return the UUID for *bot_key* if it exists, else None."""
    with _lock:
        return _cache.get(bot_key)


def label(bot_uuid: str) -> str:
    """Short 8-char prefix suitable for display, e.g. 'a1b2c3d4'."""
    return bot_uuid[:8] if bot_uuid else ""


def get_all() -> dict[str, str]:
    """Return a snapshot of {bot_key: uuid} for all registered bots."""
    with _lock:
        return dict(_cache)


def remove(bot_key: str) -> None:
    """Remove a bot's UUID from the registry (call when a bot is deleted)."""
    with _lock:
        if bot_key in _cache:
            del _cache[bot_key]
            _save()


def remove_all() -> int:
    """Drop every bot UUID (fresh fleet reset). Returns entries removed."""
    with _lock:
        n = len(_cache)
        if not n:
            return 0
        _cache.clear()
        _save()
        log.info("Bot registry cleared (%d entries)", n)
        return n
