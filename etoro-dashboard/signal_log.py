"""
Persistent append-only log of every strategy signal decision (LLM and rule-based).

One JSON record per line (JSONL).  Thread-safe — multiple signal workers
may write concurrently; a module-level lock serialises file access.

Record shape:
  {
    "ts":                   "2026-06-06T10:32:15+00:00",   # UTC ISO-8601
    "type":                 "entry" | "exit",
    "instrument_id":        1011,
    "instrument_label":     "EUR/USD",
    "interval":             "1 Minute",
    "trigger_at":           "10:32:00",
    "signal":               "BUY" | "SELL" | "HOLD",       # entry only
    "action":               "CLOSE" | "HOLD",              # exit only
    "current_signal":       "BUY_LONG",                    # raw LLM token
    "confidence":           72,
    "reasoning":            "...",
    "risk_warning":         "...",
    "spread_impact":        "...",
    "observations":         ["Expected next: UP", ...],
    "expected_direction_next": "UP",
    "nearest_support":      "1.08200",
    "nearest_resistance":   "1.08900",
    "risk_level":           "MEDIUM",
    "trend_strength":       "STRONG",                      # exit only
    "profitable_before_spread": true,
    "profitable_after_spread":  false,
  }
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

LOG_PATH = Path(os.environ.get("SIGNAL_LOG_PATH", "/tmp/etoro_signal_log.jsonl"))
EXEC_PATH = Path(os.environ.get("SIGNAL_EXEC_PATH", "/tmp/etoro_signal_execution.json"))
_lock = threading.Lock()

# Execution outcomes keyed by bot+instrument+interval+trigger+type (sidecar store).
_exec_loaded = False
_executions: dict[str, dict] = {}
_EXEC_MAX_KEYS = 3000
_exec_dirty = False
_exec_last_save = 0.0
_EXEC_SAVE_INTERVAL = 5.0

# ── In-memory cache ───────────────────────────────────────────────────────────
# The Bots / Signals fragments call load() many times per refresh cycle (once
# per bot card).  Reading and reparsing the JSONL file on every call meant a
# 5 MB read every few seconds with a 10k-entry log.  Cache the last
# _CACHE_MAX records in memory so load() is a fast in-process scan; append()
# writes through to disk AND updates the cache.
_CACHE_MAX = 5000
_MAX_LOAD_BYTES = 4_000_000          # tail read — avoid blocking tab switches on huge logs
_cache: list[dict] = []
_cache_loaded: bool = False
_total_count: int = 0


def _read_tail_text(path: Path, max_bytes: int) -> str:
    """Read at most the last *max_bytes* of a file (may start mid-line)."""
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_text(encoding="utf-8", errors="ignore")
    with path.open("rb") as f:
        f.seek(size - max_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="ignore")
    nl = text.find("\n")
    return text[nl + 1:] if nl != -1 else text


def _load_cache_from_disk_locked() -> None:
    """Populate the cache from disk on first use.  Caller must hold _lock."""
    global _cache, _cache_loaded, _total_count
    if _cache_loaded:
        return
    _cache_loaded = True
    if not LOG_PATH.exists():
        return
    try:
        text = _read_tail_text(LOG_PATH, _MAX_LOAD_BYTES)
    except Exception as exc:
        log.warning("Signal log initial load failed: %s", exc)
        return
    all_records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            all_records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _total_count = len(all_records)
    # Keep only the tail in memory; older records can still be re-read from disk
    # but in practice the UI only ever asks for the most recent ones.
    if _total_count > _CACHE_MAX:
        _cache = all_records[-_CACHE_MAX:]
    else:
        _cache = all_records


# ── write ──────────────────────────────────────────────────────────────────────

def _exec_key(
    *,
    bot_id: str,
    instrument_id: int,
    interval: str,
    trigger_at: str,
    signal_type: str,
) -> str:
    return f"{bot_id}|{instrument_id}|{interval}|{trigger_at}|{signal_type}"


def _load_executions_locked() -> None:
    global _exec_loaded, _executions
    if _exec_loaded:
        return
    _exec_loaded = True
    if not EXEC_PATH.exists():
        return
    try:
        data = json.loads(EXEC_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _executions = data
    except Exception as exc:
        log.warning("Signal execution store load failed: %s", exc)


def _trim_executions_locked() -> None:
    overflow = len(_executions) - _EXEC_MAX_KEYS
    if overflow <= 0:
        return
    for old_key in list(_executions.keys())[:overflow]:
        _executions.pop(old_key, None)


def _save_executions_locked(*, force: bool = False) -> None:
    global _exec_dirty, _exec_last_save
    if not _exec_dirty and not force:
        return
    now = time.monotonic()
    if not force and now - _exec_last_save < _EXEC_SAVE_INTERVAL:
        return
    try:
        EXEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXEC_PATH.write_text(json.dumps(_executions), encoding="utf-8")
        _exec_dirty = False
        _exec_last_save = now
    except Exception as exc:
        log.warning("Signal execution store save failed: %s", exc)


def _attach_execution_meta(rec: dict, executions: dict[str, dict]) -> dict:
    """Attach exec_status / exec_reason — caller must already hold _lock."""
    key = _exec_key(
        bot_id=str(rec.get("bot_id") or ""),
        instrument_id=int(rec.get("instrument_id") or 0),
        interval=str(rec.get("interval") or ""),
        trigger_at=str(rec.get("trigger_at") or ""),
        signal_type=str(rec.get("type") or "entry"),
    )
    meta = executions.get(key)
    if not meta:
        return rec
    out = dict(rec)
    out["exec_status"] = meta.get("status")
    out["exec_reason"] = meta.get("reason", "")
    out["exec_at"] = meta.get("at", "")
    return out


def annotate_execution(
    *,
    bot_id: str,
    instrument_id: int,
    interval: str,
    trigger_at: str,
    signal_type: str,
    status: str,
    reason: str = "",
) -> None:
    """Record whether a logged signal resulted in an eToro order.

    status: executed | skipped | not_applicable
    """
    if not bot_id or not trigger_at:
        return
    from datetime import datetime, timezone

    key = _exec_key(
        bot_id=bot_id,
        instrument_id=instrument_id,
        interval=interval,
        trigger_at=trigger_at,
        signal_type=signal_type,
    )
    payload = {
        "status": status,
        "reason": reason,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    global _exec_dirty
    try:
        with _lock:
            _load_executions_locked()
            _executions[key] = payload
            _trim_executions_locked()
            _exec_dirty = True
            _save_executions_locked()
            # Keep in-memory signal cache in sync for immediate UI refresh.
            _load_cache_from_disk_locked()
            for i, rec in enumerate(_cache):
                if _exec_key(
                    bot_id=str(rec.get("bot_id") or ""),
                    instrument_id=int(rec.get("instrument_id") or 0),
                    interval=str(rec.get("interval") or ""),
                    trigger_at=str(rec.get("trigger_at") or ""),
                    signal_type=str(rec.get("type") or "entry"),
                ) == key:
                    updated = dict(rec)
                    updated["exec_status"] = status
                    updated["exec_reason"] = reason
                    updated["exec_at"] = payload["at"]
                    _cache[i] = updated
                    break
    except Exception as exc:
        log.warning("Signal execution annotate failed: %s", exc)


def append(record: dict[str, Any]) -> None:
    """Append one signal record to the JSONL log.  Thread-safe."""
    global _total_count
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            _load_cache_from_disk_locked()
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            _cache.append(record)
            _total_count += 1
            # Trim cache tail to bound memory
            overflow = len(_cache) - _CACHE_MAX
            if overflow > 0:
                del _cache[:overflow]
    except Exception as exc:
        log.warning("Signal log write failed: %s", exc)


# ── read ───────────────────────────────────────────────────────────────────────

def load(
    *,
    instrument_ids: Optional[list[int]] = None,
    signal_type: Optional[str] = None,       # "entry" | "exit"
    decisions: Optional[list[str]] = None,   # ["BUY", "SELL", ...]
    interval: Optional[str] = None,          # "1 Minute", "15 Minutes", …
    bot_id: Optional[str] = None,            # instruments.toml key e.g. "btc_15m"
    limit: int = 100,
) -> list[dict]:
    """Return records newest-first, with optional filters, capped at *limit*."""
    dec_upper = [d.upper() for d in decisions] if decisions else None
    records: list[dict] = []

    with _lock:
        _load_cache_from_disk_locked()
        _load_executions_locked()
        exec_map = _executions
        # Iterate cache in reverse — newest first.  Stop as soon as we collect
        # `limit` matches so common (small-limit) queries are O(limit).
        for rec in reversed(_cache):
            if instrument_ids is not None and rec.get("instrument_id") not in instrument_ids:
                continue
            if signal_type and rec.get("type") != signal_type:
                continue
            if dec_upper:
                sig = (rec.get("signal") or rec.get("action") or "").upper()
                if sig not in dec_upper:
                    continue
            if interval and rec.get("interval") != interval:
                continue
            # bot_id filter: require exact match when caller specifies a bot_id.
            # Records with no bot_id are treated as belonging to no specific bot
            # and are excluded from bot-scoped queries.
            if bot_id and rec.get("bot_id") != bot_id:
                continue

            records.append(_attach_execution_meta(rec, exec_map))
            if len(records) >= limit:
                break

    return records


def unique_instruments() -> list[dict]:
    """Return [{instrument_id, instrument_label}] seen in the log, de-duped."""
    seen: dict[int, str] = {}
    with _lock:
        _load_cache_from_disk_locked()
        for rec in _cache:
            iid = rec.get("instrument_id")
            label = rec.get("instrument_label", "")
            if isinstance(iid, int) and iid not in seen:
                seen[iid] = label
    return [{"instrument_id": k, "instrument_label": v} for k, v in seen.items()]


def total_count() -> int:
    """Total number of signal records logged this process lifetime."""
    with _lock:
        _load_cache_from_disk_locked()
        return _total_count
