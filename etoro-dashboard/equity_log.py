"""Equity snapshot log — the ground-truth P&L layer.

The trade journal prices closes off ticks (~$0.8/trade optimistic, and once
reported −$434 on a −$3.8k equity day) and eToro's settled-history endpoint
is capped at 200 rows, so neither can tell you what a day REALLY made.
Equity can: this module appends a snapshot every few minutes to

    /app/data/equity_log.jsonl      {"ts": iso-utc, "equity": float}

driven by the positions poller (no extra API calls — it reuses the sizer's
cached account snapshot).  `day_stats()` turns the log into the day's true
equity change for the General Stats reconciliation line.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

LOG_PATH = Path(os.environ.get("EQUITY_LOG_PATH", "/app/data/equity_log.jsonl"))
SNAPSHOT_EVERY_SEC = 300.0          # one point every 5 min ≈ 288/day ≈ 20 KB/day

_lock = threading.Lock()
_last_write = 0.0


def maybe_snapshot(client, is_demo: bool) -> None:
    """Append an equity point if the cadence is due.  Cheap: reuses the
    position sizer's cached account snapshot (TTL-guarded), so this adds no
    API traffic of its own.  Never raises."""
    global _last_write
    now = time.time()
    with _lock:
        if now - _last_write < SNAPSHOT_EVERY_SEC:
            return
        _last_write = now
    try:
        import position_sizer
        snap = position_sizer._account_snapshot(client, is_demo)
        eq = float(snap.get("equity") or 0.0) if snap else 0.0
        if eq <= 0:
            return
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
                "equity": round(eq, 2),
            }) + "\n")
    except Exception:
        log.debug("Equity snapshot failed", exc_info=True)


def _read_tail(max_bytes: int = 512_000) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        size = LOG_PATH.stat().st_size
        with open(LOG_PATH, "rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, 2)
            text = f.read().decode(errors="ignore")
        rows = []
        for line in text.strip().splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows
    except Exception:
        return []


def day_stats(day: Optional[str] = None) -> Optional[dict]:
    """True equity stats for a UTC day: first/last/min snapshot + change.

    Returns {"first": $, "last": $, "low": $, "change": $, "n": points} or
    None when no snapshots exist for that day."""
    day = day or datetime.now(tz=timezone.utc).date().isoformat()
    pts = [r for r in _read_tail() if str(r.get("ts", "")).startswith(day)]
    if not pts:
        return None
    eqs = [float(r["equity"]) for r in pts if r.get("equity")]
    if not eqs:
        return None
    return {
        "first": eqs[0], "last": eqs[-1], "low": min(eqs),
        "change": round(eqs[-1] - eqs[0], 2), "n": len(eqs),
    }


def journal_day_pnl(day: Optional[str] = None) -> float:
    """Journal-claimed bot P&L for the same day — the optimistic number the
    reconciliation line compares against."""
    day = day or datetime.now(tz=timezone.utc).date().isoformat()
    try:
        import trade_journal
        return round(sum(
            float(r.get("pnl_dollars") or 0.0)
            for r in trade_journal.closed_records()
            if str(r.get("ts", ""))[:10] == day and (r.get("bot_id") or "").strip()
        ), 2)
    except Exception:
        return 0.0
