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


def _read_all(max_bytes: int = 2_000_000) -> list[dict]:
    """Load the full equity log (or the tail if very large)."""
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


def _read_tail(max_bytes: int = 512_000) -> list[dict]:
    return _read_all(max_bytes=max_bytes)


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def series_since(
    start_date,
    *,
    end_date=None,
) -> list[tuple[datetime, float]]:
    """(utc_dt, equity) points on or after start_date (date in any tz sense)."""
    start_s = start_date.isoformat() if hasattr(start_date, "isoformat") else str(start_date)
    end_s = end_date.isoformat() if end_date and hasattr(end_date, "isoformat") else None
    out: list[tuple[datetime, float]] = []
    for row in _read_all():
        ts = _parse_ts(str(row.get("ts") or ""))
        if ts is None:
            continue
        day = ts.date().isoformat()
        if day < start_s:
            continue
        if end_s and day > end_s:
            continue
        try:
            eq = float(row["equity"])
        except (TypeError, ValueError, KeyError):
            continue
        if eq > 0:
            out.append((ts, eq))
    out.sort(key=lambda x: x[0])
    return out


def equity_curve_df(start_date, *, end_date=None):
    """DataFrame: time (UTC), equity, change_from_start.  None if empty."""
    import pandas as pd

    pts = series_since(start_date, end_date=end_date)
    if not pts:
        return None
    base = pts[0][1]
    rows = [
        {"time": ts, "equity": eq, "change": round(eq - base, 2)}
        for ts, eq in pts
    ]
    return pd.DataFrame(rows)


def period_stats_since(start_date, *, end_date=None) -> Optional[dict]:
    """Summary for a date range from equity snapshots."""
    pts = series_since(start_date, end_date=end_date)
    if not pts:
        return None
    eqs = [e for _, e in pts]
    return {
        "first_ts": pts[0][0],
        "last_ts": pts[-1][0],
        "first": eqs[0],
        "last": eqs[-1],
        "low": min(eqs),
        "high": max(eqs),
        "change": round(eqs[-1] - eqs[0], 2),
        "n": len(pts),
    }


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
