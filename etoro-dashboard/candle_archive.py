"""Growing on-disk candle archive — the data moat.

eToro returns at most ~1000 candles per interval (10 days of 15m), so every
backtest and optimization runs on a thin window.  This module appends each
closed candle to the persistent volume so history accumulates forever:

    /app/data/candles/{instrument_id}_{interval_secs}.jsonl
    one line per candle: {"t": iso-utc, "o","h","l","c"}

A daemon thread refreshes each configured (instrument, interval) series on
its own cadence (a few REST calls per hour in total — negligible next to the
tick feeds).  `load_merged()` gives consumers (backtester / fleet sweeps) the
archive merged with a fresh fetch, deduped and time-sorted, so optimizations
automatically deepen as the archive grows.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

ARCHIVE_DIR = Path(os.environ.get("CANDLE_ARCHIVE_DIR", "/app/data/candles"))

# Series worth archiving: every configured instrument at the fleet's working
# intervals.  1m is deliberately excluded (the 1m fleet is retired; volume
# would dwarf its value).
ARCHIVE_SECS = (600, 900, 1800, 3600)

_lock = threading.Lock()
_last_ts: dict[tuple[int, int], str] = {}     # (iid, secs) → last archived iso ts
_thread: Optional[threading.Thread] = None
_started = False


def _path(iid: int, secs: int) -> Path:
    return ARCHIVE_DIR / f"{iid}_{secs}.jsonl"


def _df_rows(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in df.iterrows():
        t = r["time"]
        iso = t.isoformat() if hasattr(t, "isoformat") else str(t)
        out.append({
            "t": iso,
            "o": float(r["Open"]), "h": float(r["High"]),
            "l": float(r["Low"]),  "c": float(r["Close"]),
        })
    return out


def append_candles(iid: int, secs: int, df: pd.DataFrame) -> int:
    """Append candles newer than the archive tail.  The LAST fetched bar is
    skipped — it is still forming and would archive a partial candle."""
    if df is None or len(df) < 2:
        return 0
    closed = df.iloc[:-1]
    rows = _df_rows(closed)
    key = (iid, secs)
    path = _path(iid, secs)
    with _lock:
        last = _last_ts.get(key)
        if last is None and path.exists():
            try:
                tail = path.read_bytes()[-200:].decode(errors="ignore").strip().splitlines()
                last = json.loads(tail[-1])["t"] if tail else ""
            except Exception:
                last = ""
        last = last or ""
        fresh = [r for r in rows if r["t"] > last]
        if not fresh:
            return 0
        try:
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for r in fresh:
                    f.write(json.dumps(r) + "\n")
            _last_ts[key] = fresh[-1]["t"]
        except Exception:
            log.warning("Candle archive write failed for %s_%s", iid, secs, exc_info=True)
            return 0
    return len(fresh)


def load_archive(iid: int, secs: int) -> Optional[pd.DataFrame]:
    """Full archived series as a backtester-shaped DataFrame (or None)."""
    path = _path(iid, secs)
    if not path.exists():
        return None
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None
    if not rows:
        return None
    df = pd.DataFrame(rows).rename(columns={
        "t": "time", "o": "Open", "h": "High", "l": "Low", "c": "Close",
    })
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)


def load_merged(iid: int, secs: int, fetched: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Archive + fresh fetch, deduped on time.  Returns whichever is longer /
    the union — backtests silently deepen as the archive grows."""
    arch = load_archive(iid, secs)
    if arch is None or arch.empty:
        return fetched
    if fetched is None or fetched.empty:
        return arch
    f = fetched.copy()
    f["time"] = pd.to_datetime(f["time"], utc=True)
    merged = (
        pd.concat([arch, f[["time", "Open", "High", "Low", "Close"]]])
        .drop_duplicates(subset="time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return merged


def _archiver_loop(api_key: str, user_key: str) -> None:
    from etoro_client import get_shared_client
    import instrument_config

    client = get_shared_client(api_key, user_key)
    next_due: dict[tuple[int, int], float] = {}
    while True:
        try:
            specs = instrument_config.load_specs()
            iids = sorted({s.instrument_id for s in specs if s.instrument_id})
            now = time.time()
            for iid in iids:
                for secs in ARCHIVE_SECS:
                    key = (iid, secs)
                    if now < next_due.get(key, 0.0):
                        continue
                    # Refresh each series once per its own interval (+ jitter
                    # via the scan cadence) — a closed candle appears at most
                    # one interval late, REST load stays trivial.
                    next_due[key] = now + secs
                    try:
                        df = client.get_hist_candles(iid, secs, 60)
                        n = append_candles(iid, secs, df)
                        if n:
                            log.debug("Archived %d candle(s) for %s_%s", n, iid, secs)
                    except Exception:
                        log.debug("Archive fetch failed for %s_%s", iid, secs)
                    time.sleep(1.0)        # stay far under the REST rate limit
        except Exception:
            log.exception("Candle archiver iteration failed")
        time.sleep(30.0)


def ensure_archiver(api_key: str, user_key: str) -> None:
    """Start the background archiver once per process."""
    global _thread, _started
    if _started and _thread and _thread.is_alive():
        return
    if not api_key or not user_key:
        return
    _started = True
    _thread = threading.Thread(
        target=_archiver_loop, args=(api_key, user_key),
        daemon=True, name="candle-archiver",
    )
    _thread.start()
    log.info("Candle archiver started (%s, intervals %s)", ARCHIVE_DIR, ARCHIVE_SECS)


def archive_stats() -> dict[str, int]:
    """series → archived candle count (for the UI / sanity checks)."""
    out: dict[str, int] = {}
    if not ARCHIVE_DIR.exists():
        return out
    for p in sorted(ARCHIVE_DIR.glob("*.jsonl")):
        try:
            with open(p, "rb") as f:
                out[p.stem] = sum(1 for _ in f)
        except Exception:
            continue
    return out
