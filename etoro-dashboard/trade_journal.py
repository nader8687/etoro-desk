"""
Durable trade-outcome journal + evidence-based learning loop.

This is the system's long-term memory of how its own decisions actually played
out.  Every closed trade is appended to a JSONL file (so it survives container
restarts - unlike trade_manager._closed which is in-memory only) together with
the full decision context: strategy, confidence, spread/slippage, position
size, holding time, exit reason, execution-risk tier, hour-of-day and final
P&L.

Three consumers read it back:

  performance_stats()  ->  Dashboard "Performance & Lessons" view: win/loss
                          patterns sliced by strategy, direction, confidence,
                          exec-risk, holding time and hour.

  entry_guidance()     ->  trading_engine._maybe_open_trade: a CONSERVATIVE,
                          evidence-based veto.  It only blocks a setup when a
                          *statistically meaningful* bucket (≥ MIN_BUCKET_N
                          trades) has clearly negative expectancy.  With little
                          or no history it always allows - so the bot's
                          behaviour is unchanged until real evidence accrues.
                          This is the anti-overfitting guarantee.

  llm_memory_block()   ->  injected into the Vision-LLM prompt so the model sees
                          a short, honest summary of its recent track record on
                          this instrument and the lessons that follow from it.

Design notes
------------
* Thread-safe: a module lock serialises file writes and cache mutation, mirroring
  signal_log.py.  record() is called from trade_manager close paths which run in
  engine tick threads.
* Bounded memory: only the last _CACHE_MAX records are held in RAM; the file is
  the source of truth and is re-readable.
* Pure-Python / no pandas dependency in the hot path - record() must be cheap.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

JOURNAL_PATH = Path(os.environ.get("TRADE_JOURNAL_PATH", "/tmp/etoro_trade_journal.jsonl"))

# ── Learning-loop tunables ────────────────────────────────────────────────────
# Minimum trades in a bucket before we let it influence a decision.  This is the
# core anti-overfitting lever: small samples are noise, so they never veto.
MIN_BUCKET_N      = 8
# A bucket is "clearly losing" only when BOTH conditions hold on a real sample.
LOSE_WINRATE_MAX  = 0.60    # ≤ 60 % winners — raised from 0.40: the journal's
                            # documented failure mode is HIGH-winrate buckets
                            # that still lose money (win pennies, lose big); at
                            # 0.40 the veto could never see them.
LOSE_PROFIT_FACTOR_MAX = 0.75   # gross wins / gross losses < 0.75
# Confidence buckets used for slicing + guidance.
CONF_LOW_MAX  = 55          # < 55  -> "low"
CONF_HIGH_MIN = 70          # ≥ 70  -> "high"; between is "mid"

# Memory bounds - keep RAM and prompt tokens predictable regardless of how
# large the on-disk journal grows:
#   _CACHE_MAX        caps records held in RAM (analysis window).
#   _MAX_LOAD_BYTES   caps the initial disk read (tail only) so a multi-MB
#                     journal never loads in full.
#   _MEM_*_CHARS      hard caps on the text injected into LLM prompts so the
#                     context window can never overflow from memory growth.
_CACHE_MAX = 5000
_MAX_LOAD_BYTES = 4_000_000          # ~4 MB tail ~ 12k records
_MEM_ENTRY_CHARS = 600               # entry-prompt memory budget (~160 tokens)
_MEM_EXIT_CHARS = 420                # exit-prompt memory budget (~110 tokens)
_MEM_WINDOW = 40                     # only the last N trades feed the memory block

# Estimated round-trip commission as % of notional.  eToro crypto is spread-based
# (no explicit commission), so 0.0 by default; override via TRADE_FEE_PCT to model
# a fee-bearing venue in analytics/backtests without touching live behaviour.
_FEE_PCT = float(os.environ.get("TRADE_FEE_PCT", "0.0"))

# Confidence-recalibration tunables (Bayesian shrinkage toward realised win rate).
CONF_CAL_K = 12             # sample size at which the empirical winrate earns ~50% weight
CONF_CAL_MIN_N = 5          # below this many samples, return raw confidence (no overfit)
CONF_CAL_WINDOW = 60        # rolling closed-trade window per strategy

_lock = threading.Lock()
_cache: list[dict] = []
_cache_loaded = False
_total_count = 0
# (st_mtime_ns, st_size) of the file at the time the cache was last loaded.  Used
# to detect external writes (e.g. a backfill script, or another process) so a
# long-running Streamlit process re-reads the journal instead of serving a stale
# in-memory copy — the cause of the Performance tab appearing empty.
_loaded_sig: "tuple | None" = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _conf_bucket(confidence: Optional[int]) -> str:
    c = int(confidence or 0)
    if c < CONF_LOW_MAX:
        return "low"
    if c >= CONF_HIGH_MIN:
        return "high"
    return "mid"


def _hold_bucket(minutes: float) -> str:
    if minutes < 2:
        return "<2m"
    if minutes < 10:
        return "2-10m"
    if minutes < 30:
        return "10-30m"
    return ">30m"


def _safe_iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _asset_class_of(instrument_label: str) -> str:
    """Asset class for a label (delegates to exit_profiles; 'unknown' on error)."""
    try:
        import exit_profiles
        return exit_profiles.asset_class(instrument_label)
    except Exception:
        return "unknown"


def _take_profit_pct_of(strategy: str, instrument_label: str) -> float:
    """Configured hard take-profit % for this strategy/asset (0 = trailing-only)."""
    try:
        import exit_profiles
        _, tp = exit_profiles.resolve(strategy, instrument_label=instrument_label)
        return round(float(tp), 4)
    except Exception:
        return 0.0


def calibrated_confidence(
    strategy: str,
    direction: str,
    raw_confidence: int,
    *,
    window: int = CONF_CAL_WINDOW,
) -> int:
    """Shrink a strategy's stated confidence toward its REALISED win rate.

    cal = raw·(1−w) + winrate·100·w,  with  w = n/(n+CONF_CAL_K).
    Conservative by construction: with < CONF_CAL_MIN_N samples it returns the
    raw confidence untouched, and the empirical weight grows only as evidence
    accrues — so it can never overfit to a handful of trades.  Intended for
    sizing / edge decisions and analytics, NOT to retroactively rewrite the
    confidence buckets that entry_guidance learns from.
    """
    raw = int(raw_confidence or 0)
    try:
        rows = closed_records(strategy=strategy)
    except Exception:
        return raw
    rows = [r for r in rows if (r.get("bot_id") or "").strip()]
    if direction:
        rows = [r for r in rows if (r.get("direction") or "").upper() == direction.upper()]
    rows = rows[:window]
    n = len(rows)
    if n < CONF_CAL_MIN_N:
        return raw
    winrate = sum(1 for r in rows if r.get("win")) / n
    w = n / (n + CONF_CAL_K)
    cal = raw * (1 - w) + winrate * 100.0 * w
    return int(max(1, min(99, round(cal))))


def _read_tail_text(path: Path, max_bytes: int) -> str:
    """Read at most the last *max_bytes* of a file, dropping any leading partial
    line.  Bounds initial-load RAM no matter how large the journal grows."""
    size = path.stat().st_size
    if size <= max_bytes:
        return path.read_text(encoding="utf-8")
    with path.open("rb") as f:
        f.seek(size - max_bytes)
        data = f.read()
    text = data.decode("utf-8", errors="ignore")
    nl = text.find("\n")
    return text[nl + 1:] if nl != -1 else text


def _truncate(text: str, max_chars: int) -> str:
    """Hard-cap a memory string, cutting on a line boundary when possible."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars * 0.5:
        cut = cut[:nl]
    return cut.rstrip() + " ..."


def _file_sig() -> "tuple | None":
    """(mtime_ns, size) of the journal file, or None if it doesn't exist."""
    try:
        stt = JOURNAL_PATH.stat()
        return (stt.st_mtime_ns, stt.st_size)
    except OSError:
        return None


def _load_cache_locked() -> None:
    """Load the journal into the in-memory cache, reloading whenever the file has
    changed on disk since the last load.  Crucially this does NOT latch on a
    transient read failure, so a hiccup at boot can't leave the cache empty for
    the life of the process."""
    global _cache, _cache_loaded, _total_count, _loaded_sig
    sig = _file_sig()
    if _cache_loaded and sig == _loaded_sig:
        return                              # cache already reflects the file
    if sig is None:                         # file not present yet — nothing to load
        _cache_loaded = True
        _loaded_sig = None
        return
    try:
        text = _read_tail_text(JOURNAL_PATH, _MAX_LOAD_BYTES)
    except Exception as exc:
        log.warning("Trade journal load failed (will retry): %s", exc)
        return                              # do not latch — retry on next access
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    _total_count = len(records)
    _cache = records[-_CACHE_MAX:] if _total_count > _CACHE_MAX else records
    _cache_loaded = True
    _loaded_sig = sig


# ── write ─────────────────────────────────────────────────────────────────────

def record(trade: Any, closed: Any) -> None:
    """Append one closed-trade outcome to the journal.

    `trade`  = the live PaperTrade (carries entry decision context).
    `closed` = the ClosedTrade (carries exit price/time/profit/reason).
    Both are duck-typed to avoid importing trade_manager (circular).
    Never raises - journaling must not break the trade-close path.
    """
    global _total_count, _loaded_sig
    try:
        entry_dt = _parse_dt(getattr(closed, "entry_time", None))
        exit_dt  = _parse_dt(getattr(closed, "exit_time", None)) or datetime.now(timezone.utc)
        holding_min = (
            (exit_dt - entry_dt).total_seconds() / 60.0 if entry_dt else 0.0
        )

        entry_price = float(getattr(closed, "entry_price", 0.0) or 0.0)
        exit_price  = float(getattr(closed, "exit_price", 0.0) or 0.0)
        spread      = float(getattr(closed, "entry_spread", 0.0) or 0.0)
        direction   = str(getattr(closed, "direction", "") or "")
        profit_px   = float(getattr(closed, "profit", 0.0) or 0.0)   # price terms
        amount      = float(getattr(trade, "trade_amount", 0.0) or 0.0)

        # P&L as a percent of entry, and in dollars on the position notional.
        pnl_pct = (profit_px / entry_price * 100.0) if entry_price else 0.0
        units   = (amount / entry_price) if entry_price else 0.0
        pnl_dollars = profit_px * units
        slippage_pct = (spread / entry_price * 100.0) if entry_price else 0.0

        rec = {
            "ts":            exit_dt.isoformat(),
            "entry_time":    _safe_iso(getattr(closed, "entry_time", "")),
            "exit_time":     exit_dt.isoformat(),
            "instrument_id": getattr(closed, "instrument_id", 0),
            "instrument_label": getattr(closed, "instrument_label", ""),
            "bot_id":        getattr(closed, "bot_id", "") or "",
            # Legacy virtual trade — kept OUT of money stats.
            "shadow":        bool(getattr(closed, "shadow", False)),
            "direction":     direction,
            # No strategy ⇒ the trade wasn't opened by a strategy/LLM bot, so it's
            # a manual/user trade — label it "manual" so it forms one filterable
            # category instead of an ambiguous "unknown" bucket.
            "strategy":      (getattr(trade, "strategy", "") or "").strip() or "manual",
            "signal":        getattr(closed, "signal", "") or "",
            "confidence":    int(getattr(closed, "confidence", 0) or 0),
            "entry_price":   entry_price,
            "exit_price":    exit_price,
            "entry_spread":  spread,
            "slippage_pct":  round(slippage_pct, 4),
            "trade_amount":  amount,
            "exec_risk":     getattr(trade, "exec_risk", "") or "",
            "net_edge_pct":  float(getattr(trade, "net_edge_pct", 0.0) or 0.0),
            # ── Richer context for analytics & backtesting ───────────────────
            "interval":      getattr(trade, "interval", "") or "",
            "entry_reason":  getattr(trade, "entry_reason", "") or "",
            "regime":        getattr(trade, "regime", "") or "",
            "atr_pct_entry": round(float(getattr(trade, "atr_pct_entry", 0.0) or 0.0), 4),
            "stop_pct_entry": round(float(getattr(trade, "stop_pct_entry", 0.0) or 0.0), 4),
            "asset_class":   _asset_class_of(getattr(closed, "instrument_label", "")),
            "take_profit_pct": _take_profit_pct_of(
                getattr(trade, "strategy", ""), getattr(closed, "instrument_label", "")),
            "fee_pct":       round(_FEE_PCT, 4),
            "fee_dollars":   round(amount * _FEE_PCT / 100.0, 4),
            "confidence_calibrated": int(getattr(trade, "confidence_calibrated", 0) or 0),
            "reason":        getattr(closed, "reason", "") or "",
            # Exit-tuning evidence: best unrealised $ P&L the trade ever reached
            # and the stop level it ran with — lets us backtest "would a tighter
            # TP / trailing stop have banked this?" from the journal alone.
            "peak_pnl":      round(float(getattr(trade, "peak_pnl", 0.0) or 0.0), 4),
            "stop_loss_price": float(getattr(closed, "stop_loss_price", 0.0) or 0.0),
            "holding_min":   round(holding_min, 2),
            "pnl_abs":       round(profit_px, 6),
            "pnl_dollars":   round(pnl_dollars, 4),
            "pnl_pct":       round(pnl_pct, 4),
            "win":           profit_px > 0,
            "hour":          entry_dt.hour if entry_dt else exit_dt.hour,
            "etoro_position_id": getattr(closed, "etoro_position_id", None),
        }

        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            _load_cache_locked()
            with JOURNAL_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            _cache.append(rec)
            _total_count += 1
            overflow = len(_cache) - _CACHE_MAX
            if overflow > 0:
                del _cache[:overflow]
            _loaded_sig = _file_sig()   # our own write — keep cache in sync, no reload
        log.info(
            "Journaled %s %s %s pnl=$%.2f (%.2f%%) reason=%s hold=%.1fm",
            rec["instrument_label"], rec["direction"], rec["strategy"],
            rec["pnl_dollars"], rec["pnl_pct"], rec["reason"], rec["holding_min"],
        )
    except Exception:
        log.warning("Trade journal record failed", exc_info=True)


# ── read ──────────────────────────────────────────────────────────────────────

def _filtered(
    instrument_id: Optional[int] = None,
    strategy: Optional[str] = None,
    bot_id: Optional[str] = None,
    include_shadow: bool = False,
) -> list[dict]:
    """Legacy shadow (virtual) trades are EXCLUDED by default."""
    with _lock:
        _load_cache_locked()
        rows = list(_cache)
    out = []
    for r in rows:
        if not include_shadow and r.get("shadow"):
            continue
        if instrument_id is not None and r.get("instrument_id") != instrument_id:
            continue
        if strategy and r.get("strategy") != strategy:
            continue
        if bot_id and r.get("bot_id") != bot_id:
            continue
        out.append(r)
    return out


def strategy_recent(
    strategy: str,
    limit: int = 40,
    include_shadow: bool = False,
) -> list[dict]:
    """Most-recent closed BOT trades for one strategy (oldest→newest)."""
    rows = [
        r for r in _filtered(None, strategy or None, include_shadow=include_shadow)
        if (r.get("bot_id") or "").strip()
    ]
    return rows[-int(limit):]


def bot_recent(
    bot_id: str,
    limit: int = 40,
    include_shadow: bool = False,
) -> list[dict]:
    """Most-recent closed trades for one bot UUID (oldest→newest)."""
    bid = (bot_id or "").strip()
    if not bid:
        return []
    rows = [
        r for r in _filtered(None, None, bid, include_shadow=include_shadow)
        if (r.get("bot_id") or "").strip() == bid
    ]
    return rows[-int(limit):]


def context_recent(
    strategy: str,
    interval: str,
    instrument_id: int = 0,
    limit: int = 40,
    include_shadow: bool = False,
) -> list[dict]:
    """Closed trades for one bot context: strategy + interval + instrument."""
    strat = (strategy or "").strip()
    iv = (interval or "").strip()
    rows = [
        r for r in _filtered(instrument_id or None, strat or None, include_shadow=include_shadow)
        if (r.get("bot_id") or "").strip()
    ]
    if iv:
        rows = [r for r in rows if (r.get("interval") or "").strip() == iv]
    return rows[-int(limit):]


def _agg(rows: list[dict]) -> dict:
    """Aggregate a list of trade records into summary stats."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "expectancy": 0.0, "total_pnl": 0.0, "avg_hold_min": 0.0,
        }
    wins   = [r for r in rows if r.get("win")]
    losses = [r for r in rows if not r.get("win")]
    gross_win  = sum(r.get("pnl_dollars", 0.0) for r in wins)
    gross_loss = sum(r.get("pnl_dollars", 0.0) for r in losses)   # ≤ 0
    total      = gross_win + gross_loss
    avg_win  = gross_win / len(wins) if wins else 0.0
    avg_loss = gross_loss / len(losses) if losses else 0.0
    # profit factor = gross profit / gross loss magnitude
    pf = (gross_win / abs(gross_loss)) if gross_loss < 0 else (float("inf") if gross_win > 0 else 0.0)
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / n,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": pf,
        "expectancy": total / n,
        "total_pnl": total,
        "avg_hold_min": sum(r.get("holding_min", 0.0) for r in rows) / n,
    }


def _group_agg(rows: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        groups.setdefault(str(k), []).append(r)
    return {k: _agg(v) for k, v in groups.items()}


def performance_stats(
    instrument_id: Optional[int] = None,
    strategy: Optional[str] = None,
    *,
    rows: Optional[list[dict]] = None,
) -> dict:
    """Full performance breakdown for the dashboard.  Safe on empty data.

    Pass *rows* to analyse a pre-filtered subset (e.g. a date range from the
    Performance tab) instead of the full journal."""
    if rows is None:
        rows = _filtered(instrument_id, strategy)
    else:
        rows = list(rows)
        if instrument_id is not None:
            rows = [r for r in rows if r.get("instrument_id") == instrument_id]
        if strategy:
            rows = [r for r in rows if r.get("strategy") == strategy]
    return {
        "overall":       _agg(rows),
        "by_strategy":   _group_agg(rows, lambda r: (r.get("strategy") or "").strip() or "manual"),
        "by_direction":  _group_agg(rows, lambda r: r.get("direction") or "?"),
        "by_reason":     _group_agg(rows, lambda r: r.get("reason") or "?"),
        "by_confidence": _group_agg(rows, lambda r: _conf_bucket(r.get("confidence"))),
        "by_exec_risk":  _group_agg(rows, lambda r: r.get("exec_risk") or "unknown"),
        "by_holding":    _group_agg(rows, lambda r: _hold_bucket(r.get("holding_min", 0.0))),
        "by_hour":       _group_agg(rows, lambda r: f"{int(r.get('hour', 0)):02d}h"),
        "sample":        rows[-200:],
    }


def avg_hold_min(strategy: str, *, window: int = 30) -> float:
    """Average holding time (minutes) of a strategy's recent closed BOT trades.

    Used by cash-freeing to judge whether an open position has *overstayed* its
    strategy's natural duration.  Returns 0.0 when there isn't enough history to
    be meaningful (so callers treat overstay as unknown rather than guess)."""
    try:
        rows = [r for r in _filtered(None, strategy or None) if (r.get("bot_id") or "").strip()]
    except Exception:
        return 0.0
    vals = [float(r.get("holding_min") or 0.0) for r in rows[-window:]]
    vals = [v for v in vals if v > 0]
    if len(vals) < 3:
        return 0.0
    return sum(vals) / len(vals)


# ── learning: evidence-based entry guidance ───────────────────────────────────

def _learning_thresholds() -> tuple[int, float, float]:
    try:
        import user_settings
        ls = user_settings.learning_settings()
        return ls.min_bucket_n, ls.lose_winrate_max, ls.lose_profit_factor_max
    except Exception:
        return MIN_BUCKET_N, LOSE_WINRATE_MAX, LOSE_PROFIT_FACTOR_MAX


def entry_guidance(
    instrument_id: int,
    strategy: str,
    direction: str,
    confidence: int,
    exec_risk: str = "",
) -> dict:
    """Conservative, evidence-based pre-trade check.

    Returns {"allow": bool, "reason": str, "caution": str}.

    BLOCKS (allow=False) only when a *specific, well-sampled* bucket matching
    this setup has clearly negative expectancy - i.e. ≥ MIN_BUCKET_N trades with
    win-rate ≤ LOSE_WINRATE_MAX AND profit-factor < LOSE_PROFIT_FACTOR_MAX AND
    negative total P&L.  Otherwise always allows (optionally with a soft caution
    note).  With little/no history every setup is allowed -> no overfitting, no
    behaviour change until evidence exists.
    """
    try:
        import user_settings
        if not user_settings.learning_settings().entry_guidance_enabled:
            return {"allow": True, "reason": "", "caution": ""}
    except Exception:
        pass

    min_bucket_n, lose_wr, lose_pf = _learning_thresholds()
    rows = _filtered(instrument_id, strategy or None)
    if len(rows) < min_bucket_n:
        return {"allow": True, "reason": "", "caution": ""}

    direction = (direction or "").upper()
    conf_b    = _conf_bucket(confidence)

    # Candidate sub-buckets, most-specific first.  Each is only consulted when it
    # has a meaningful sample of its own.
    candidates = [
        ("direction+confidence",
         [r for r in rows if r.get("direction") == direction
          and _conf_bucket(r.get("confidence")) == conf_b]),
        ("confidence",
         [r for r in rows if _conf_bucket(r.get("confidence")) == conf_b]),
        ("direction",
         [r for r in rows if r.get("direction") == direction]),
    ]
    if exec_risk:
        candidates.insert(0, (
            "exec_risk+direction",
            [r for r in rows if r.get("exec_risk") == exec_risk
             and r.get("direction") == direction],
        ))

    for label, bucket in candidates:
        if len(bucket) < min_bucket_n:
            continue
        stats = _agg(bucket)
        clearly_losing = (
            stats["win_rate"] <= lose_wr
            and stats["profit_factor"] < lose_pf
            and stats["total_pnl"] < 0
        )
        if clearly_losing:
            return {
                "allow": False,
                "reason": (
                    f"Historically weak setup [{label}={direction}/{conf_b}"
                    f"{('/' + exec_risk) if exec_risk else ''}]: "
                    f"{stats['wins']}/{stats['n']} wins "
                    f"({stats['win_rate']*100:.0f}%), "
                    f"PF {stats['profit_factor']:.2f}, "
                    f"net ${stats['total_pnl']:+.2f} over {stats['n']} trades"
                ),
                "caution": "",
            }
        # Marginal bucket -> allow but flag a caution for observability.
        if stats["win_rate"] < 0.5 and stats["total_pnl"] < 0:
            return {
                "allow": True,
                "reason": "",
                "caution": (
                    f"{label} {direction}/{conf_b}: "
                    f"{stats['win_rate']*100:.0f}% win, net ${stats['total_pnl']:+.2f} "
                    f"(n={stats['n']}) - marginal, watch closely"
                ),
            }
    return {"allow": True, "reason": "", "caution": ""}


# ── learning: inline memory block for the Vision LLM ──────────────────────────

def _fmt_pf(pf: float) -> str:
    # ASCII-only - this string flows into LLM prompts and logs which may run on
    # non-UTF-8 streams; the Streamlit view has its own unicode formatter.
    if pf == float("inf"):
        return "inf"
    return f"{pf:.2f}"


def recent_episodes(
    instrument_id: int,
    strategy: str,
    direction: Optional[str] = None,
    n: int = 2,
) -> list[str]:
    """A few concrete recent trades, newest first - the 'episodic' half of
    experience.  Each line is compact (one short sentence) to stay token-cheap."""
    rows = _filtered(instrument_id, strategy or None)
    if direction:
        rows = [r for r in rows if r.get("direction") == direction.upper()]
    out: list[str] = []
    for r in reversed(rows[-20:]):
        out.append(
            f"{r.get('direction','?')} @conf{int(r.get('confidence',0))} -> "
            f"{'WIN' if r.get('win') else 'LOSS'} ${r.get('pnl_dollars',0.0):+.0f} "
            f"in {r.get('holding_min',0.0):.0f}m ({r.get('reason','?')})"
        )
        if len(out) >= n:
            break
    return out


def llm_memory_block(
    instrument_id: int,
    strategy: str = "llm",
    max_trades: int = _MEM_WINDOW,
    max_chars: int = _MEM_ENTRY_CHARS,
) -> str:
    """A short, honest, token-bounded track record for the entry prompt.

    Combines a reflective summary (aggregate stats + best/worst setup + exit
    pattern) with 1-2 concrete recent episodes, then hard-caps the whole thing
    to max_chars so the LLM context can never overflow from memory growth.
    Empty string when there isn't enough history - so the prompt is unchanged on
    a fresh system.  Never prescriptive beyond evidence-grounded lessons.
    """
    rows = _filtered(instrument_id, strategy or None)
    if len(rows) < _learning_thresholds()[0]:
        return ""
    rows = rows[-max_trades:]
    o = _agg(rows)

    lines = [
        f"Your recent record on this instrument ({strategy}): "
        f"{o['wins']}/{o['n']} wins ({o['win_rate']*100:.0f}%), "
        f"profit factor {_fmt_pf(o['profit_factor'])}, "
        f"net ${o['total_pnl']:+.2f}, avg hold {o['avg_hold_min']:.0f} min."
    ]

    # Best & worst direction/confidence buckets, when each is well-sampled.
    dir_conf = _group_agg(
        rows, lambda r: f"{r.get('direction','?')}/{_conf_bucket(r.get('confidence'))}"
    )
    ranked = [
        (k, s) for k, s in dir_conf.items() if s["n"] >= max(4, MIN_BUCKET_N // 2)
    ]
    ranked.sort(key=lambda kv: kv[1]["expectancy"])
    if ranked:
        worst_k, worst = ranked[0]
        if worst["expectancy"] < 0 and worst["win_rate"] <= 0.45:
            lines.append(
                f"Weakest setup: {worst_k} lost on average "
                f"(${worst['expectancy']:+.2f}/trade, {worst['win_rate']*100:.0f}% win, "
                f"n={worst['n']}) - be stricter here; prefer HOLD unless the signal is strong."
            )
        best_k, best = ranked[-1]
        if best["expectancy"] > 0 and best_k != worst_k:
            lines.append(
                f"Strongest setup: {best_k} was profitable "
                f"(${best['expectancy']:+.2f}/trade, {best['win_rate']*100:.0f}% win, "
                f"n={best['n']})."
            )

    # Exit-reason lesson: are we bleeding from stop-losses (late/poor entries) or
    # giving back gains (late exits)?
    by_reason = _group_agg(rows, lambda r: r.get("reason") or "?")
    sl = by_reason.get("stop_loss")
    if sl and sl["n"] >= 4 and sl["total_pnl"] < 0:
        share = sl["n"] / o["n"] * 100
        lines.append(
            f"{share:.0f}% of trades hit the stop-loss (net ${sl['total_pnl']:+.2f}) - "
            f"avoid late or low-conviction entries that get stopped out."
        )

    # Episodic recall - a couple of concrete recent outcomes.
    eps = recent_episodes(instrument_id, strategy, n=2)
    if eps:
        lines.append("Recent trades: " + "; ".join(eps) + ".")

    lines.append(
        "Use this as discipline, not a guarantee - judge the current chart on its own merits."
    )
    return _truncate("\n".join(lines), max_chars)


def exit_memory_block(
    instrument_id: int,
    strategy: str = "llm",
    max_chars: int = _MEM_EXIT_CHARS,
) -> str:
    """Token-bounded exit-discipline memory for the open-position prompt.

    Reflects on whether past exits captured gains or gave them back, whether
    losers were held too long, and how stop-outs went - the 'when to let go'
    wisdom.  Empty until enough history exists.
    """
    rows = _filtered(instrument_id, strategy or None)
    if len(rows) < _learning_thresholds()[0]:
        return ""
    rows = rows[-_MEM_WINDOW:]
    by_reason = _group_agg(rows, lambda r: r.get("reason") or "?")
    lines: list[str] = []

    # Discretionary (LLM) closes - did they tend to be the right call?
    llm_r = by_reason.get("llm")
    if llm_r and llm_r["n"] >= 3:
        verdict = (
            "they captured gains overall"
            if llm_r["total_pnl"] > 0
            else "they tended to be mistimed (net loss) - be more selective"
        )
        lines.append(
            f"Your discretionary closes: {llm_r['wins']}/{llm_r['n']} good, "
            f"net ${llm_r['total_pnl']:+.2f} - {verdict}."
        )

    # Holding-time asymmetry - are losers held longer than winners?
    wins   = [r for r in rows if r.get("win")]
    losses = [r for r in rows if not r.get("win")]
    if len(wins) >= 3 and len(losses) >= 3:
        wh = sum(r.get("holding_min", 0.0) for r in wins) / len(wins)
        lh = sum(r.get("holding_min", 0.0) for r in losses) / len(losses)
        if lh > wh * 1.3:
            lines.append(
                f"You held losers longer ({lh:.0f}m) than winners ({wh:.0f}m) - "
                f"cut losing positions sooner."
            )
        elif wh > lh * 1.3:
            lines.append(
                f"Winners ran ({wh:.0f}m) longer than losers ({lh:.0f}m) - good; "
                f"keep letting winners breathe and exit losers decisively."
            )

    sl = by_reason.get("stop_loss")
    if sl and sl["n"] >= 3 and sl["total_pnl"] < 0:
        lines.append(
            f"{sl['n']} trades ran into the stop (net ${sl['total_pnl']:+.2f}) - "
            f"when momentum clearly turns against you, close before the stop hits."
        )

    if not lines:
        return ""
    lines.append("Protect profits - a gain given back is worse than a gain banked.")
    return _truncate("\n".join(lines), max_chars)


def add_external_records(records: list[dict]) -> int:
    """Append pre-built journal records (e.g. backfilled from eToro trade
    history), deduped against existing rows by etoro_position_id.  Returns the
    number of NEW records written.  Lets the Performance tab show real outcomes
    immediately, before the engine has closed trades itself."""
    if not records:
        return 0
    global _total_count, _loaded_sig
    added = 0
    try:
        JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            _load_cache_locked()
            seen = {
                r.get("etoro_position_id")
                for r in _cache
                if r.get("etoro_position_id") is not None
            }
            with JOURNAL_PATH.open("a", encoding="utf-8") as f:
                for rec in records:
                    pid = rec.get("etoro_position_id")
                    if pid is not None and pid in seen:
                        continue
                    if pid is not None:
                        seen.add(pid)
                    f.write(json.dumps(rec, default=str) + "\n")
                    _cache.append(rec)
                    _total_count += 1
                    added += 1
            overflow = len(_cache) - _CACHE_MAX
            if overflow > 0:
                del _cache[:overflow]
            _loaded_sig = _file_sig()   # our own write — keep cache in sync
    except Exception:
        log.warning("Trade journal external ingest failed", exc_info=True)
    if added:
        log.info("Backfilled %d external trade(s) into the journal", added)
    return added


def position_bot_map() -> dict[str, str]:
    """Return {str(etoro_position_id): bot_uuid} for journaled trades that carry
    both — lets the History view label each closed trade with the bot that
    opened it.  Trades with no bot (manual / eToro-backfilled) are omitted."""
    with _lock:
        _load_cache_locked()
        out: dict[str, str] = {}
        for r in _cache:
            pid = r.get("etoro_position_id")
            bid = r.get("bot_id")
            if pid is not None and bid:
                out[str(pid)] = bid
        return out


def position_close_meta_map() -> dict[str, dict]:
    """Return {str(etoro_position_id): {"reason", "strategy"}} for the History
    tab's close-method column.  Newest journal row wins per position id."""
    with _lock:
        _load_cache_locked()
        out: dict[str, dict] = {}
        for r in _cache:
            pid = r.get("etoro_position_id")
            if pid is None:
                continue
            out[str(pid)] = {
                "reason": (r.get("reason") or "").strip(),
                "strategy": (r.get("strategy") or "").strip(),
            }
        return out


def total_count() -> int:
    with _lock:
        _load_cache_locked()
        return _total_count


def bot_realized() -> dict:
    """Realized P&L from BOT-opened closed trades only (excludes manual/eToro-
    imported trades, which carry no bot_id).  Returns {'n', 'total_pnl', 'wins'}."""
    with _lock:
        _load_cache_locked()
        rows = [
            r for r in _cache
            if (r.get("bot_id") or "").strip() and not r.get("shadow")
        ]
    total = sum(float(r.get("pnl_dollars") or 0.0) for r in rows)
    wins  = sum(1 for r in rows if r.get("win"))
    return {"n": len(rows), "total_pnl": round(total, 2), "wins": wins}


def bot_realized_since(min_close_date) -> dict:
    """Realized bot P&L from closes on or after min_close_date (display timezone).

    Uses exit_time when present, else ts.  Returns {'n', 'total_pnl', 'wins'}."""
    import timez
    from datetime import datetime as _dt

    with _lock:
        _load_cache_locked()
        rows = [
            r for r in _cache
            if (r.get("bot_id") or "").strip() and not r.get("shadow")
        ]
    total = 0.0
    wins = 0
    n = 0
    for r in rows:
        raw = r.get("exit_time") or r.get("ts")
        if not raw:
            continue
        try:
            dt = _dt.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            continue
        loc = timez.to_local(dt)
        if not loc or loc.date() < min_close_date:
            continue
        pnl = float(r.get("pnl_dollars") or 0.0)
        total += pnl
        n += 1
        if r.get("win"):
            wins += 1
    return {"n": n, "total_pnl": round(total, 2), "wins": wins}


def closed_records(
    limit: Optional[int] = None,
    instrument_id: Optional[int] = None,
    strategy: Optional[str] = None,
    bot_id: Optional[str] = None,
) -> list[dict]:
    """Durable closed-trade records, newest first.

    Backs the P&L tab's closed-trade list so it survives dashboard restarts
    (unlike the in-memory session list).  Optional filters mirror _filtered();
    `limit` caps how many of the most-recent records are returned."""
    rows = _filtered(instrument_id, strategy or None, bot_id or None)
    rows = list(reversed(rows))   # newest first
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows
