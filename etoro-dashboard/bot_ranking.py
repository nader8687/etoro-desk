"""
Evidence-based bot ranking — ADVISORY per-bot "bleeding" flags.

Each bot (strategy + interval + instrument) is judged on its OWN closed-trade
history in the journal — not pooled across every bot that shares a strategy name.
A candlestick bot on 1m can be BLEEDING while the same strategy on 15m is fine.

  FLAG    → recent profit factor below the Settings threshold over enough closed
            trades for THAT bot (strategy + interval + asset).
  UNFLAG  → that bot's rolling window recovers (hysteresis threshold in Settings).

Purely informational — a 🔴 BLEEDING badge on the Bots tab.  Flagged bots keep
trading live; disable or delete them on the card if you agree.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

BOOT_DELAY  = 15.0


def _cfg():
    """Live ranking thresholds — Settings tab → user_settings.json."""
    import user_settings
    return user_settings.ranking_settings()

_PATH = Path(os.environ.get("BLEEDING_BOTS_PATH", "/app/data/bleeding_bots.json"))
# Legacy path — strategy-level flags from an older build; ignored on load.
_LEGACY_PATH = Path(os.environ.get("BLEEDING_STRATS_PATH", "/app/data/bleeding_strategies.json"))

_lock = threading.Lock()
_bleeding: set[str] = set()          # bot UUIDs currently flagged
_last_eval: dict[str, dict] = {}     # bot_id -> latest stats (for UI)
_reviewer_started = False


def _save() -> None:
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(sorted(_bleeding)), encoding="utf-8")
    except Exception:
        log.warning("Could not persist bleeding-bot set", exc_info=True)


def _is_bot_id(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _load() -> None:
    for path in (_PATH, _LEGACY_PATH):
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for item in data:
                s = str(item).strip()
                if _is_bot_id(s):
                    _bleeding.add(s)
        except Exception:
            log.warning("Could not load bleeding flags from %s", path, exc_info=True)


_load()


def _pf_value(pf) -> float:
    if pf == float("inf"):
        return float("inf")
    try:
        return float(pf)
    except (TypeError, ValueError):
        return 1.0


def _stats_from_rows(rows: list) -> dict:
    n = len(rows)
    gross_win  = sum(float(r.get("pnl_dollars") or 0) for r in rows
                     if float(r.get("pnl_dollars") or 0) > 0)
    gross_loss = -sum(float(r.get("pnl_dollars") or 0) for r in rows
                      if float(r.get("pnl_dollars") or 0) < 0)
    if gross_loss <= 0:
        pf = float("inf") if gross_win > 0 else 1.0
    else:
        pf = gross_win / gross_loss
    wins = sum(1 for r in rows if float(r.get("pnl_dollars") or 0) > 0)
    return {
        "n": n,
        "pf": round(pf, 3) if pf != float("inf") else float("inf"),
        "win_rate": round(wins / n, 3) if n else 0.0,
    }


def _trade_rows(
    bot_id: str = "",
    *,
    strategy: str = "",
    interval: str = "",
    instrument_id: int = 0,
    limit: int | None = None,
) -> list:
    if limit is None:
        limit = _cfg().window
    import trade_journal
    bid = (bot_id or "").strip()
    if bid:
        return trade_journal.bot_recent(bid, limit=limit, include_shadow=False)
    return trade_journal.context_recent(
        strategy, interval, instrument_id, limit=limit, include_shadow=False,
    )


def bot_stats(
    bot_id: str = "",
    *,
    strategy: str = "",
    interval: str = "",
    instrument_id: int = 0,
) -> dict:
    """Rolling-window stats for one bot from its own closed trades."""
    return _stats_from_rows(_trade_rows(
        bot_id, strategy=strategy, interval=interval, instrument_id=instrument_id,
    ))


def _bleeding_now(
    bot_id: str,
    stats: dict | None = None,
    *,
    strategy: str = "",
    interval: str = "",
    instrument_id: int = 0,
) -> bool:
    bid = (bot_id or "").strip()
    if not bid:
        return False
    stats = stats if stats is not None else bot_stats(
        bid, strategy=strategy, interval=interval, instrument_id=instrument_id,
    )
    n = int(stats.get("n") or 0)
    pf = _pf_value(stats.get("pf", 1.0))
    with _lock:
        in_set = bid in _bleeding
    cfg = _cfg()
    if n < cfg.min_trades:
        return in_set
    if pf == float("inf"):
        return False
    if in_set:
        return pf < cfg.pf_recover
    return pf < cfg.pf_flag


def is_bleeding(
    bot_id: str = "",
    *,
    strategy: str = "",
    interval: str = "",
    instrument_id: int = 0,
) -> bool:
    return _bleeding_now(
        bot_id, strategy=strategy, interval=interval, instrument_id=instrument_id,
    )


def _short_interval(interval: str) -> str:
    iv = (interval or "").strip().lower()
    mapping = {
        "1 minute": "1m", "5 minutes": "5m", "10 minutes": "10m",
        "15 minutes": "15m", "30 minutes": "30m", "1 hour": "1h",
        "4 hours": "4h", "1 day": "1d", "1 week": "1w",
    }
    return mapping.get(iv, interval or "?")


def card_caption(
    bot_id: str,
    *,
    strategy: str = "",
    interval: str = "",
    instrument_label: str = "",
    instrument_id: int = 0,
) -> str | None:
    """Formatted BLEEDING line for one bot card, or None when not advisory."""
    bid = (bot_id or "").strip()
    if not bid:
        return None
    # Fast path: only flagged bots need a journal scan.  On a 100+ bot fleet this
    # was scanning the full trade journal once per card per Bots-tab refresh.
    with _lock:
        flagged = bid in _bleeding
    if not flagged:
        return None
    stats = bot_stats(
        bid, strategy=strategy, interval=interval, instrument_id=instrument_id,
    )
    if not _bleeding_now(
        bid, stats, strategy=strategy, interval=interval, instrument_id=instrument_id,
    ):
        return None
    pf = stats["pf"]
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    wr = float(stats.get("win_rate") or 0) * 100
    n = int(stats.get("n") or 0)
    tf = _short_interval(interval)
    asset = (instrument_label or "").split("(")[0].strip() or "this asset"
    return (
        f"🔴 **BLEEDING** — pf {pf_s} over last {n} closed {tf} trades on "
        f"{asset} (win {wr:.0f}%); still trading live — turn off if you want it stopped"
    )


def review() -> dict:
    """Advisory evaluation pass over every configured bot."""
    import bot_registry
    import instrument_config

    flagged, cleared = [], []
    for spec in instrument_config.load_specs():
        bot_id = (bot_registry.get(spec.key) or "").strip()
        if not bot_id:
            continue
        stats = bot_stats(
            bot_id,
            strategy=spec.strategy,
            interval=spec.interval,
            instrument_id=spec.instrument_id,
        )
        with _lock:
            _last_eval[bot_id] = stats
            already = bot_id in _bleeding
        cfg = _cfg()
        if stats["n"] < cfg.min_trades:
            continue
        pf = _pf_value(stats["pf"])
        if pf == float("inf"):
            continue
        if not already and pf < cfg.pf_flag:
            with _lock:
                _bleeding.add(bot_id)
            flagged.append((spec.key, bot_id, spec, stats))
        elif already and pf >= cfg.pf_recover:
            with _lock:
                _bleeding.discard(bot_id)
            cleared.append((spec.key, bot_id, spec, stats))
    if flagged or cleared:
        with _lock:
            _save()
        import engine_notify
        for bot_key, _bid, spec, stats in flagged:
            tf = _short_interval(spec.interval)
            msg = (
                f"Ranking: `{bot_key}` ({spec.strategy} · {tf} · {spec.label}) is BLEEDING — "
                f"pf {stats['pf']:.2f} over its last {stats['n']} trades "
                f"(wins {stats['win_rate']*100:.0f}%).  It KEEPS TRADING — disable on "
                f"the Bots tab if you agree."
            )
            log.info(msg)
            engine_notify.push("info", msg)
        for bot_key, _bid, spec, stats in cleared:
            tf = _short_interval(spec.interval)
            msg = (
                f"Ranking: `{bot_key}` ({spec.strategy} · {tf}) recovered — pf "
                f"{stats['pf']:.2f} over its last {stats['n']} trades; bleeding flag cleared."
            )
            log.info(msg)
            engine_notify.push("info", msg)
    return {
        "flagged": [k for k, _, _, _ in flagged],
        "cleared": [k for k, _, _, _ in cleared],
    }


def status() -> dict:
    """{bot_id: {"bleeding": bool, n, pf, win_rate}} — live per-bot from the journal."""
    import bot_registry
    import instrument_config

    out: dict = {}
    for spec in instrument_config.load_specs():
        bot_id = (bot_registry.get(spec.key) or "").strip()
        if not bot_id:
            continue
        stats = bot_stats(
            bot_id,
            strategy=spec.strategy,
            interval=spec.interval,
            instrument_id=spec.instrument_id,
        )
        with _lock:
            _last_eval[bot_id] = stats
        out[bot_id] = {
            "bot_key": spec.key,
            "strategy": spec.strategy,
            "interval": spec.interval,
            "instrument_label": spec.label,
            "bleeding": _bleeding_now(
                bot_id, stats,
                strategy=spec.strategy,
                interval=spec.interval,
                instrument_id=spec.instrument_id,
            ),
            **stats,
        }
    return out


def ensure_reviewer() -> None:
    """Start the periodic advisory review thread (idempotent)."""
    global _reviewer_started
    with _lock:
        if _reviewer_started:
            return
        _reviewer_started = True

    def _loop() -> None:
        time.sleep(BOOT_DELAY)
        while True:
            try:
                review()
            except Exception:
                log.warning("Bot-ranking review failed", exc_info=True)
            time.sleep(_cfg().review_sec)

    threading.Thread(target=_loop, daemon=True, name="bot-ranking").start()
    cfg = _cfg()
    log.info(
        "Bot-ranking reviewer started (per-bot ADVISORY: flag pf<%.2f, clear pf>=%.2f, "
        "min n=%d, window=%d, every %.0fs); %d bot(s) currently flagged bleeding",
        cfg.pf_flag, cfg.pf_recover, cfg.min_trades, cfg.window, cfg.review_sec,
        len(_bleeding),
    )
