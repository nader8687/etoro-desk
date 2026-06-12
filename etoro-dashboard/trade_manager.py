"""
Trade state — survives Streamlit reruns (module-level store).

ONE open position per instrument. Orders are placed on the eToro **demo**
(virtual money) account when a client is provided at open/close.

Entry: BUY → LONG @ ask,  SELL → SHORT @ bid  (+ eToro demo market order)
Exit:  LLM advisory on candle close, stop-loss on eToro + local backup
"""
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import trade_journal

if TYPE_CHECKING:
    from etoro_client import EToroClient

log = logging.getLogger(__name__)

Direction = Literal["LONG", "SHORT"]
STOP_LOSS_MULT = 2.0
# Floor stop distance — spread-only stops are far too tight on crypto.
STOP_LOSS_MIN_PCT = 2.5
# Loss within this many × entry-spread (in $) is spread recovery — never LLM-close.
# FALLBACK ONLY — Settings tab (behavior.spread_recovery_mult) is the truth.
SPREAD_RECOVERY_MULT = 2.0
# Minimum LLM exit confidence required to CUT a real loss (a loss beyond the
# spread-recovery zone).  Below this we ride to the mechanical stop-loss; at or
# above it a confident "the trend has turned" CLOSE is honoured.
# FALLBACK ONLY — Settings tab (behavior.llm_loss_cut_min_conf) is the truth.
LLM_LOSS_CUT_MIN_CONF = 70


def _spread_recovery_mult() -> float:
    try:
        import user_settings
        return float(user_settings.behavior_settings().spread_recovery_mult)
    except Exception:
        return SPREAD_RECOVERY_MULT


def _llm_loss_cut_min_conf() -> int:
    try:
        import user_settings
        return int(user_settings.behavior_settings().llm_loss_cut_min_conf)
    except Exception:
        return LLM_LOSS_CUT_MIN_CONF


def _extract_pid(pos: dict) -> Optional[int]:
    """Extract position ID from a raw eToro position/response dict."""
    for key in ("positionID", "positionId", "PositionID", "position_id", "id"):
        val = pos.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def _parse_etoro_open_date(etoro_pos: dict) -> Optional[datetime]:
    """Parse eToro's real position open timestamp into a tz-aware UTC datetime.

    The `open_date` field is normalised by etoro_client.normalize_position()
    from the various camelCase/PascalCase variants eToro returns.  Handles both
    ISO-8601 strings and Unix epoch timestamps (seconds or milliseconds).
    Returns None when absent or unparseable so callers keep their fallback.
    """
    raw = etoro_pos.get("open_date")
    if raw in (None, "", 0):
        return None

    # Numeric epoch (seconds or milliseconds)
    if isinstance(raw, (int, float)) or (isinstance(raw, str) and raw.strip().isdigit()):
        try:
            ts = float(raw)
            if ts > 1e11:          # too large for seconds → milliseconds
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None

    # ISO-8601 string
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _reconcile_trade_locked(trade: "PaperTrade", etoro_pos: dict) -> None:
    """Fill pending position ID and sync entry_time from eToro's real order time.

    Caller MUST hold `_lock`.  Idempotent: entry_time is synced exactly once
    (guarded by etoro_open_time_synced); the position ID is filled only while
    still missing.  This is how a self-opened trade's optimistic datetime.now()
    entry_time gets replaced by the authoritative eToro open timestamp.
    """
    if trade.etoro_position_id is None:
        pid = _extract_pid(etoro_pos)
        if pid:
            trade.etoro_position_id = pid
            log.info(
                "Resolved pending position ID for instrument %s → %s",
                trade.instrument_id, pid,
            )
    if not trade.etoro_open_time_synced:
        real_open = _parse_etoro_open_date(etoro_pos)
        if real_open is not None:
            trade.entry_time = real_open
            trade.etoro_open_time_synced = True
            log.info(
                "Synced entry_time for instrument %s to eToro open time %s",
                trade.instrument_id, real_open.isoformat(),
            )


def reconcile_from_etoro(bot_id: str, etoro_pos: dict) -> None:
    """Public: reconcile THIS bot's open trade — position ID + entry_time — from
    live eToro portfolio data.  Safe to call every tick (short-circuits once
    synced).  Records the position→bot owner once the id is known."""
    pid = None
    with _lock:
        trade = _open.get(bot_id)
        if trade is not None:
            _reconcile_trade_locked(trade, etoro_pos)
            pid = trade.etoro_position_id
    if pid is not None and bot_id:
        _set_owner(pid, bot_id)


def compute_stop_loss_price(
    direction: Direction,
    entry_price: float,
    spread: float,
    min_pct: Optional[float] = None,
) -> float:
    """Stop price = entry ∓ max(2× spread, min_pct% of entry).

    `min_pct` comes from the strategy's exit profile (exit_profiles.py) so
    tight-target strategies (mean-revert/arb) risk proportionally less than
    trend strategies.  Falls back to the global STOP_LOSS_MIN_PCT.
    """
    pct = STOP_LOSS_MIN_PCT if min_pct is None else float(min_pct)
    dist = max(STOP_LOSS_MULT * spread, entry_price * pct / 100)
    if direction == "LONG":
        return entry_price - dist
    return entry_price + dist


@dataclass
class PaperTrade:
    instrument_id: int
    instrument_label: str
    direction: Direction
    entry_price: float
    entry_spread: float
    stop_loss_price: float
    entry_time: datetime
    signal: str
    confidence: int
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    peak_pnl: float = 0.0
    etoro_position_id: Optional[int] = None
    trade_amount: float = 0.0
    on_etoro: bool = False
    opened_by_bot: bool = False
    bot_id: str = ""   # UUID from bot_registry of the bot that opened this trade
    # True once entry_time has been replaced by eToro's real open timestamp.
    # Self-opened trades start with an optimistic datetime.now() and get
    # reconciled to the authoritative eToro time on the next portfolio refresh.
    etoro_open_time_synced: bool = False
    # Entry-decision context — carried so the trade journal can learn which
    # setups win/lose.  Populated by open_trade() from the signal result.
    strategy: str = ""          # strategy key that generated the entry
    exec_risk: str = ""         # execution-risk tier at entry (LOW/MEDIUM/HIGH)
    net_edge_pct: float = 0.0   # modelled net edge at entry
    # Extra entry context — carried for the trade journal / analytics.  All
    # optional with safe defaults so older call sites keep working unchanged.
    interval: str = ""          # timeframe label, e.g. "15 Minutes"
    entry_reason: str = ""      # human-readable signal reasoning at entry
    regime: str = ""            # market regime label at entry, e.g. "up/high"
    atr_pct_entry: float = 0.0  # ATR% at entry (regime/vol context)
    stop_pct_entry: float = 0.0 # stop distance actually used (% of entry)
    confidence_calibrated: int = 0  # win-rate-shrunk confidence at entry (analytics)
    # Legacy journal field — older builds journaled virtual bench trades as
    # shadow=True; new entries are always live on eToro.
    shadow: bool = False
    # Set once the trade has been underwater (never meaningfully green) longer
    # than recovery_hold_mult × the strategy's avg hold — close on next ≥$0 tick.
    recovery_armed: bool = False
    # Breakeven floor applied: when the recovery exit caught the trade back at
    # ≥$0, the stop was RAISED TO ENTRY instead of closing — downside locked at
    # ~no loss while TP and the ATR chandelier trail keep the upside open.
    breakeven_set: bool = False
    # ── Chandelier ATR trailing state (golden rule 2xATR) ─────────────────────
    # peak_price: best favourable price seen since entry (LONG: highest bid;
    #             SHORT: lowest ask) — the chandelier's anchor.
    # trail_stop_price: the RATCHETED stop level = peak_price ∓ k·ATR.  Moves
    #             only in the trade's favour, never widens (the ratchet rule).
    #             Initialised to the entry 2xATR hard stop, so trail and hard
    #             stop start as the same line and the trail only tightens.
    peak_price: float = 0.0
    trail_stop_price: float = 0.0


@dataclass
class ClosedTrade:
    instrument_id: int
    instrument_label: str
    direction: Direction
    entry_price: float
    entry_spread: float
    entry_time: datetime
    signal: str
    confidence: int
    exit_price: float
    exit_time: datetime
    profit: float
    reason: str
    trade_id: str = ""
    stop_loss_price: float = 0.0
    etoro_position_id: Optional[int] = None
    bot_id: str = ""   # UUID of the bot that opened this trade (from bot_registry)
    # LLM exit signal — populated only when reason == "llm"
    llm_reasoning: Optional[str] = None
    llm_observations: Optional[str] = None
    shadow: bool = False   # legacy virtual trade — excluded from money stats
    # ACTUAL dollar P&L (units × per-unit price move).  `profit` above is the
    # raw PRICE move per unit — on BTC a "−71.2" profit is −71.2 price points,
    # i.e. only ≈ −$1 on a $1k position.  Display code must use THIS field.
    pnl_dollars: float = 0.0


# ── Trade stores keyed by BOT (UUID) ──────────────────────────────────────────
# Each bot independently holds at most one open position, so trades are keyed by
# the bot's UUID — NOT by instrument_id.  This lets many bots trade the same
# instrument at once (e.g. an XRP supertrend bot and an XRP RSI bot both LONG).
_lock = threading.Lock()
# Serialises the open critical-section (snapshot → submit order → resolve new
# position id) across ALL bots.  Many bots fire on the same candle close; opening
# one at a time keeps the before/after portfolio diff unambiguous and avoids the
# eToro 429 burst that concurrent opens used to trigger.  Opens are infrequent and
# each takes ~3s, so the staggering is negligible.
_open_serialize_lock = threading.Lock()
_open: dict[str, PaperTrade] = {}      # bot_id (UUID) → live trade
_closed: list[ClosedTrade] = []
# Bots whose close is in-flight (eToro HTTP roundtrip in progress).  Treated as
# "still open" for has_open / get_open so no second order can fire while we await
# the close response.  See _finalize_close.
_closing: dict[str, PaperTrade] = {}   # bot_id → trade being closed
# Bots whose open order is in-flight.  Reserved by open_trade() before submitting
# the market order so a bot can't double-fire while its own response is pending.
_opening: set[str] = set()             # bot_ids with an open in-flight
_last_error: Optional[str] = None

# ── Re-adoption guard ──────────────────────────────────────────────────────────
# Position ids we closed recently.  The background positions cache (and eToro's
# own API) can keep listing a just-closed position for several seconds; without
# this guard the owning bot instantly re-adopts its own dead position and
# "closes" it again, journaling phantom duplicate P&L records (observed: the
# same position closed 10-14×).  TTL instead of permanent: if the pid is STILL
# in the portfolio after the TTL, the earlier close was a false vanish and the
# bot may legitimately re-adopt and manage it again.
_RECENT_CLOSE_TTL_SEC = 180.0
_recently_closed_pids: dict[int, float] = {}   # position_id → monotonic ts


def _mark_recently_closed(position_id: Optional[int]) -> None:
    if position_id:
        now = time.monotonic()
        _recently_closed_pids[int(position_id)] = now
        # opportunistic pruning — the dict stays tiny
        for pid, ts in list(_recently_closed_pids.items()):
            if now - ts > _RECENT_CLOSE_TTL_SEC:
                _recently_closed_pids.pop(pid, None)


def _is_recently_closed(position_id) -> bool:
    if not position_id:
        return False
    ts = _recently_closed_pids.get(int(position_id))
    return ts is not None and (time.monotonic() - ts) <= _RECENT_CLOSE_TTL_SEC

# ── Persisted position→bot ownership ──────────────────────────────────────────
# Maps eToro position_id → owning bot UUID, persisted to the data volume so that
# after a restart each bot re-adopts ITS OWN position (instead of a sibling's),
# and the Portfolio/History "Bot" column attributes correctly across restarts.
_OWNER_PATH = Path(os.environ.get("POSITION_OWNERS_PATH", "/app/data/position_owners.json"))
_position_owner: dict[str, str] = {}   # str(position_id) → bot_id (UUID)
# Dedicated re-entrant lock for the owner map.  Many engine threads (plus the
# backfill loop) mutate/persist it concurrently; without serialisation the
# non-atomic file writes interleave and corrupt position_owners.json, which then
# fails to load and silently wipes every attribution on the next reload.  Kept
# separate from `_lock` so it can be held while `_lock` is already held.
_owner_file_lock = threading.RLock()


def _save_owners() -> None:
    """Atomically persist the owner map (temp file + rename) under the file lock,
    so concurrent writers can never leave a half-written / corrupted JSON file."""
    with _owner_file_lock:
        try:
            _OWNER_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _OWNER_PATH.with_name(_OWNER_PATH.name + ".tmp")
            tmp.write_text(json.dumps(_position_owner), encoding="utf-8")
            os.replace(tmp, _OWNER_PATH)   # atomic on the same filesystem
        except Exception:
            log.warning("Could not persist position-owner map", exc_info=True)


def _load_owners() -> None:
    with _owner_file_lock:
        try:
            if not _OWNER_PATH.exists():
                return
            text = _OWNER_PATH.read_text(encoding="utf-8").strip()
            if not text:
                return
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Salvage a file corrupted by a previous non-atomic write: decode
                # just the first valid JSON object and ignore trailing garbage,
                # so we never wipe every attribution because of trailing bytes.
                data, _end = json.JSONDecoder().raw_decode(text)
                log.warning("Recovered position-owner map from a corrupted file")
            if isinstance(data, dict):
                _position_owner.update({str(k): str(v) for k, v in data.items()})
                _save_owners()   # rewrite cleanly so the corruption can't recur
        except Exception as exc:
            log.warning("Could not load position-owner map: %s", exc)


def _set_owner(position_id, bot_id: str) -> None:
    if position_id is None or not bot_id:
        return
    with _owner_file_lock:
        if _position_owner.get(str(position_id)) != bot_id:
            _position_owner[str(position_id)] = bot_id
            _save_owners()


def _clear_owner(position_id) -> None:
    if position_id is None:
        return
    with _owner_file_lock:
        if str(position_id) in _position_owner:
            _position_owner.pop(str(position_id), None)
            _save_owners()


def owner_of_position(position_id) -> Optional[str]:
    """Bot UUID that owns this eToro position, or None (manual / unknown)."""
    if position_id is None:
        return None
    return _position_owner.get(str(position_id))


# ── Open-lineage map (partial-close attribution) ─────────────────────────────
# eToro partial closes (cash freeing) create NEW position_ids for each shaved
# slice, all sharing the parent's open time + open rate.  The owner map only
# knew the live position id, so History labelled every trim slice "Manual".
# This map keys (instrument, direction, open-time, open-rate) → bot UUID so ALL
# slices from the same opening inherit the correct bot.
_LINEAGE_PATH = Path(os.environ.get(
    "OPEN_LINEAGE_PATH",
    str(_OWNER_PATH.parent / "open_lineage.json"),
))
_open_lineage: dict[str, str] = {}   # lineage_key → bot_uuid


def _normalize_open_dt(value) -> Optional[str]:
    """UTC open timestamp normalized to second precision (for stable matching)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def lineage_key(
    instrument_id: int,
    direction: str,
    open_dt,
    open_rate: float,
) -> Optional[str]:
    ts = _normalize_open_dt(open_dt)
    if not ts or not instrument_id or not direction:
        return None
    try:
        rate = round(float(open_rate), 5)
    except (TypeError, ValueError):
        return None
    return f"{int(instrument_id)}|{str(direction).upper()}|{ts}|{rate}"


def lineage_key_from_history(row: dict) -> Optional[str]:
    direction = row.get("direction")
    if not direction:
        ib = row.get("isBuy")
        if ib is not None:
            direction = "LONG" if ib else "SHORT"
    iid = row.get("instrumentId") or row.get("instrument_id")
    od = row.get("openTimestamp") or row.get("open_time") or row.get("open_date")
    rate = row.get("openRate") or row.get("open_rate") or row.get("entry_price")
    if iid is None or not direction or rate is None:
        return None
    try:
        return lineage_key(int(iid), direction, od, float(rate))
    except (TypeError, ValueError):
        return None


def _save_lineage() -> None:
    with _owner_file_lock:
        try:
            _LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _LINEAGE_PATH.with_name(_LINEAGE_PATH.name + ".tmp")
            tmp.write_text(json.dumps(_open_lineage), encoding="utf-8")
            os.replace(tmp, _LINEAGE_PATH)
        except Exception:
            log.warning("Could not persist open-lineage map", exc_info=True)


def _load_lineage() -> None:
    with _owner_file_lock:
        try:
            if not _LINEAGE_PATH.exists():
                return
            data = json.loads(_LINEAGE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _open_lineage.update({str(k): str(v) for k, v in data.items()})
        except Exception as exc:
            log.warning("Could not load open-lineage map: %s", exc)


def register_open_lineage(
    instrument_id: int,
    direction: str,
    open_dt,
    open_rate: float,
    bot_id: str,
) -> None:
    """Record which bot opened a position line (survives partial-close child ids)."""
    if not bot_id:
        return
    key = lineage_key(instrument_id, direction, open_dt, open_rate)
    if not key:
        return
    with _owner_file_lock:
        if _open_lineage.get(key) != bot_id:
            _open_lineage[key] = bot_id
            _save_lineage()


def owner_by_lineage_key(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return _open_lineage.get(key)


def resolve_history_owner(row: dict) -> Optional[str]:
    """Best-effort bot UUID for a closed eToro history row."""
    pid = row.get("positionId") or row.get("position_id") or row.get("positionID")
    if pid is not None:
        uuid = owner_of_position(pid)
        if uuid:
            return uuid
        try:
            import trade_journal
            uuid = trade_journal.position_bot_map().get(str(pid))
            if uuid:
                return uuid
        except Exception:
            pass
    return owner_by_lineage_key(lineage_key_from_history(row))


def claim_trim_history_slices(
    client: "EToroClient",
    trade: "PaperTrade",
    *,
    is_demo: bool = True,
) -> int:
    """After a partial trim, tag any new closed history slices with this bot.

    eToro issues a fresh position_id per shaved slice; this walks recent
    history for matching open-line rows and persists ownership immediately."""
    if not trade.bot_id:
        return 0
    from datetime import timedelta

    register_open_lineage(
        trade.instrument_id, trade.direction, trade.entry_time, trade.entry_price,
        trade.bot_id,
    )
    lk = lineage_key(
        trade.instrument_id, trade.direction, trade.entry_time, trade.entry_price,
    )
    if not lk:
        return 0
    tagged = 0
    try:
        md = (datetime.now(tz=timezone.utc) - timedelta(hours=12)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for row in client.get_all_trade_history(md, demo=is_demo):
            if lineage_key_from_history(row) != lk:
                continue
            pid = row.get("positionId") or row.get("position_id")
            if pid is None:
                continue
            if owner_of_position(pid) != trade.bot_id:
                _set_owner(pid, trade.bot_id)
                tagged += 1
    except Exception as exc:
        log.debug("claim_trim_history_slices failed: %s", exc)
    return tagged


def propagate_cluster_owners(rows: list[dict]) -> int:
    """Inherit bot ownership across history rows that share the same open line.

    Partial cash-freeing trims create many closed rows with distinct position
    ids but identical open time/rate.  If ANY slice in a cluster is attributed,
    persist that owner onto every sibling pid + the lineage key.
    Returns the number of position ids newly tagged."""
    from collections import defaultdict

    clusters: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        lk = lineage_key_from_history(r)
        if lk:
            clusters[lk].append(r)

    tagged = 0
    for lk, members in clusters.items():
        owner = owner_by_lineage_key(lk)
        if not owner:
            for m in members:
                pid = m.get("positionId") or m.get("position_id")
                owner = resolve_history_owner(m)
                if owner:
                    break
        if not owner:
            continue
        with _owner_file_lock:
            if _open_lineage.get(lk) != owner:
                _open_lineage[lk] = owner
                _save_lineage()
        for m in members:
            pid = m.get("positionId") or m.get("position_id")
            if pid is None:
                continue
            if owner_of_position(pid) != owner:
                _set_owner(pid, owner)
                tagged += 1
    return tagged


_load_owners()
_load_lineage()


def get_last_error() -> Optional[str]:
    with _lock:
        return _last_error


def _set_error(msg: Optional[str]) -> None:
    with _lock:
        global _last_error
        _last_error = msg


def has_open(bot_id: str) -> bool:
    """True when THIS bot has a live trade OR a close/open in-flight."""
    with _lock:
        return bot_id in _open or bot_id in _closing or bot_id in _opening


def signals_paused(bot_id: str) -> bool:
    return has_open(bot_id)


def get_open(bot_id: str) -> Optional[PaperTrade]:
    """Return THIS bot's open trade — falls through to a closing trade so callers
    monitoring P&L during the close window still see the position."""
    with _lock:
        return _open.get(bot_id) or _closing.get(bot_id)


def get_open_for_instrument(instrument_id: int) -> list[PaperTrade]:
    """All live/closing trades on an instrument (across all bots) — for UI views
    that are instrument-centric (chart, Portfolio)."""
    with _lock:
        seen: dict[str, PaperTrade] = {**_closing, **_open}
        return [t for t in seen.values() if t.instrument_id == instrument_id]


def find_open_by_position_id(position_id) -> Optional[PaperTrade]:
    """Locate a live/closing trade by its eToro position id (manual close path)."""
    if position_id is None:
        return None
    with _lock:
        for t in {**_closing, **_open}.values():
            if t.etoro_position_id is not None and str(t.etoro_position_id) == str(position_id):
                return t
    return None


def held_position_ids() -> set:
    """eToro position ids currently tracked (open or closing) across ALL bots —
    used by adoption so a bot never claims a position another bot already holds."""
    with _lock:
        return {
            t.etoro_position_id
            for t in {**_closing, **_open}.values()
            if t.etoro_position_id is not None
        }


def is_bot_owned_position(position_id) -> bool:
    """True when an open eToro position belongs to a bot (persisted owner map
    or in-memory tracked trade).  Survives restarts before re-adoption."""
    if position_id is None:
        return False
    if owner_of_position(position_id):
        return True
    with _lock:
        for t in {**_open, **_closing}.values():
            if (
                t.etoro_position_id is not None
                and str(t.etoro_position_id) == str(position_id)
                and (t.bot_id or "").strip()
            ):
                return True
    return False


def bot_owned_positions(positions: list[dict]) -> list[dict]:
    """Filter a positions-cache snapshot to bot-attributed rows only."""
    return [p for p in positions if is_bot_owned_position(p.get("position_id"))]


def get_all_open() -> list["PaperTrade"]:
    """Snapshot of every bot's currently-open trade — for portfolio-level logic
    such as cash freeing (ranking the weakest positions to trim)."""
    with _lock:
        return list(_open.values())


def record_trim(position_id, fraction_closed: float) -> None:
    """After a PARTIAL close, scale the local trade's size + peak P&L down by the
    closed fraction so trailing-stop math (which reads trade_amount and peak_pnl)
    stays consistent with the now-smaller position.

    Also RE-AFFIRMS the position's bot ownership: a position trimmed by cash-
    freeing can later close outside the owning bot's normal close path (so it's
    never journaled with a bot_id).  Persisting the owner here keeps it attributed
    to its bot in the Portfolio/History views instead of showing 'Manual'."""
    if not (0.0 < fraction_closed < 1.0):
        return
    keep = 1.0 - fraction_closed
    owner_bot = None
    with _lock:
        for t in _open.values():
            if t.etoro_position_id is not None and str(t.etoro_position_id) == str(position_id):
                t.trade_amount *= keep
                t.peak_pnl     *= keep
                owner_bot = t.bot_id
                break
    if owner_bot:                       # outside _lock — _set_owner does file I/O
        _set_owner(position_id, owner_bot)
        with _lock:
            for t in _open.values():
                if t.etoro_position_id is not None and str(t.etoro_position_id) == str(position_id):
                    register_open_lineage(
                        t.instrument_id, t.direction, t.entry_time, t.entry_price, owner_bot,
                    )
                    break


def backfill_all_owners(
    client: "EToroClient",
    *,
    demo: bool = True,
    window_sec: float = 180.0,
) -> dict[str, str]:
    """Recover bot ownership for eToro positions/trades that have no recorded owner.

    Covers BOTH still-open positions (Portfolio) and closed trades (History) in a
    single pass, so each entry signal is consumed exactly once across the whole
    set.  Matches each unowned position to the bot that logged a same-direction
    entry signal at the same moment (open time ≈ signal.ts) — recovering ownership
    lost before per-bot id tracking was reliable.

    Matching rules (conservative — avoids mislabelling):
      • same instrument_id and direction (BUY→LONG, SELL→SHORT)
      • |open time − signal.ts| ≤ window_sec, nearest wins
      • each signal is consumed once (1:1), so N positions opened together map to
        the N distinct bots that fired at that candle
    Positions with no qualifying signal (manual trades, instruments no bot trades)
    are left as-is (Manual).  Owners are written to the persisted owner map, which
    both the Portfolio and History views consult.

    Returns {str(position_id): bot_uuid} for the newly-attributed positions.
    """
    import signal_log

    def _parse(s) -> Optional[datetime]:
        if not s:
            return None
        try:
            return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except Exception:
            return None

    def _norm(raw: dict) -> Optional[tuple]:
        """(position_id, instrument_id, direction, open_dt, open_rate) from either
        a normalized open position or a raw eToro closed-trade dict."""
        pid = raw.get("position_id")
        if pid is None:
            pid = raw.get("positionId") or raw.get("positionID")
        iid = raw.get("instrument_id")
        if iid is None:
            iid = raw.get("instrumentId") or raw.get("instrumentID")
        direction = raw.get("direction")
        if not direction:
            ib = raw.get("isBuy")
            if ib is None:
                ib = raw.get("is_buy")
            if ib is not None:
                direction = "LONG" if ib else "SHORT"
        od = _parse(raw.get("open_date") or raw.get("openTimestamp")
                    or raw.get("openDateTime") or raw.get("open_time"))
        rate = raw.get("open_rate") or raw.get("openRate") or raw.get("entry_price")
        if pid is None or iid is None or not direction or od is None or rate is None:
            return None
        try:
            pid = int(pid)
            iid = int(iid)
            rate = float(rate)
        except (TypeError, ValueError):
            return None
        return (pid, iid, direction, od, rate)

    # ── Gather every position we know of (open + closed) ──────────────────────
    candidates: list[tuple] = []
    try:
        raw_pf = client.get_portfolio(demo=demo)
        for p in client._dig_positions(raw_pf):
            n = _norm(client.normalize_position(p))
            if n:
                candidates.append(n)
    except Exception:
        log.warning("Backfill: portfolio fetch failed", exc_info=True)
    try:
        hist = client.get_all_trade_history("2026-01-01T00:00:00", demo=demo)
        for h in hist:
            n = _norm(h)
            if n:
                candidates.append(n)
    except Exception:
        log.warning("Backfill: trade-history fetch failed", exc_info=True)

    # ── Signal-log entry events ───────────────────────────────────────────────
    events: list[list] = []   # [ts, iid, direction, bot_uuid, used]
    for r in signal_log.load(signal_type="entry", limit=50000):
        sig = (r.get("signal") or "").upper()
        if sig not in ("BUY", "SELL"):
            continue
        bid = r.get("bot_id")
        ts = _parse(r.get("ts"))
        if not bid or ts is None:
            continue
        try:
            ev_iid = int(r.get("instrument_id"))
        except (TypeError, ValueError):
            continue
        events.append([ts, ev_iid, "LONG" if sig == "BUY" else "SHORT", bid, False])

    # ── Match (earliest first) ────────────────────────────────────────────────
    # Partial cash-freeing creates many closed rows per open line — inherit
    # ownership from lineage / cluster siblings before consuming entry signals.
    assigned: dict[str, str] = {}
    seen_pids: set = set()
    cluster_owner: dict[str, str] = {}
    for pid, iid, direction, od, rate in candidates:
        if owner_of_position(pid):
            lk = lineage_key(iid, direction, od, rate)
            if lk:
                cluster_owner.setdefault(lk, owner_of_position(pid))  # type: ignore[arg-type]

    for pid, iid, direction, od, rate in sorted(candidates, key=lambda c: c[3]):
        if pid in seen_pids:
            continue
        seen_pids.add(pid)
        if owner_of_position(pid) is not None:
            continue
        lk = lineage_key(iid, direction, od, rate)
        inherited = (owner_by_lineage_key(lk) if lk else None) or (
            cluster_owner.get(lk) if lk else None
        )
        if inherited:
            _set_owner(pid, inherited)
            assigned[str(pid)] = inherited
            if lk:
                cluster_owner.setdefault(lk, inherited)
            continue
        best = None
        best_diff = window_sec + 1
        for e in events:
            if e[4] or e[1] != iid or e[2] != direction:
                continue
            diff = abs((od - e[0]).total_seconds())
            if diff <= window_sec and diff < best_diff:
                best, best_diff = e, diff
        if best is not None:
            best[4] = True
            _set_owner(pid, best[3])
            assigned[str(pid)] = best[3]
            register_open_lineage(iid, direction, od, rate, best[3])
            if lk:
                cluster_owner.setdefault(lk, best[3])
    if assigned:
        log.info("Backfilled ownership for %d position(s)/trade(s) from signal log", len(assigned))
    return assigned


def _claim_new_position_id(
    client: "EToroClient",
    instrument_id: int,
    is_buy: bool,
    before_ids: set,
    bot_id: str,
    retries: int = 10,
    delay: float = 1.5,
) -> Optional[int]:
    """Resolve THIS bot's freshly-opened position id by diffing the portfolio.

    eToro's by-amount open response carries no position id, and many positions
    can exist on one instrument/direction at once, so we cannot match by
    (instrument, direction) alone — every bot would latch onto the first one.

    Instead we look for a position id that:
      • appeared AFTER this order (not in `before_ids`), and
      • is not already held by another bot's live trade, and
      • is not already owned in the persisted map by a different bot.

    The pick+claim is atomic under `_lock`, so two bots opening the same
    instrument/direction concurrently each grab a distinct new position id.
    Retries with backoff to absorb eToro's portfolio-propagation lag.
    """
    for attempt in range(retries):
        try:
            current = client.position_ids_for_instrument(instrument_id, is_buy)
        except Exception:
            current = set()
        new_ids = sorted(pid for pid in current if pid not in before_ids)
        if new_ids:
            with _lock:
                held = {
                    t.etoro_position_id
                    for t in {**_closing, **_open}.values()
                    if t.etoro_position_id is not None
                }
                for pid in new_ids:
                    if pid in held:
                        continue
                    owner = _position_owner.get(str(pid))
                    if owner is not None and owner != bot_id:
                        continue
                    # Claim it now so concurrent siblings skip this pid.
                    _position_owner[str(pid)] = bot_id
                    _save_owners()
                    return pid
        time.sleep(delay * (attempt + 1))
    return None


def get_bot_position_ids() -> set[int]:
    """eToro position IDs opened by EtoroDesk auto-trade this session."""
    with _lock:
        return {
            t.etoro_position_id
            for t in _open.values()
            if t.opened_by_bot and t.etoro_position_id is not None
        }


def position_opened_by_bot(position_id: int | None) -> bool:
    if position_id is None:
        return False
    with _lock:
        return any(
            t.opened_by_bot and t.etoro_position_id == position_id
            for t in _open.values()
        )


def adopt_etoro_position(
    instrument_id: int,
    instrument_label: str,
    etoro_pos: dict,
    ask: float,
    bid: float,
    bot_id: str = "",
    strategy: str = "",
) -> Optional[PaperTrade]:
    """Re-hydrate local trade state from an existing eToro demo position (e.g. after restart).

    Also resolves a pending position ID: when open_trade registers a trade without
    an etoro_position_id (eToro processing lag), this call fills it in from the
    live portfolio data so stop-loss and manual close can operate normally.
    """
    # A position we closed moments ago can linger in the (stale) positions
    # cache — adopting it back would create phantom duplicate closes.
    if _is_recently_closed(etoro_pos.get("position_id")):
        return None

    with _lock:
        # Existing trade for THIS bot?  Look in _open then _closing (close HTTP in
        # flight) so we never create a duplicate for the same bot.
        existing = _open.get(bot_id) or _closing.get(bot_id)
        if bot_id in _opening and existing is None:
            # Open call is mid-flight — its own code path will register the trade.
            return None
        if existing is not None:
            # Fill in missing position ID and sync entry_time from eToro's real
            # open timestamp (idempotent — happens at most once each).
            _reconcile_trade_locked(existing, etoro_pos)
            if strategy and not (existing.strategy or "").strip():
                existing.strategy = strategy
            _ex_pid = existing.etoro_position_id
            if _ex_pid is not None and bot_id:
                _set_owner(_ex_pid, bot_id)
            return existing

    raw_dir = (etoro_pos.get("direction") or "").upper()
    if raw_dir in ("LONG", "SHORT"):
        direction: Direction = raw_dir  # type: ignore[assignment]
    elif etoro_pos.get("is_buy") is False:
        direction = "SHORT"
    else:
        direction = "LONG"

    entry_price = etoro_pos.get("open_rate")
    if not entry_price:
        entry_price = ask if direction == "LONG" else bid

    spread = ask - bid
    if spread <= 0:
        spread = 0.0001

    stop_loss = etoro_pos.get("stop_loss")
    if stop_loss is None:
        import exit_profiles
        stop_loss = compute_stop_loss_price(
            direction, float(entry_price), spread,
            min_pct=exit_profiles.stop_loss_min_pct(strategy, instrument_label),
        )

    # Use the actual eToro open timestamp so the chart entry-arrow lands on the
    # correct candle.  Fall back to now() only when eToro omits the field.
    real_open = _parse_etoro_open_date(etoro_pos)
    entry_time = real_open if real_open is not None else datetime.now(tz=timezone.utc)

    trade = PaperTrade(
        instrument_id=instrument_id,
        instrument_label=instrument_label,
        direction=direction,
        entry_price=float(entry_price),
        entry_spread=spread,
        stop_loss_price=float(stop_loss),
        entry_time=entry_time,
        signal="ADOPTED",
        confidence=0,
        etoro_position_id=etoro_pos.get("position_id"),
        trade_amount=float(etoro_pos.get("amount") or 0),
        on_etoro=True,
        bot_id=bot_id,
        strategy=strategy or "",   # so the journal attributes adopted trades correctly
        etoro_open_time_synced=real_open is not None,
        peak_price=float(entry_price),
        trail_stop_price=float(stop_loss),
    )
    with _lock:
        _open[bot_id] = trade
    _set_owner(trade.etoro_position_id, bot_id)
    register_open_lineage(
        instrument_id, direction, entry_time, trade.entry_price, bot_id,
    )
    log.info(
        "Adopted eToro %s position #%s for instrument %s @ %.5f (bot=%s)",
        direction, trade.etoro_position_id, instrument_id, trade.entry_price, bot_id,
    )
    return trade


def get_closed(instrument_id: Optional[int] = None) -> list[ClosedTrade]:
    with _lock:
        if instrument_id is None:
            return list(_closed)
        return [t for t in _closed if t.instrument_id == instrument_id]


def total_realised_pnl(instrument_id: Optional[int] = None) -> float:
    return sum(t.profit for t in get_closed(instrument_id))


def unrealised_pnl(trade: PaperTrade, ask: float, bid: float) -> float:
    """Unrealised P&L per price unit (bid/ask vs entry)."""
    if trade.direction == "LONG":
        return bid - trade.entry_price
    return trade.entry_price - ask


def _amount_and_units(
    entry_price: float,
    trade: Optional[PaperTrade] = None,
    etoro_pos: Optional[dict] = None,
) -> tuple[float, float]:
    amount = float((trade.trade_amount if trade else 0) or (etoro_pos or {}).get("amount") or 0)
    units = float((etoro_pos or {}).get("units") or 0)
    if not units and amount and entry_price:
        units = amount / entry_price
    return amount, units


def dollar_unrealised_pnl(
    direction: Direction,
    entry_price: float,
    ask: float,
    bid: float,
    *,
    trade: Optional[PaperTrade] = None,
    etoro_pos: Optional[dict] = None,
) -> tuple[float, float, float, float]:
    """
    Live mark-to-market from ticks.
    Returns (current_price, pnl_dollars, pnl_pct, pnl_per_unit).
    """
    amount, units = _amount_and_units(entry_price, trade=trade, etoro_pos=etoro_pos)
    if direction == "LONG":
        current = bid
        pnl_dollars = (units * bid - amount) if units and amount else 0.0
        pnl_unit = bid - entry_price
    else:
        current = ask
        pnl_dollars = (amount - units * ask) if units and amount else 0.0
        pnl_unit = entry_price - ask
    pnl_pct = (pnl_dollars / amount * 100) if amount else 0.0
    return current, pnl_dollars, pnl_pct, pnl_unit


def minutes_open(trade: PaperTrade) -> float:
    return (datetime.now(tz=timezone.utc) - trade.entry_time).total_seconds() / 60


def update_peak_pnl(
    trade: PaperTrade,
    ask: float,
    bid: float,
    etoro_pos: Optional[dict] = None,
) -> None:
    _, pnl_dollars, _, _ = dollar_unrealised_pnl(
        trade.direction, trade.entry_price, ask, bid, trade=trade, etoro_pos=etoro_pos,
    )
    with _lock:
        # _open is keyed by bot UUID (not instrument_id) since the multi-bot
        # refactor.  Match on bot_id and confirm it's still THIS trade so a
        # closing/replaced trade's peak isn't clobbered.
        cur = _open.get(trade.bot_id)
        if cur is None or cur is not trade:
            return
        if pnl_dollars > cur.peak_pnl:
            cur.peak_pnl = pnl_dollars
        # Chandelier anchor: best FAVOURABLE price seen since entry — the
        # exit-side quote (LONG closes at bid, SHORT at ask).
        if cur.direction == "LONG":
            fav = float(bid or 0.0)
            if fav > 0 and fav > (cur.peak_price or cur.entry_price):
                cur.peak_price = fav
        else:
            fav = float(ask or 0.0)
            if fav > 0 and (cur.peak_price <= 0 or fav < cur.peak_price):
                cur.peak_price = fav


def distance_to_stop(trade: PaperTrade, ask: float, bid: float) -> float:
    if trade.direction == "LONG":
        return bid - trade.stop_loss_price
    return trade.stop_loss_price - ask


def position_context(trade: PaperTrade, ask: float, bid: float) -> dict:
    return build_exit_position_context(ask, bid, trade=trade)


def build_exit_position_context(
    ask: float,
    bid: float,
    *,
    trade: Optional[PaperTrade] = None,
    etoro_pos: Optional[dict] = None,
) -> dict:
    """Full live position snapshot for the exit LLM (tick-based $ P&L)."""
    if trade:
        direction: Direction = trade.direction
        entry = trade.entry_price
        spread = trade.entry_spread
        stop = trade.stop_loss_price
        mins = minutes_open(trade)
        peak = trade.peak_pnl
    elif etoro_pos:
        raw_dir = (etoro_pos.get("direction") or "").upper()
        if raw_dir in ("LONG", "SHORT"):
            direction = raw_dir  # type: ignore[assignment]
        elif etoro_pos.get("is_buy") is False:
            direction = "SHORT"
        else:
            direction = "LONG"
        entry = float(etoro_pos.get("open_rate") or 0)
        spread = max(ask - bid, 0.0001)
        stop = float(etoro_pos.get("stop_loss") or 0)
        mins = 0.0
        peak = 0.0
    else:
        return {}

    current, pnl_dollars, pnl_pct, pnl_unit = dollar_unrealised_pnl(
        direction, entry, ask, bid, trade=trade, etoro_pos=etoro_pos,
    )
    amount, units = _amount_and_units(entry, trade=trade, etoro_pos=etoro_pos)
    spread_cost = units * spread if units else 0.0

    return {
        "direction":           direction,
        "entry_price":         entry,
        "current_price":       current,
        "unrealised_pnl":      pnl_dollars,
        "unrealised_pnl_unit": pnl_unit,
        "pnl_pct":             pnl_pct,
        "amount_invested":     amount,
        "units":               units,
        "entry_spread":        spread,
        "spread_cost":         spread_cost,
        "stop_loss_price":     stop,
        "minutes_open":        mins,
        "peak_pnl":            peak,
        "in_profit":           pnl_dollars > spread_cost,
        "spread_cost":         spread_cost,
        "spread_recovery_limit": _spread_recovery_mult() * spread_cost,
        "in_spread_recovery_zone": (
            pnl_dollars <= spread_cost
            and pnl_dollars > -(_spread_recovery_mult() * spread_cost)
        ),
    }


def llm_close_veto_reason(
    trade: PaperTrade,
    ask: float,
    bid: float,
    *,
    exit_result: Optional[dict] = None,
) -> str:
    """Human-readable reason when an LLM CLOSE is blocked; empty if allowed."""
    _, pnl_dollars, _, _ = dollar_unrealised_pnl(
        trade.direction, trade.entry_price, ask, bid, trade=trade,
    )
    _, units = _amount_and_units(trade.entry_price, trade=trade)
    spread_cost = units * trade.entry_spread if units else 0.0
    recovery_limit = _spread_recovery_mult() * spread_cost

    if pnl_dollars > spread_cost:
        return ""

    if spread_cost and pnl_dollars > -recovery_limit:
        return (
            f"Loss ${pnl_dollars:+.2f} within spread-recovery zone "
            f"(limit ±${recovery_limit:.2f}) — waiting for stop or clearer loss"
        )

    conf = int(exit_result.get("confidence", 0)) if exit_result else 0
    min_conf = _llm_loss_cut_min_conf()
    if conf >= min_conf:
        return ""
    return (
        f"LLM loss-cut needs ≥{min_conf}% confidence "
        f"(signal {conf}%)"
    )


def should_llm_close(
    trade: PaperTrade,
    ask: float,
    bid: float,
    trend_strength: str,
    *,
    exit_result: Optional[dict] = None,
) -> bool:
    """
    Gate: should we act on the LLM's CLOSE recommendation?

    1. In profit above the spread  → honour CLOSE (lock in the gain).
    2. Within the spread-recovery zone (tiny loss / near breakeven) → never close
       on the LLM; that's noise — let it recover or hit the mechanical stop.
    3. A REAL loss beyond the recovery zone → honour CLOSE when the exit signal is
       confident (the LLM has clearly seen the trend turn against the position).
       This lets the bot cut losers on a decisive reversal instead of riding all
       the way to the hard stop-loss.
    """
    reason = llm_close_veto_reason(trade, ask, bid, exit_result=exit_result)
    if reason:
        return False
    _, pnl_dollars, _, _ = dollar_unrealised_pnl(
        trade.direction, trade.entry_price, ask, bid, trade=trade,
    )
    _, units = _amount_and_units(trade.entry_price, trade=trade)
    spread_cost = units * trade.entry_spread if units else 0.0
    if pnl_dollars <= spread_cost:
        conf = int(exit_result.get("confidence", 0)) if exit_result else 0
        log.info(
            "LLM loss-cut: closing %s at $%.2f (conf %d, trend %s)",
            trade.direction, pnl_dollars, conf, trend_strength,
        )
    return True


def should_stop_loss(trade: PaperTrade, ask: float, bid: float) -> bool:
    if trade.direction == "LONG":
        return bid <= trade.stop_loss_price
    return ask >= trade.stop_loss_price


def trailing_stop_trigger_price(trade: PaperTrade, trail_pct: float) -> float:
    """Return the price level at which the trailing stop fires.

    Prefers the live CHANDELIER level (trail_stop_price — ratcheted 2xATR from
    peak) when the trade carries one; falls back to the legacy %-from-peak
    reconstruction otherwise.  Returns 0.0 when there is nothing to trail from.
    """
    if getattr(trade, "trail_stop_price", 0.0) > 0:
        return float(trade.trail_stop_price)
    if trail_pct <= 0 or trade.peak_pnl <= 0 or not trade.entry_price:
        return 0.0
    units = trade.trade_amount / trade.entry_price if trade.trade_amount else 0.0
    if not units:
        return 0.0
    # Reconstruct the PEAK (best) price from entry ± peak P&L per unit.  A LONG's
    # best price is ABOVE entry (price rose); a SHORT's best price is BELOW entry
    # (price fell) — so the per-unit peak P&L is SUBTRACTED for a short.  The
    # trailing then fires on a trail_pct pullback FROM that best price.
    peak_per_unit = trade.peak_pnl / units
    if trade.direction == "LONG":
        peak_price = trade.entry_price + peak_per_unit
        return peak_price * (1.0 - trail_pct / 100.0)
    peak_price = trade.entry_price - peak_per_unit
    return peak_price * (1.0 + trail_pct / 100.0)


def check_take_profit(
    bot_id: str,
    ask: float,
    bid: float,
    take_profit_pct: float,
    client: Optional["EToroClient"] = None,
) -> Optional["ClosedTrade"]:
    """Close THIS bot's trade when profit reaches take_profit_pct % of entry.

    Fires on the way UP once the target is reached.  Complementary to
    check_trailing_stop() (which fires on a pullback from peak).
    Safe to call every tick — returns None when not triggered or disabled.
    """
    if take_profit_pct <= 0:
        return None
    with _lock:
        trade = _open.get(bot_id)
        if not trade:
            return None
        if trade.direction == "LONG":
            pnl_pct = (bid - trade.entry_price) / trade.entry_price * 100
        else:  # SHORT
            pnl_pct = (trade.entry_price - ask) / trade.entry_price * 100
        if pnl_pct < take_profit_pct:
            return None
        # Claim the trade — move it out of _open atomically so no other thread
        # can act on it (or open a duplicate) while we do the HTTP close.
        del _open[bot_id]
        _closing[bot_id] = trade
    return _finalize_close(
        trade, ask, bid, "take_profit", client,
        llm_reasoning=f"Take profit: +{pnl_pct:.2f}% reached {take_profit_pct:.1f}% target",
    )


def check_trailing_stop(
    bot_id: str,
    ask: float,
    bid: float,
    trail_pct: float,
    client: Optional["EToroClient"] = None,
    *,
    atr_pct: Optional[float] = None,
    atr_mult: float = 0.0,
) -> Optional["ClosedTrade"]:
    """Close THIS bot's trade when price retreats to the trailing stop.

    TWO MODES, checked every tick:

    • CHANDELIER ATR (preferred — golden rule 2xATR; active when atr_mult>0 and
      a live atr_pct is supplied):
          LONG : level = max(level_prev, peak_price − k·ATR)
          SHORT: level = min(level_prev, peak_price + k·ATR)
      The level RATCHETS — it moves only in the trade's favour, never widens —
      and it is armed FROM ENTRY (level starts at the entry 2xATR hard stop),
      so it is the live stop, not just a profit lock.  k·ATR is recomputed each
      candle from current volatility, so the buffer breathes with the market.

    • LEGACY %-FROM-PEAK (fallback when no ATR is available): fires when price
      pulls back trail_pct% from the peak, armed only once in profit
      (peak_pnl > 0) — the original behaviour, unchanged.
    """
    use_atr = bool(atr_mult and atr_mult > 0 and atr_pct and atr_pct > 0)
    if not use_atr and trail_pct <= 0:
        return None
    with _lock:
        trade = _open.get(bot_id)
        if not trade:
            return None
        if use_atr:
            anchor = trade.peak_price or trade.entry_price
            atr_price = (float(atr_pct) / 100.0) * anchor
            if trade.direction == "LONG":
                candidate = anchor - atr_mult * atr_price
                level = max(trade.trail_stop_price or 0.0, candidate)
                trade.trail_stop_price = level
                triggered = bid > 0 and bid <= level
            else:
                candidate = anchor + atr_mult * atr_price
                prev = trade.trail_stop_price
                level = candidate if prev <= 0 else min(prev, candidate)
                trade.trail_stop_price = level
                triggered = ask > 0 and ask >= level
            reason_txt = (
                f"Chandelier ATR trail hit: {atr_mult:.1f}x ATR({atr_pct:.3f}%) "
                f"from peak {anchor:.5f} -> level {level:.5f}"
            )
        else:
            if trade.peak_pnl <= 0:
                return None
            stop = trailing_stop_trigger_price(trade, trail_pct)
            if not stop:
                return None
            triggered = (
                (trade.direction == "LONG"  and bid <= stop)
                or (trade.direction == "SHORT" and ask >= stop)
            )
            reason_txt = f"Trailing stop triggered: price pulled back >{trail_pct}% from peak"
        if not triggered:
            return None
        del _open[bot_id]
        _closing[bot_id] = trade
    return _finalize_close(
        trade, ask, bid, "trailing_stop", client,
        llm_reasoning=reason_txt,
    )


def check_recovery_exit(
    bot_id: str,
    ask: float,
    bid: float,
    strategy: str,
    client: Optional["EToroClient"] = None,
    *,
    enabled: bool = True,
    hold_mult: float = 2.5,
    breakeven_stop: bool = True,
) -> Optional["ClosedTrade"]:
    """Long-underwater trade recovers to ≥ $0 after overstaying — act on it.

    Arms once the position has never been meaningfully green (peak P&L within
    the spread-recovery zone) for at least ``hold_mult`` × the strategy's
    average hold time.  On the first tick with dollar P&L ≥ 0:

    • breakeven_stop=True (default): DON'T close — raise the stop to the entry
      price (a breakeven floor; never widens the existing stop) and let the
      take-profit / ATR chandelier trail keep the upside open.  If price breaks
      out, the winners run; if it rolls back over, check_stop_loss closes at
      ~no loss with reason "breakeven_stop".

    • breakeven_stop=False: legacy behaviour — close immediately at ≥ $0
      (reason "recovery_exit").
    """
    if not enabled or hold_mult <= 0:
        return None
    import trade_journal

    with _lock:
        trade = _open.get(bot_id)
        if not trade:
            return None

    strat = (strategy or trade.strategy or "").strip()
    avg_hold = trade_journal.avg_hold_min(strat)
    if avg_hold <= 0:
        return None

    _, pnl_d, pnl_pct, _ = dollar_unrealised_pnl(
        trade.direction, trade.entry_price, ask, bid, trade=trade,
    )
    amount, units = _amount_and_units(trade.entry_price, trade=trade)
    spread_cost = units * trade.entry_spread if units else 0.0
    green_threshold = _spread_recovery_mult() * spread_cost
    hold_min = minutes_open(trade)
    required_min = hold_mult * avg_hold

    with _lock:
        trade = _open.get(bot_id)
        if not trade:
            return None
        if trade.breakeven_set:
            return None   # floor already in place — stop/trail handle it from here
        if not trade.recovery_armed:
            if trade.peak_pnl <= green_threshold and hold_min >= required_min:
                trade.recovery_armed = True
                log.info(
                    "Recovery exit armed on %s (%s): underwater %.0f min "
                    "(≥ %.0f min = %.1f× %.0f min avg hold) — %s at ≥$0",
                    trade.instrument_label, strat, hold_min, required_min,
                    hold_mult, avg_hold,
                    "breakeven floor" if breakeven_stop else "will close",
                )
        if not trade.recovery_armed or pnl_d < 0:
            return None

        if breakeven_stop:
            # ── Breakeven floor: lock in "no loss", keep the upside open ──────
            # Raise (never widen) both the hard stop and the chandelier level to
            # the entry price.  TP and the ATR trail stay live: a breakout keeps
            # running; a roll-over closes at ~$0 via check_stop_loss.
            be = float(trade.entry_price)
            if trade.direction == "LONG":
                trade.stop_loss_price = max(trade.stop_loss_price or 0.0, be)
                trade.trail_stop_price = max(trade.trail_stop_price or 0.0, be)
            else:
                trade.stop_loss_price = (
                    min(trade.stop_loss_price, be) if trade.stop_loss_price > 0 else be
                )
                prev_trail = trade.trail_stop_price
                trade.trail_stop_price = be if prev_trail <= 0 else min(prev_trail, be)
            trade.breakeven_set = True
            trade.recovery_armed = False
            log.info(
                "Breakeven floor set on %s (%s): stop -> entry %.5f after "
                "%.0f min underwater; TP/ATR-trail keep the upside open",
                trade.instrument_label, strat, be, hold_min,
            )
            return None

        del _open[bot_id]
        _closing[bot_id] = trade

    return _finalize_close(
        trade, ask, bid, "recovery_exit", client,
        llm_reasoning=(
            f"Recovery exit: underwater {hold_min:.0f} min "
            f"(≥{hold_mult:.1f}× {avg_hold:.0f} min avg hold), "
            f"closed at breakeven ${pnl_d:+.2f} ({pnl_pct:+.2f}%)"
        ),
    )


def _etoro_close(client: "EToroClient", trade: PaperTrade) -> None:
    if not trade.on_etoro:
        return
    # If position ID was never resolved (eToro lag at open), try one live lookup
    if not trade.etoro_position_id:
        log.warning(
            "Position ID missing for instrument %s — attempting portfolio lookup before close",
            trade.instrument_id,
        )
        try:
            pos = client.find_demo_position(
                trade.instrument_id, trade.direction == "LONG"
            )
            if pos:
                pid = _extract_pid(pos)
                if pid:
                    trade.etoro_position_id = pid
                    log.info(
                        "Resolved missing position ID for instrument %s → %s (at close)",
                        trade.instrument_id, pid,
                    )
        except Exception as exc:
            log.warning("Portfolio lookup for position ID failed: %s", exc)
    if not trade.etoro_position_id:
        log.error(
            "Cannot close eToro position for instrument %s — position ID unknown",
            trade.instrument_id,
        )
        return
    try:
        # Exits never wait — but tell the pacer so the next ENTRY slot backs
        # off one spacing and the combined order stream stays under the per-key
        # rate limit (entries yield to exits, not the other way round).
        import order_executor
        order_executor.note_priority_order()
        client.close_demo_position(trade.etoro_position_id, trade.instrument_id)
        _set_error(None)
        log.info("eToro demo position %s closed", trade.etoro_position_id)
    except Exception as exc:
        msg = str(exc)
        if "404" in msg or "not found" in msg.lower():
            log.warning("eToro position %s already closed", trade.etoro_position_id)
            _set_error(None)
        else:
            _set_error(f"eToro close failed: {msg}")
            raise


def open_trade(
    instrument_id: int,
    instrument_label: str,
    signal: str,
    confidence: int,
    ask: float,
    bid: float,
    client: Optional["EToroClient"] = None,
    demo_amount: float = 0,
    bot_id: str = "",
    bot_key: str = "",
    strategy: str = "",
    exec_risk: str = "",
    net_edge_pct: float = 0.0,
    atr_pct: Optional[float] = None,
    interval: str = "",
    entry_reason: str = "",
    regime: str = "",
    confidence_calibrated: int = 0,
) -> Optional[PaperTrade]:
    signal = signal.upper()
    if signal == "BUY":
        direction: Direction = "LONG"
        entry_price = ask
        is_buy = True
    elif signal == "SELL":
        direction = "SHORT"
        entry_price = bid
        is_buy = False
    else:
        return None

    spread = ask - bid
    if spread <= 0:
        log.warning("Invalid spread %.5f — skipping open", spread)
        return None

    if not client or demo_amount <= 0:
        _set_error("eToro demo client or trade amount not configured")
        return None

    import exit_profiles
    # Regime-aware stop: widen the fixed floor toward k×ATR% in volatile regimes.
    # Falls back to the fixed floor when atr_pct is None (unchanged behaviour).
    stop_pct_used = exit_profiles.adaptive_stop_pct(strategy, instrument_label, atr_pct, bot_key)
    stop_loss = compute_stop_loss_price(
        direction, entry_price, spread,
        min_pct=stop_pct_used,
    )

    # ── Reserve THIS BOT before the HTTP order so it can't double-fire while its
    # own response is pending.  Keyed by bot_id, so sibling bots on the same
    # instrument are free to open their own positions concurrently.
    if not bot_id:
        _set_error("open_trade requires a bot_id")
        return None
    with _lock:
        if bot_id in _open or bot_id in _closing or bot_id in _opening:
            return None
        _opening.add(bot_id)

    try:
        etoro_position_id: Optional[int] = None
        # ── Steps 0-2 run under the GLOBAL open lock so concurrent bots open
        # one at a time.  This keeps the before/after portfolio diff
        # unambiguous and avoids the eToro 429 burst of simultaneous opens.
        with _open_serialize_lock:
            # ── Step 0: snapshot existing position ids BEFORE the order ──────
            try:
                before_ids = client.position_ids_for_instrument(instrument_id, is_buy)
            except Exception:
                before_ids = set()

            # ── Step 1: submit order — if THIS raises, no position was created ──
            try:
                resp = client.open_demo_market_by_amount(
                    instrument_id=instrument_id,
                    is_buy=is_buy,
                    amount=demo_amount,
                    leverage=1,
                    stop_loss_rate=stop_loss,
                )
            except PermissionError as exc:
                _set_error(str(exc))
                return None
            except Exception as exc:
                _set_error(f"eToro open failed: {exc}")
                log.error("eToro open failed: %s", exc)
                return None

            # ── Step 2: resolve position ID ──────────────────────────────────
            try:
                etoro_position_id = client.extract_position_id(resp)
            except Exception:
                etoro_position_id = None
            if etoro_position_id is None:
                etoro_position_id = _claim_new_position_id(
                    client, instrument_id, is_buy, before_ids, bot_id,
                )

        _set_error(None)
        trade = PaperTrade(
            instrument_id=instrument_id,
            instrument_label=instrument_label,
            direction=direction,
            entry_price=entry_price,
            entry_spread=spread,
            stop_loss_price=stop_loss,
            entry_time=datetime.now(tz=timezone.utc),
            signal=signal,
            confidence=confidence,
            etoro_position_id=etoro_position_id,   # may be None — filled by adopt flow
            trade_amount=demo_amount,
            on_etoro=True,
            shadow=False,
            opened_by_bot=True,
            bot_id=bot_id,
            strategy=strategy,
            exec_risk=exec_risk,
            net_edge_pct=net_edge_pct,
            interval=interval,
            entry_reason=entry_reason,
            regime=regime,
            atr_pct_entry=float(atr_pct or 0.0),
            stop_pct_entry=float(stop_pct_used),
            confidence_calibrated=int(confidence_calibrated or 0),
            # Chandelier trail starts AT the entry 2xATR hard stop and only
            # ratchets in the trade's favour from here.
            peak_price=float(entry_price),
            trail_stop_price=float(stop_loss),
        )

        with _lock:
            _open[bot_id] = trade
        if etoro_position_id is not None:
            _set_owner(etoro_position_id, bot_id)
        register_open_lineage(
            instrument_id, direction, trade.entry_time, entry_price, bot_id,
        )

        if etoro_position_id:
            log.info(
                "eToro demo %s opened @ %.5f  $%.0f  pos=%s  stop=%.5f  (bot=%s)",
                direction, entry_price, demo_amount, etoro_position_id, stop_loss, bot_id,
            )
        else:
            log.warning(
                "eToro demo %s order submitted for instrument %s but position ID not yet "
                "visible — adopt_etoro_position will resolve it on next portfolio refresh",
                direction, instrument_id,
            )
        return trade
    finally:
        # Always release the reservation, success or failure
        with _lock:
            _opening.discard(bot_id)


def _finalize_close(
    trade: PaperTrade,
    ask: float,
    bid: float,
    reason: str,
    client: Optional["EToroClient"] = None,
    llm_reasoning: Optional[str] = None,
    llm_observations: Optional[str] = None,
) -> ClosedTrade:
    """Complete a close that was claimed via _claim_for_close / direct _closing insertion.

    Performs the eToro HTTP close *without* holding ``_lock`` so concurrent
    bots' has_open / get_open / update_peak_pnl calls don't block during the
    network round-trip.  On HTTP failure the trade is moved back to ``_open``
    so the engine can retry on the next tick.
    """
    try:
        if client and trade.on_etoro:
            _etoro_close(client, trade)
    except Exception:
        with _lock:
            _closing.pop(trade.bot_id, None)
            # Reinsert only if this bot hasn't taken a new trade in the meantime
            _open.setdefault(trade.bot_id, trade)
        raise

    if trade.direction == "LONG":
        exit_price = bid
        profit = bid - trade.entry_price
    else:
        exit_price = ask
        profit = trade.entry_price - ask

    # Real dollar P&L: per-unit price move × units held.
    _units = (trade.trade_amount / trade.entry_price) if (trade.trade_amount and trade.entry_price) else 0.0
    _pnl_dollars = profit * _units

    closed = ClosedTrade(
        instrument_id=trade.instrument_id,
        instrument_label=trade.instrument_label,
        direction=trade.direction,
        entry_price=trade.entry_price,
        entry_spread=trade.entry_spread,
        entry_time=trade.entry_time,
        signal=trade.signal,
        confidence=trade.confidence,
        exit_price=exit_price,
        exit_time=datetime.now(tz=timezone.utc),
        profit=profit,
        reason=reason,
        trade_id=trade.trade_id,
        stop_loss_price=trade.stop_loss_price,
        etoro_position_id=trade.etoro_position_id,
        bot_id=trade.bot_id,
        llm_reasoning=llm_reasoning,
        llm_observations=llm_observations,
        shadow=getattr(trade, "shadow", False),
        pnl_dollars=round(_pnl_dollars, 4),
    )
    with _lock:
        _closing.pop(trade.bot_id, None)
        _closed.append(closed)
        # Block immediate re-adoption of this dead position from a stale
        # positions cache (the duplicate-P&L loop).
        _mark_recently_closed(trade.etoro_position_id)
    # Ownership is STICKY: we deliberately do NOT clear the position→bot mapping on
    # close.  A closed eToro position id never reappears in the portfolio, so a
    # stale entry can't cause a wrong adoption — but keeping it means a *false*
    # vanish (transient positions-cache gap) can never strip a still-open
    # position's identity, and History can always attribute the trade.
    # Durable outcome journal (survives restarts) — drives the learning loop.
    trade_journal.record(trade, closed)
    log.info(
        "Closed %s @ %.5f  profit=%.5f  (%s)", trade.direction, exit_price, profit, reason
    )
    return closed


def check_stop_loss(
    bot_id: str,
    ask: float,
    bid: float,
    client: Optional["EToroClient"] = None,
) -> Optional[ClosedTrade]:
    with _lock:
        trade = _open.get(bot_id)
        if not trade or not should_stop_loss(trade, ask, bid):
            return None
        del _open[bot_id]
        _closing[bot_id] = trade
    # A stop raised to entry by the recovery breakeven floor closes at ~$0 —
    # journal it distinctly so analytics separate "saved from loss" from real
    # stop-loss damage.
    if getattr(trade, "breakeven_set", False):
        return _finalize_close(
            trade, ask, bid, "breakeven_stop", client,
            llm_reasoning=(
                "Breakeven floor hit: price rolled back to entry after the "
                "recovery arm — closed at ~no loss instead of riding red again"
            ),
        )
    return _finalize_close(trade, ask, bid, "stop_loss", client)


def close_llm(
    bot_id: str,
    ask: float,
    bid: float,
    client: Optional["EToroClient"] = None,
    reasoning: Optional[str] = None,
    observations: Optional[str] = None,
) -> Optional[ClosedTrade]:
    with _lock:
        trade = _open.get(bot_id)
        if not trade:
            return None
        del _open[bot_id]
        _closing[bot_id] = trade
    return _finalize_close(
        trade, ask, bid, "llm", client,
        llm_reasoning=reasoning,
        llm_observations=observations,
    )


def close_manual(
    bot_id: str,
    ask: float,
    bid: float,
    client: Optional["EToroClient"] = None,
    reason: str = "manual",
) -> Optional[ClosedTrade]:
    """Close a bot's tracked trade locally.

    `reason="manual"`   → the user closed it (portfolio table / eToro app).
    `reason="external"` → the position vanished from eToro (its own stop-loss /
                          take-profit fired, or eToro merged it) and we detected
                          it via portfolio reconciliation, not a bot exit signal.
    """
    with _lock:
        trade = _open.pop(bot_id, None)
        if not trade:
            return None
        _closing[bot_id] = trade

    # HTTP close runs *outside* the lock so other bots' tick loops don't block.
    try:
        if client and trade.on_etoro:
            _etoro_close(client, trade)
    except Exception:
        with _lock:
            _closing.pop(bot_id, None)
            _open.setdefault(bot_id, trade)
        raise

    closed = ClosedTrade(
        instrument_id=trade.instrument_id,
        instrument_label=trade.instrument_label,
        direction=trade.direction,
        entry_price=trade.entry_price,
        entry_spread=trade.entry_spread,
        entry_time=trade.entry_time,
        signal=trade.signal,
        confidence=trade.confidence,
        exit_price=bid if trade.direction == "LONG" else ask,
        exit_time=datetime.now(tz=timezone.utc),
        profit=unrealised_pnl(trade, ask, bid),
        reason=reason,
        trade_id=trade.trade_id,
        stop_loss_price=trade.stop_loss_price,
        etoro_position_id=trade.etoro_position_id,
        bot_id=trade.bot_id,
    )
    with _lock:
        _closing.pop(bot_id, None)
        _closed.append(closed)
        # Same re-adoption guard as _finalize_close — external/manual closes
        # were the main source of the duplicate-records loop.
        _mark_recently_closed(trade.etoro_position_id)
    # Ownership is sticky — see _finalize_close.  Not cleared on close.
    trade_journal.record(trade, closed)
    return closed


def close_by_position_id(
    position_id,
    ask: float,
    bid: float,
    client: Optional["EToroClient"] = None,
) -> Optional[ClosedTrade]:
    """Manually close a tracked position by its eToro position id (UI close
    button).  Resolves the owning bot so the right trade is closed."""
    trade = find_open_by_position_id(position_id)
    if trade is None:
        return None
    return close_manual(trade.bot_id, ask, bid, client)
