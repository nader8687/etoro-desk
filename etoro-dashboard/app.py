import logging
import os
import threading
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests as http_requests
import streamlit as st

import log_buffer
logging.basicConfig(level=logging.INFO)
log_buffer.install(logging.INFO)   # capture all module logs into the Logs tab

from etoro_client import EToroClient, get_shared_client
import engine_notify
import bot_ranking
import bot_registry
import market_data_hub
import positions_cache
import prompt_preview
import runtime_persist
import instrument_config
import position_sizer
import signal_log
import signal_worker
from signal_worker import display_asset_name
import tick_manager
import strategies
import timez
import trade_journal
import trade_manager
import trading_engine
from trading_engine import EngineConfig
import ui
from tick_manager import State
from views import tables as vtables

VISUAL_BOT_URL = os.environ.get("VISUAL_BOT_URL", "http://visual-bot:8080")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="eToro Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()

_PNL_PERIOD_OPTIONS = ("Today", "Yesterday", "7 days", "30 days", "All time", "Custom")
_DEFAULT_PERIOD = "Today"
# General Stats sidebar: bot P&L and open positions from this date onward (display TZ).
_GENERAL_STATS_SINCE = date(2026, 6, 3)


def _init_persistent_state() -> None:
    """Defaults for keys that must survive tab switches (Streamlit widget cleanup)."""
    saved = runtime_persist.load()
    for key in (
        "engine_instrument_id", "engine_selected_label",
        "engine_interval_label", "engine_interval_seconds", "engine_candle_count",
        "demo_trade_amount", "auto_trade_active", "display_tz",
    ):
        if key in saved:
            st.session_state.setdefault(key, saved[key])
    try:
        import user_settings as _us
        _cfg = _us.load()
        st.session_state.setdefault(
            "demo_trade_amount",
            float(_cfg["trading"].get("demo_trade_amount", 1000.0)),
        )
        st.session_state.setdefault(
            "display_tz",
            str(_cfg["display"].get("display_tz", "Asia/Dubai") or "Asia/Dubai"),
        )
    except Exception:
        st.session_state.setdefault("display_tz", "Asia/Dubai")
    for _pk in ("pnl_period_mode", "perf_period_mode", "hist_period_mode"):
        st.session_state.setdefault(_pk, _DEFAULT_PERIOD)
    # Live chart defaults ON; restore persisted choice if the user explicitly turned it off
    _saved_live = saved.get("live_feed", True)
    st.session_state.setdefault("live_feed", _saved_live)
    st.session_state.setdefault("feed_live", False)
    if "_live_feed_toggle" not in st.session_state:
        st.session_state["_live_feed_toggle"] = _saved_live


def _default_instrument_label() -> str:
    for lbl in INSTRUMENT_LABELS:
        if "EUR/USD" in lbl:
            return lbl
    return INSTRUMENT_LABELS[0] if INSTRUMENT_LABELS else ""


def _restore_trading_toolbar_widgets() -> None:
    """Re-hydrate Trading toolbar widget keys after leaving the tab (Streamlit clears them).

    The Trading tab is bound to one bot via the bot selector (key
    `_trading_bot_select`, set from `engine_active_bot_key`); only the candle
    count is an independent widget that needs restoring here."""
    if "_candle_count" not in st.session_state:
        st.session_state["_candle_count"] = int(
            st.session_state.get("engine_candle_count", 300)
        )


def _active_bot_key() -> "str | None":
    """Return the bot the Trading tab is bound to (engine_active_bot_key).

    The Trading tab is bound to exactly one bot, chosen in its bot selector and
    set by Bots-tab "View →".  This returns that bot's key as long as it is still
    a configured bot.  We do NOT validate it against engine_interval_label: the
    bot IS the source of truth for the interval (the toolbar derives interval
    from the bot), so cross-checking against a possibly-stale interval would
    wrongly invalidate a freshly-selected bot before the toolbar runs.
    """
    key = st.session_state.get("engine_active_bot_key")
    if not key:
        return None
    for spec in instrument_config.load_specs():
        if spec.key == key:
            return key
    # Bot key no longer in instruments.toml — invalidate.
    st.session_state.pop("engine_active_bot_key", None)
    return None


def _active_strategy_for(instrument_id: int) -> str:
    """Strategy key of the bot the Trading tab is currently bound to.

    The Trading tab always follows exactly one bot (chosen in its toolbar), so
    this resolves that bot's strategy — a non-LLM bot shows its own signal panel,
    never the AI panel + LLM prompt.  Falls back to the instrument's primary bot
    only if no active bot is set (e.g. before the toolbar has run).
    """
    return trading_engine.get_strategy(instrument_id, bot_id=_active_bot_key())


def _short_interval(interval_label: str) -> str:
    """'1 Minute' -> '1m', '15 Minutes' -> '15m', '1 Hour' -> '1h'."""
    parts = (interval_label or "").split()
    if len(parts) >= 2 and parts[0].isdigit():
        u = parts[1].lower()
        unit = ("m" if u.startswith("min") else "h" if u.startswith("hour")
                else "d" if u.startswith("day") else "w" if u.startswith("week") else u[:1])
        return f"{parts[0]}{unit}"
    return interval_label


_FREQ_FILTER_LABELS: dict[str, str] = {
    "__all__":   "All frequencies",
    "scalp":     "Fast (1m–5m)",
    "intraday":  "Intraday (10m–1h)",
    "swing":     "Swing (4h)",
    "daily":     "Daily",
    "weekly":    "Weekly+",
}


def _freq_bucket(interval_secs: int) -> str:
    """Map candle interval to a trading-frequency bucket for Bots-tab filtering."""
    secs = int(interval_secs or 60)
    if secs <= 300:
        return "scalp"
    if secs <= 3600:
        return "intraday"
    if secs <= 14400:
        return "swing"
    if secs <= 86400:
        return "daily"
    return "weekly"


def _bot_display_name(bot_key: str) -> str:
    """Friendly bot label for the Portfolio/History 'Bot' column — appends the
    interval so two bots on the same asset (e.g. an XRP 1m and an XRP 15m) are
    unmistakable.  Falls back to the raw key if the bot isn't in the config."""
    if not bot_key:
        return "Bot"
    for spec in instrument_config.load_specs():
        if spec.key == bot_key:
            return f"{bot_key} · {_short_interval(spec.interval)}"
    return bot_key


def _trading_bot_options() -> list[dict]:
    """All configured bots for the Trading-tab bot selector (resolved from
    instruments.toml).  The Trading tab is bound to exactly one of these at a
    time; each entry carries everything the tab needs to render that bot."""
    specs = instrument_config.load_specs()
    resolved = instrument_config.resolve_ids(specs, globals().get("ALL_INSTRUMENTS") or {})
    strat_names = strategies.display_names()
    out: list[dict] = []
    for s in resolved:
        asset = display_asset_name(s.label)
        strat = strat_names.get(s.strategy, s.strategy)
        out.append({
            "key":            s.key,
            "label":          s.label,
            "instrument_id":  s.instrument_id,
            "interval":       s.interval,
            "interval_secs":  s.interval_secs,
            "strategy":       s.strategy,
            "candle_count":   s.candle_count,
            "display":        f"{asset} · {s.interval} · {strat}",
        })
    return out


def _on_trading_bot_pick() -> None:
    """Selectbox on_change — bind the Trading tab to the bot the user picked.

    Using a callback (which fires ONLY on real user interaction) makes
    engine_active_bot_key authoritative: a stale or cleared widget value can
    never override an external selection made via Bots-tab 'View →'.
    """
    sel = st.session_state.get("_trading_bot_select")
    if sel:
        st.session_state["engine_active_bot_key"] = sel


def _active_chart_snapshot(instrument_id: int):
    """Chart snapshot for the bot the Trading tab is bound to (its own interval).

    Never falls back to the instrument's primary hub — that is often a sibling
    bot on a different interval (e.g. 15m vs 1m) and would draw the wrong chart."""
    active = _active_bot_key()
    if active:
        return market_data_hub.get_snapshot(bot_id=active)
    return market_data_hub.get_snapshot(instrument_id)


def _persist_widget_state() -> None:
    """Sync engine_* persistent keys from the active bot (bot-centric model).

    The Trading tab is bound to one bot; engine_* (instrument, interval, …) are
    derived from that bot's spec so every consumer stays consistent regardless of
    which tab triggered the rerun.  Falls back to leaving values untouched when no
    bot is active yet (first boot, before the Trading toolbar has run)."""
    if "_live_feed_toggle" in st.session_state:
        st.session_state["live_feed"] = st.session_state["_live_feed_toggle"]

    active = st.session_state.get("engine_active_bot_key")
    if active:
        instruments = globals().get("ALL_INSTRUMENTS") or {}
        for spec in instrument_config.load_specs():
            if spec.key != active:
                continue
            iid = spec.instrument_id or instruments.get(spec.label, 0)
            st.session_state["engine_selected_label"]   = spec.label
            if iid:
                st.session_state["engine_instrument_id"] = iid
            st.session_state["engine_interval_label"]   = spec.interval
            st.session_state["engine_interval_seconds"] = spec.interval_secs
            # INTERVALS is defined later in the module; this function runs from
            # the sidebar before that point, so resolve it order-safely.
            _intervals = globals().get("INTERVALS") or {}
            if spec.interval in _intervals:
                st.session_state["engine_api_name"] = _intervals[spec.interval][0]
            break

    if "_candle_count" in st.session_state:
        st.session_state["engine_candle_count"] = int(st.session_state["_candle_count"])


def _on_live_feed_toggle() -> None:
    """UI only — toggles live chart display; engine keeps running in background."""
    st.session_state["live_feed"] = st.session_state.get("_live_feed_toggle", True)
    runtime_persist.save(dict(st.session_state))


def _apply_start_auto_trade() -> None:
    """Enable auto-trade on booted engines and the current Trading-tab instrument."""
    st.session_state["auto_trade_active"] = True
    trading_engine.set_all_auto_trade(True)
    iid = st.session_state.get("engine_instrument_id")
    if not iid:
        return
    trading_engine.set_desired_live(True)
    candle_count = int(st.session_state.get("engine_candle_count", 300))
    api_name = st.session_state.get("engine_api_name", "OneMinute")
    hist_cache_key = f"hist_{iid}_{api_name}_{candle_count}"
    config = EngineConfig(
        instrument_id=iid,
        instrument_label=st.session_state.get("engine_selected_label", ""),
        interval_label=st.session_state.get("engine_interval_label", "1 Minute"),
        interval_seconds=st.session_state.get("engine_interval_seconds", 60),
        candle_count=candle_count,
        trading_active=True,
        demo_amount=float(st.session_state.get("demo_trade_amount", 1000.0)),
        is_demo=st.session_state.get("is_demo", True),
        api_key=os.environ.get("ETORO_API_KEY", ""),
        user_key=os.environ.get("ETORO_USER_KEY", ""),
    )
    hist_df = st.session_state.get(hist_cache_key, pd.DataFrame())
    trading_engine.update_from_ui(config, hist_df)
    trading_engine.set_auto_trade(iid, True)


_init_persistent_state()

# Bumps on every full-script rerun (tab switch, button, etc.) — NOT on fragment-only
# timer ticks.  General Stats uses this to avoid recomputing on navigation.
st.session_state["_parent_run_token"] = st.session_state.get("_parent_run_token", 0) + 1

# Apply the user's chosen display timezone for this rerun.  All presentation
# helpers (ui, views.tables, charts) localise through timez; storage stays UTC.
timez.set_active(st.session_state.get("display_tz") or "Asia/Dubai")

_perf_log = logging.getLogger("perf")


def _timed(label: str, fn, *a, **k):
    """Run fn and log a ⏱ line when it exceeds 150ms — pinpoints render stalls."""
    t0 = time.perf_counter()
    try:
        return fn(*a, **k)
    finally:
        dt = (time.perf_counter() - t0) * 1000
        if dt > 150:
            _perf_log.info("⏱ %s took %.0fms (nav=%s)", label, dt, st.session_state.get("main_nav"))

# ── Sidebar ───────────────────────────────────────────────────────────────────

api_key  = os.environ.get("ETORO_API_KEY", "")
user_key = os.environ.get("ETORO_USER_KEY", "")


@st.cache_data(ttl=30, show_spinner=False)
def _visual_bot_ok(_url: str) -> bool:
    try:
        r = http_requests.get(f"{_url}/health", timeout=3)
        return r.json().get("status") == "ok"
    except Exception:
        return False


# ── Background visual-bot health probe ────────────────────────────────────────
# The sidebar renders on EVERY rerun (and twice per tab switch).  Probing the
# visual-bot /health synchronously there blocks the render for the request
# latency (≈0.8s observed, up to the 3s timeout when it's unreachable), which is
# the tab-switch lag.  Run the probe on a background thread instead and let the
# sidebar read the last result instantly.
_VBOT_OK: bool = False
_vbot_poller_started = False


def _vbot_health_loop() -> None:
    global _VBOT_OK
    while True:
        try:
            r = http_requests.get(f"{VISUAL_BOT_URL}/health", timeout=3)
            _VBOT_OK = (r.json().get("status") == "ok")
        except Exception:
            _VBOT_OK = False
        time.sleep(10)


def _ensure_vbot_poller() -> None:
    global _vbot_poller_started
    if _vbot_poller_started:
        return
    _vbot_poller_started = True
    threading.Thread(target=_vbot_health_loop, daemon=True, name="vbot-health").start()


_GS_REFRESH_SEC = 60.0   # panel recompute + background eToro probe interval

# ── Background General Stats probes ───────────────────────────────────────────
# liquidity_summary and trade history are synchronous eToro REST calls.  Doing
# them in the sidebar render path blocked reruns — a daemon thread refreshes
# both; the sidebar reads the last result.
_GS_LIQUIDITY: dict = {"equity": None, "free_cash": None, "spendable": None, "reserve": None}
_GS_HISTORY: list[dict] = []
_gs_liq_poller_started = False
_gs_bootstrapped = False


def _trade_entry_local_date(trade) -> date | None:
    if not trade or not getattr(trade, "entry_time", None):
        return None
    loc = timez.to_local(trade.entry_time)
    return loc.date() if loc else None


def _bot_positions_since(positions: list, since: date) -> list:
    """Bot-owned open rows with entry on or after `since` (display timezone)."""
    open_by_pid: dict[str, object] = {
        str(t.etoro_position_id): t
        for t in trade_manager.get_all_open()
        if t.etoro_position_id is not None
    }
    filtered: list = []
    for p in trade_manager.bot_owned_positions(positions):
        pid = str(p.get("position_id") or "")
        trade = open_by_pid.get(pid)
        if trade is None:
            continue
        opened = _trade_entry_local_date(trade)
        if opened is not None and opened >= since:
            filtered.append(p)
    return filtered


def _etoro_row_close_local_date(row: dict) -> date | None:
    """Close date of an eToro history row in the user's display timezone."""
    ts = row.get("closeTimestamp") or row.get("close_timestamp")
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        loc = timez.to_local(dt)
        return loc.date() if loc else None
    except (TypeError, ValueError):
        return None


def _bot_realized_etoro_since(trades: list[dict], since: date) -> tuple[float, int]:
    """Sum eToro netProfit for bot-attributed closes on or after `since`."""
    total = 0.0
    n = 0
    for t in trades:
        closed = _etoro_row_close_local_date(t)
        if closed is None or closed < since:
            continue
        if not trade_manager.resolve_history_owner(t):
            continue
        pnl = t.get("netProfit")
        if pnl is None:
            continue
        total += float(pnl)
        n += 1
    return round(total, 2), n


def _gs_fetch_etoro() -> None:
    """One eToro liquidity + history fetch into module-level caches."""
    global _GS_LIQUIDITY, _GS_HISTORY
    if not api_key or not user_key:
        return
    import position_sizer
    c = get_shared_client(api_key, user_key)
    demo = bool(_poller_is_demo())
    _GS_LIQUIDITY = position_sizer.liquidity_summary(c, demo)
    _GS_HISTORY = c.get_all_trade_history(
        _GENERAL_STATS_SINCE.strftime("%Y-%m-%d"),
        demo=demo,
    )


def _gs_liquidity_loop() -> None:
    while True:
        try:
            _gs_fetch_etoro()
        except Exception:
            pass
        time.sleep(_GS_REFRESH_SEC)


def _gs_cache_ready(cached: dict | None) -> bool:
    """True once the panel has real eToro liquidity (not a pre-bootstrap stub)."""
    if not cached:
        return False
    liq = cached.get("liquidity") or {}
    return liq.get("free_cash") is not None or liq.get("equity") is not None


def _poller_is_demo() -> bool:
    # Module-level mirror of the sidebar selector — session_state isn't safely
    # readable from a daemon thread, so the render path keeps this in sync.
    return bool(globals().get("_gs_liq_demo", True))


def _ensure_gs_liquidity_poller() -> None:
    global _gs_liq_poller_started, _gs_bootstrapped
    if not _gs_bootstrapped:
        _gs_bootstrapped = True
        try:
            _gs_fetch_etoro()
        except Exception:
            pass
    if _gs_liq_poller_started:
        return
    _gs_liq_poller_started = True
    threading.Thread(target=_gs_liquidity_loop, daemon=True, name="gs-liquidity").start()


_GS_PANEL_REFRESH_SEC = 60


@st.fragment(run_every=_GS_PANEL_REFRESH_SEC)
def _render_general_stats_panel() -> None:
    """Account-wide eToro stats for the sidebar (under Live Market).

    Invested / unrealized / equity match eToro's Virtual Portfolio bar (same
    get_pnl snapshot).  Bot realized is a separate figure: bot-attributed closed
    trades since _GENERAL_STATS_SINCE.

    Auto-refreshing FRAGMENT: recomputes at most once per minute (timer only).
    Tab switches repaint the last cached numbers — they do NOT trigger a
    refresh.  eToro REST runs in a background thread; this path is cache-only."""
    import position_sizer

    since = _GENERAL_STATS_SINCE
    cached = st.session_state.get("_gs_cached")
    parent_token = st.session_state.get("_parent_run_token", 0)
    seen_parent = st.session_state.get("_gs_seen_parent_token", -1)
    is_parent_rerun = seen_parent != parent_token
    if is_parent_rerun:
        st.session_state["_gs_seen_parent_token"] = parent_token
    globals()["_gs_liq_demo"] = st.session_state.get("is_demo", True)
    _ensure_gs_liquidity_poller()

    # First run + 60s timer + incomplete bootstrap refresh; tab switches do not.
    should_refresh = (
        cached is None
        or not _gs_cache_ready(cached)
        or not is_parent_rerun
    )

    if should_refresh:
        liquidity = dict(_GS_LIQUIDITY)
        invested   = float(liquidity.get("invested") or 0)
        unrealized = float(liquidity.get("unrealized") or 0)
        open_value = float(liquidity.get("open_value") or (invested + unrealized))
        pos_count  = int(liquidity.get("position_count") or 0)

        hist = list(_GS_HISTORY)
        if hist:
            bot_realized, _realized_n = _bot_realized_etoro_since(hist, since)
            realized_source = "etoro"
        else:
            try:
                bot_realized = float(trade_journal.bot_realized_since(since)["total_pnl"])
            except Exception:
                bot_realized = 0.0
            realized_source = "journal"

        cached = {
            "invested": invested, "unrealized": unrealized,
            "open_value": open_value, "position_count": pos_count,
            "bot_realized": bot_realized, "liquidity": liquidity,
            "realized_source": realized_source,
        }
        st.session_state["_gs_cached"] = cached
    else:
        invested   = float(cached.get("invested") or 0)
        unrealized = float(cached.get("unrealized") or 0)
        open_value = float(cached.get("open_value") or 0)
        bot_realized = float(cached.get("bot_realized") or 0)
        pos_count  = int(cached.get("position_count") or 0)
        liquidity  = dict(cached.get("liquidity") or {})

    def _acct_money(val: float | None) -> str:
        return f"${float(val):,.2f}" if val is not None else "—"

    with st.container(border=True):
        ui.section_title("General Stats")
        cap = (
            f"Account totals from eToro (matches Virtual Portfolio). "
            f"Bot realized = bot-attributed closes since {since.strftime('%Y-%m-%d')} "
            f"({timez.active_name()})"
        )
        st.caption(cap)
        c1, c2 = st.columns(2)
        c1.metric(
            "Invested", _acct_money(invested),
            help=f"Total capital in all {pos_count} open position(s) — eToro Total Invested",
        )
        c2.metric(
            "Value now", _acct_money(open_value),
            help="Invested + unrealized P&L on open positions (eToro)",
        )
        c3, c4 = st.columns(2)
        c3.metric(
            "P/L", _acct_money(unrealized),
            delta=f"{unrealized:+,.2f}" if liquidity.get("unrealized") is not None else None,
            help="Account unrealized P&L on open positions — eToro Profit/Loss",
        )
        _real_src = cached.get("realized_source", "etoro")
        _real_help = (
            f"Bot-attributed closed-trade P&L since {since.isoformat()} "
            f"(not the eToro portfolio P/L above)"
            if _real_src == "etoro"
            else "Journal fallback until eToro history loads"
        )
        c4.metric(
            "Bot realized", f"${bot_realized:,.2f}", delta=f"{bot_realized:+,.2f}",
            help=_real_help,
        )
        # Account-wide liquidity (NOT bot-filtered): free cash − reserve = spendable.
        st.caption("Account liquidity")
        equity    = liquidity.get("equity")
        free_cash = liquidity.get("free_cash")
        reserve   = liquidity.get("reserve")
        spendable = liquidity.get("spendable")

        def _money(v) -> str:
            return _acct_money(v)

        c5, c6 = st.columns(2)
        c5.metric(
            "Free cash", _money(free_cash),
            help="Uninvested credit on the eToro account (before reserve policy)",
        )
        c6.metric(
            "Reserve", _money(reserve),
            help=f"Cash floor kept untouched ({position_sizer.cash_reserve_pct():.0f}% of "
                 f"free cash); cash-freeing may dip to {position_sizer.reserve_hard_pct():.0f}% "
                 f"only to fund a strong signal",
        )
        c7, c8 = st.columns(2)
        c7.metric(
            "Spendable", _money(spendable),
            help="Free cash minus the reserve — what new bot trades can deploy right now",
        )
        # Equity ground truth vs journal claim — the reconciliation line.
        try:
            import equity_log
            _eq_day = equity_log.day_stats()
            if _eq_day:
                _j_day = equity_log.journal_day_pnl()
                _gap = _eq_day["change"] - _j_day
                st.caption(
                    f"📊 Equity today: **${_eq_day['change']:+,.2f}** · journal "
                    f"claims ${_j_day:+,.2f} (gap ${_gap:+,.2f} = slippage, fees, "
                    f"manual & open-position drift) · day low "
                    f"${_eq_day['low']:,.0f} · {_eq_day['n']} snapshots"
                )
        except Exception:
            pass
        c8.metric(
            "Equity", _money(equity),
            help="Total account value: free cash + invested + unrealized P&L",
        )


with st.sidebar:
    ui.render_sidebar_branding()

    with st.container(border=True):
        ui.section_title("Account")
        account_type = st.radio(
            "Environment", ["Demo", "Real"], horizontal=True,
            help="Auto-trading executes on Demo virtual money only",
            label_visibility="collapsed",
        )
    is_demo = account_type == "Demo"
    st.session_state["is_demo"] = is_demo

    with st.container(border=True):
        ui.section_title("Auto Trading")
        if is_demo:
            demo_trade_amount = st.number_input(
                "Size per trade ($)", min_value=10.0,
                value=float(st.session_state.get(
                    "demo_trade_amount", position_sizer.demo_trade_default(),
                )),
                step=10.0,
                help="Fallback cap when account sizing is unavailable; also editable in Settings.",
            )
            st.session_state["demo_trade_amount"] = demo_trade_amount
            _at_count = trading_engine.auto_trade_count()
            if _at_count > 0:
                if st.button(
                    f"⏹ Stop Auto-Trade ({_at_count} active)",
                    key="_sidebar_stop_all_at",
                    type="primary",
                    width="stretch",
                    help="Stop auto-trading on all instruments",
                ):
                    trading_engine.set_all_auto_trade(False)
                    st.session_state["auto_trade_active"] = False
                    runtime_persist.save(dict(st.session_state))
                    st.rerun()
            else:
                if st.button(
                    "▶ Start Auto-Trade",
                    key="_sidebar_start_all_at",
                    type="secondary",
                    width="stretch",
                    help="Enable auto-trading on all configured instruments",
                ):
                    _apply_start_auto_trade()
                    runtime_persist.save(dict(st.session_state))
                    st.rerun()
            # Strategy is chosen per-bot on the Bots tab (the authoritative
            # control); the sidebar no longer duplicates that selector.
            st.caption("Market orders on eToro demo via API")
        else:
            demo_trade_amount = 0.0
            st.session_state["demo_trade_amount"] = 0.0
            st.caption("Select **Demo** to enable auto-trading.")

    with st.container(border=True):
        ui.section_title("Live Market")
        st.toggle(
            "Live chart",
            key="_live_feed_toggle",
            on_change=_on_live_feed_toggle,
            help="Show real-time ticks on the Trading chart. "
                 "WebSocket + auto-trade engine always run in the background.",
        )
        _persist_widget_state()
        if st.session_state.get("live_feed"):
            st.caption("Live chart on · engine always running")
        else:
            st.caption("Static chart · engine still running in background")

    _render_general_stats_panel()

    st.caption("eToro Public API v1")

# ── Auth gate ─────────────────────────────────────────────────────────────────

if not api_key or not user_key:
    st.error(
        "**Missing credentials.**  \n"
        "Set `ETORO_API_KEY` and `ETORO_USER_KEY` in the `.env` file and restart the container."
    )
    st.stop()

# ── Client (cached per credentials pair) ─────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_client(k: str, u: str) -> EToroClient:
    return get_shared_client(k, u)

client = get_client(api_key, user_key)

# ── Intervals: (api_name, seconds_per_candle) ─────────────────────────────────

INTERVALS: dict[str, tuple] = {
    "1 Minute":   ("OneMinute",       60),
    "5 Minutes":  ("FiveMinutes",    300),
    "10 Minutes": ("TenMinutes",     600),
    "15 Minutes": ("FifteenMinutes", 900),
    "30 Minutes": ("ThirtyMinutes", 1800),
    "1 Hour":     ("OneHour",       3600),
    "4 Hours":    ("FourHours",    14400),
    "1 Day":      ("OneDay",       86400),
    "1 Week":     ("OneWeek",     604800),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_list(data: dict, *keys: str) -> list:
    for k in keys:
        val = data.get(k)
        if isinstance(val, list) and val:
            return val
    for v in data.values():
        if isinstance(v, dict):
            for k in keys:
                val = v.get(k)
                if isinstance(val, list) and val:
                    return val
    return []


def permission_error(label: str, *, scope_hint: str = "Trading") -> None:
    st.warning(
        f"**Permission denied (403)** for *{label}*.\n\n"
        f"Your API key needs the **{scope_hint}** scope enabled.\n\n"
        "Go to **eToro → Settings → API → edit your key → enable the required permissions**."
    )


def desk_closed_trades_df(since: datetime | None = None) -> pd.DataFrame:
    """Closed trades recorded by EtoroDesk this session (auto-trade / manual close)."""
    rows = []
    for t in trade_manager.get_closed():
        exit_at = t.exit_time if t.exit_time.tzinfo else t.exit_time.replace(tzinfo=timezone.utc)
        if since and exit_at < since:
            continue
        rows.append({
            "Bot ID":        t.bot_id[:8] if t.bot_id else "—",
            "Instrument":    t.instrument_label,
            "Direction":     t.direction,
            "Signal":        t.signal,
            "Confidence %":  t.confidence,
            "Entry":         round(t.entry_price, 5),
            "Exit":          round(t.exit_price, 5),
            "P&L":           round(t.profit, 5),
            "Reason":        t.reason,
            "Opened":        timez.fmt(t.entry_time, "%Y-%m-%d %H:%M"),
            "Closed":        timez.fmt(t.exit_time, "%Y-%m-%d %H:%M"),
            "Position #":    t.etoro_position_id or "—",
        })
    return pd.DataFrame(rows)


def render_closed_trades_summary(df: pd.DataFrame, pnl_col: str = "P&L") -> None:
    if df.empty or pnl_col not in df.columns:
        return
    total = df[pnl_col].sum()
    wins = int((df[pnl_col] > 0).sum())
    losses = int((df[pnl_col] < 0).sum())
    wr = wins / len(df) * 100 if len(df) else 0
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Closed Trades", len(df))
    h2.metric("Realised P&L", f"${total:,.2f}", delta=f"{total:+,.2f}")
    h3.metric("Win Rate", f"{wr:.1f}%")
    h4.metric("W / L", f"{wins} / {losses}")


def _render_signal_record(rec: dict) -> None:
    """One expandable card for a single strategy signal record."""
    sig_type  = rec.get("type", "entry")
    decision  = (rec.get("signal") if sig_type == "entry" else rec.get("action")) or "HOLD"
    decision  = decision.upper()
    instrument = rec.get("instrument_label", "Unknown")
    interval   = rec.get("interval", "")
    confidence = rec.get("confidence", 0)
    ts_raw     = rec.get("ts", "")
    trigger_at = rec.get("trigger_at", "")
    # Strategy display: use stored key → display name; fall back to "LLM" for old records
    strategy_key   = rec.get("strategy", "llm")
    strategy_label = strategies.display_names().get(strategy_key, strategy_key.upper())

    ts_display = timez.fmt_iso(ts_raw, "%Y-%m-%d %H:%M:%S") if ts_raw else ts_raw

    icon  = {"BUY": "🟢", "SELL": "🔴", "CLOSE": "🟠", "HOLD": "⚪"}.get(decision, "⚪")
    tag   = " · EXIT" if sig_type == "exit" else ""
    exec_status = rec.get("exec_status")
    exec_reason = (rec.get("exec_reason") or "").strip()
    exec_tag = ""
    if exec_status == "executed":
        exec_tag = " · ✅ executed"
    elif exec_status == "skipped":
        exec_tag = " · ⛔ not executed"
    elif exec_status == "not_applicable":
        exec_tag = " · — no order"
    elif decision in ("BUY", "SELL", "CLOSE"):
        exec_tag = " · ⏳ pending"
    label = (
        f"{icon} {decision}  ·  {instrument}  ·  {strategy_label}  ·  "
        f"{interval}  ·  {ts_display}  ·  {confidence}%{tag}{exec_tag}"
    )

    with st.expander(label, expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Decision", decision)
        c2.metric("Confidence", f"{confidence}%")
        c3.metric("Strategy", strategy_label)
        c4.metric("Candle", trigger_at)

        if exec_status == "executed":
            st.success("**Order executed** on eToro demo.")
        elif exec_status == "skipped":
            st.error(f"**Not executed** — {exec_reason or 'blocked by engine gate'}")
        elif exec_status == "not_applicable":
            st.info(exec_reason or "No order — signal was HOLD.")
        elif decision in ("BUY", "SELL", "CLOSE"):
            st.caption("Execution outcome not recorded yet (signal may pre-date this feature).")
        # Bot provenance
        raw_uuid = rec.get("bot_id", "")
        if raw_uuid:
            st.caption(f"Bot ID: `{raw_uuid}`")

        reasoning = rec.get("reasoning")
        if reasoning:
            st.markdown("**Reasoning**")
            st.markdown(f"> {reasoning}")

        obs = rec.get("observations")
        if obs:
            obs_str = "\n".join(f"- {o}" for o in obs) if isinstance(obs, list) else str(obs)
            st.markdown("**Observations**")
            st.info(obs_str)

        # Extra metadata row
        meta: list[str] = []
        if rec.get("expected_direction_next"):
            meta.append(f"Direction: **{rec['expected_direction_next']}**")
        if rec.get("trend_strength"):
            meta.append(f"Trend: **{rec['trend_strength']}**")
        if rec.get("nearest_support"):
            meta.append(f"Support: {rec['nearest_support']}")
        if rec.get("nearest_resistance"):
            meta.append(f"Resistance: {rec['nearest_resistance']}")
        if rec.get("risk_level"):
            meta.append(f"Risk: **{rec['risk_level']}**")
        if meta:
            st.caption("  ·  ".join(meta))

        # Execution quality row (present for all non-LLM signals and enriched LLM signals)
        net_edge = rec.get("net_edge_pct")
        slippage = rec.get("slippage_pct")
        exec_risk = rec.get("exec_risk")
        viable = rec.get("viable")
        if net_edge is not None and slippage is not None:
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(exec_risk or "", "⚪")
            viability = "✅ tradeable" if viable else "⛔ low edge"
            st.caption(
                f"📊 Exec quality · Slippage {slippage:.3f}% · "
                f"Net edge {net_edge:+.3f}% · "
                f"{risk_icon} {exec_risk} · {viability}"
            )

        risk_warning = rec.get("risk_warning")
        if risk_warning:
            st.warning(f"⚠️ {risk_warning}")


_BOT_TRADES_PAGE = 50


def _parse_journal_dt(val) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        return None


def _journal_close_local_date(r: dict):
    dt = _parse_journal_dt(r.get("exit_time") or r.get("ts"))
    if not dt:
        return None
    loc = timez.to_local(dt)
    return loc.date() if loc else None


def _filter_journal_period(
    rows: list[dict],
    mode: str,
    *,
    custom_start=None,
    custom_end=None,
) -> list[dict]:
    """Keep journal rows whose close date (display TZ) falls in the period."""
    if mode == "All time":
        return list(rows)
    today = datetime.now(timez.active_tz()).date()
    out: list[dict] = []
    for r in rows:
        d = _journal_close_local_date(r)
        if d is None:
            continue
        if mode == "Today":
            if d != today:
                continue
        elif mode == "Yesterday":
            if d != today - timedelta(days=1):
                continue
        elif mode == "7 days":
            if d < today - timedelta(days=6):
                continue
        elif mode == "30 days":
            if d < today - timedelta(days=29):
                continue
        elif mode == "Custom":
            if custom_start and d < custom_start:
                continue
            if custom_end and d > custom_end:
                continue
        out.append(r)
    return out


def _today_bot_pnl_snapshot() -> dict:
    """Bot-only P&L for the current calendar day (display timezone).

    Realised = journal closes dated today.  Unrealised = current uPnL on open
    bot-owned positions (total mark-to-market, not today's delta only)."""
    all_rows = trade_journal.closed_records()
    today_rows = _filter_journal_period(all_rows, "Today")
    realised = sum(float(r.get("pnl_dollars") or 0) for r in today_rows)
    positions = positions_cache.get_positions()
    if not positions:
        positions = st.session_state.get("_gs_positions_snap", [])
    bot_pos = trade_manager.bot_owned_positions(positions)
    unrealised = sum(float(p.get("pnl") or 0) for p in bot_pos)
    wins = sum(1 for r in today_rows if r.get("win"))
    return {
        "realised_today": realised,
        "unrealised_open": unrealised,
        "net": realised + unrealised,
        "closed_n": len(today_rows),
        "open_n": len(bot_pos),
        "wins": wins,
        "losses": len(today_rows) - wins,
        "period_lbl": _journal_period_label("Today"),
    }


def _render_today_bot_pnl_banner() -> None:
    """Top-of-P&L summary: did the bots make money today (closed + open)?"""
    snap = _today_bot_pnl_snapshot()
    lbl = snap["period_lbl"]
    net = snap["net"]
    st.markdown("#### Today — bots only")
    st.caption(
        f"**{lbl}** · excludes manual & eToro-imported trades · "
        "times in your display timezone"
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Net P&L",
        f"${net:,.2f}",
        delta=f"{net:+,.2f}",
        help="Realised today + unrealised on open bot positions (right now)",
    )
    c2.metric(
        "Realised (closed)",
        f"${snap['realised_today']:,.2f}",
        delta=f"{snap['realised_today']:+,.2f}",
        help=f"P&L from {snap['closed_n']} trade(s) closed today",
    )
    c3.metric(
        "Unrealised (open)",
        f"${snap['unrealised_open']:,.2f}",
        delta=f"{snap['unrealised_open']:+,.2f}",
        help=f"Mark-to-market on {snap['open_n']} open bot position(s) — total uPnL, not today's move only",
    )
    c4.metric("Closed today", snap["closed_n"])
    c5.metric("Open now", snap["open_n"])
    if snap["closed_n"]:
        wr = snap["wins"] / snap["closed_n"] * 100
        st.caption(
            f"Closed today: **{snap['wins']}W / {snap['losses']}L** ({wr:.0f}% win rate)"
        )
    elif snap["open_n"] == 0:
        st.caption("No bot trades closed or open yet today.")
    st.divider()


def _journal_period_label(mode: str, *, custom_start=None, custom_end=None) -> str:
    today = datetime.now(timez.active_tz()).date()
    if mode == "Today":
        return f"today ({today.strftime('%b %d')})"
    if mode == "Yesterday":
        y = today - timedelta(days=1)
        return f"yesterday ({y.strftime('%b %d')})"
    if mode == "7 days":
        return "last 7 days"
    if mode == "30 days":
        return "last 30 days"
    if mode == "Custom" and custom_start and custom_end:
        return f"{custom_start} → {custom_end}"
    if mode == "Custom":
        return "custom range"
    return "all time"


def _display_day_start_utc_date(local_day: date) -> date:
    """Map midnight on *local_day* (display TZ) to the UTC calendar date eToro expects.

    eToro history minDate is UTC-oriented.  Timestamps are UTC; we convert to the
    display zone for period filters.  The API minDate must be the UTC date at the
    start of the period in that zone — not the local calendar date string."""
    start = datetime.combine(local_day, dt_time.min, tzinfo=timez.active_tz())
    return start.astimezone(timezone.utc).date()


def _period_logical_start(mode: str, *, custom_start=None) -> date:
    """First calendar day of a History/P&L period in the display timezone."""
    today = datetime.now(timez.active_tz()).date()
    if mode == "Custom" and custom_start:
        return max(ALL_HISTORY_START, custom_start)
    if mode == "Today":
        return today
    if mode == "Yesterday":
        return today - timedelta(days=1)
    if mode == "7 days":
        return today - timedelta(days=6)
    if mode == "30 days":
        return today - timedelta(days=29)
    return ALL_HISTORY_START


def _period_min_fetch_date(
    mode: str,
    *,
    custom_start=None,
) -> datetime.date:
    """Earliest minDate for eToro history — UTC date at period start (display TZ)."""
    if mode == "All time":
        return ALL_HISTORY_START
    if mode == "Custom" and not custom_start:
        return ALL_HISTORY_START
    logical = _period_logical_start(mode, custom_start=custom_start)
    return max(ALL_HISTORY_START, _display_day_start_utc_date(logical))


def render_bot_session_trades(
    *,
    period_mode: str = _DEFAULT_PERIOD,
    custom_start=None,
    custom_end=None,
) -> None:
    """Closed trades from the durable journal — survives restarts.

    The journal is the source of truth (the in-memory session list reset on every
    dashboard restart, which is why this used to show only a handful).  We overlay
    the session's richer LLM exit rationale onto matching rows when available."""
    all_rows = trade_journal.closed_records()
    if not all_rows:
        st.caption("No closed trades recorded yet.")
        return

    mode = period_mode if period_mode in _PNL_PERIOD_OPTIONS else _DEFAULT_PERIOD
    rows = _filter_journal_period(
        all_rows, mode, custom_start=custom_start, custom_end=custom_end,
    )
    period_lbl = _journal_period_label(
        mode, custom_start=custom_start, custom_end=custom_end,
    )

    if not rows:
        st.caption(f"No closed trades in this period ({period_lbl}).")
        if mode == "Today":
            y_rows = _filter_journal_period(all_rows, "Yesterday")
            if y_rows:
                y_total = sum(float(r.get("pnl_dollars") or 0) for r in y_rows)
                st.caption(
                    f"**Yesterday:** {len(y_rows)} trade(s) · "
                    f"${y_total:+,.2f} realised — switch period above to view."
                )
        return

    # Session trades carry the full LLM rationale/observations that the journal
    # doesn't persist — index them by eToro position id to enrich matching rows.
    sess_by_pid: dict[str, object] = {}
    for s in trade_manager.get_closed():
        pid = getattr(s, "etoro_position_id", None)
        if pid is not None:
            sess_by_pid[str(pid)] = s

    total  = sum(float(r.get("pnl_dollars") or 0) for r in rows)
    wins   = sum(1 for r in rows if r.get("win"))
    losses = sum(1 for r in rows if not r.get("win"))
    wr     = wins / len(rows) * 100 if rows else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Closed ({period_lbl})", len(rows))
    m2.metric("Realised P&L", f"${total:,.2f}", delta=f"{total:+,.2f}")
    m3.metric("Win Rate", f"{wr:.0f}%")
    m4.metric("W / L", f"{wins} / {losses}")

    if mode == "Today":
        y_rows = _filter_journal_period(all_rows, "Yesterday")
        if y_rows:
            y_total = sum(float(r.get("pnl_dollars") or 0) for r in y_rows)
            y_wins  = sum(1 for r in y_rows if r.get("win"))
            st.caption(
                f"**Yesterday** ({len(y_rows)} trades · ${y_total:+,.2f} realised · "
                f"{y_wins}W) — select **Yesterday** above for full detail."
            )

    shown = st.session_state.get("_bot_trades_shown", _BOT_TRADES_PAGE)
    shown = min(shown, len(rows))

    for r in rows[:shown]:
        pnl_d   = float(r.get("pnl_dollars") or 0)
        pnl_pct = float(r.get("pnl_pct") or 0)
        sign    = "▲" if pnl_d >= 0 else "▼"
        label   = (
            r.get("instrument_label") or _label_for_instrument_id(r.get("instrument_id"))
        )
        strat   = r.get("strategy") or "manual"
        reason  = (r.get("reason") or "").upper()
        exit_dt = timez.fmt_iso(r.get("exit_time") or r.get("ts"), "%m-%d %H:%M")

        with st.expander(
            f"{sign} {label} · {r.get('direction','')} · "
            f"{'%+.2f' % pnl_d} · {strat} · {reason} · {exit_dt}",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Entry", _fmt_price(r.get("entry_price")))
            c2.metric("Exit",  _fmt_price(r.get("exit_price")))
            c3.metric("P&L",   f"${pnl_d:+,.2f}", delta=f"{pnl_pct:+.2f}%")
            c4.metric("Confidence", f"{int(r.get('confidence') or 0)}%")

            st.caption(
                f"Strategy **{strat}** · holding {float(r.get('holding_min') or 0):.0f}m"
            )

            sess = sess_by_pid.get(str(r.get("etoro_position_id")))
            if sess is not None and getattr(sess, "reason", "") == "llm" and (
                getattr(sess, "llm_reasoning", "") or getattr(sess, "llm_observations", "")
            ):
                st.markdown("**🤖 LLM Exit Rationale**")
                if getattr(sess, "llm_reasoning", ""):
                    st.markdown(f"> {sess.llm_reasoning}")
                if getattr(sess, "llm_observations", ""):
                    st.markdown("**Observations**")
                    st.info(sess.llm_observations)
            elif r.get("reason") == "stop_loss":
                st.caption("🛑 Stop-loss exit.")
            elif r.get("reason") == "manual":
                st.caption("👤 Closed manually (by you).")
            elif r.get("reason") == "external":
                st.caption(
                    "⚠ Closed on eToro's side — its own stop-loss/take-profit, the "
                    "eToro app, or a position merge. The bot detected it via "
                    "portfolio reconciliation, it did not issue this exit."
                )
            elif (r.get("signal") or "").upper() == "ADOPTED":
                st.caption(
                    "🔗 Adopted position — the bot took over an already-open eToro "
                    "position (it didn't open it from a fresh signal) and tracked it "
                    "to close."
                )
            elif r.get("signal"):
                st.caption(f"Entry signal: {r.get('signal')}")

    if shown < len(rows):
        if st.button(f"Show more ({len(rows) - shown} older)", key="_bot_trades_more"):
            st.session_state["_bot_trades_shown"] = shown + _BOT_TRADES_PAGE
            st.rerun()


def bump_portfolio_cache() -> None:
    st.session_state["portfolio_rev"] = st.session_state.get("portfolio_rev", 0) + 1
    positions_cache.invalidate()


def load_etoro_positions(demo: bool, *, force: bool = False) -> list[dict]:
    """Read shared position cache (engine is primary poller when live)."""
    cached = positions_cache.get_positions()
    if cached and not force:
        return cached
    return positions_cache.refresh_if_stale(client, demo, force=force)


def _fmt_price(val: float | None) -> str:
    return f"{val:.5f}" if val is not None else "—"


def _fmt_money(val: float | None) -> str:
    return vtables.fmt_money(val)


def _instrument_label(p: dict) -> str:
    name   = p.get("name") or ""
    symbol = p.get("symbol") or ""
    iid    = p.get("instrument_id")
    if name and symbol:
        return f"{name} ({symbol})"
    if name:
        return name
    if symbol:
        return symbol
    return str(iid or "—")


def _label_for_instrument_id(instrument_id: int | None) -> str:
    if instrument_id is None:
        return "—"
    id_map = globals().get("INSTRUMENT_ID_TO_LABEL") or {}
    if instrument_id in id_map:
        return id_map[instrument_id]
    for lbl, iid in (globals().get("ALL_INSTRUMENTS") or {}).items():
        if iid == instrument_id:
            return lbl
    return str(instrument_id)


def _history_fetch_key(is_demo: bool, min_date: datetime.date) -> str:
    return f"{'demo' if is_demo else 'real'}:{min_date.isoformat()}"


def _ensure_history_trades(
    is_demo: bool,
    min_date: datetime.date,
    *,
    force: bool = False,
) -> list[dict] | None:
    """Load closed trades; cache by account + from-date. None when load failed."""
    key = _history_fetch_key(is_demo, min_date)
    if force:
        st.session_state.pop("etoro_hist_trades", None)
        st.session_state.pop("etoro_hist_key", None)
        st.session_state.pop("hist_load_error", None)

    if (
        not force
        and st.session_state.get("etoro_hist_key") == key
        and st.session_state.get("etoro_hist_trades") is not None
    ):
        return st.session_state["etoro_hist_trades"]

    try:
        trades = fetch_all_etoro_trade_history(min_date, demo=is_demo)
        st.session_state["etoro_hist_trades"] = trades
        st.session_state["etoro_hist_key"] = key
        st.session_state.pop("hist_load_error", None)
        return trades
    except PermissionError:
        st.session_state["hist_load_error"] = "permission"
        st.session_state.pop("etoro_hist_trades", None)
        st.session_state.pop("etoro_hist_key", None)
        return None
    except Exception as exc:
        st.session_state["hist_load_error"] = str(exc)
        return None


def format_etoro_history_df(trades: list[dict]) -> pd.DataFrame:
    """Turn eToro /trade/demo/history rows into a readable table."""
    rows = []
    for t in trades:
        iid = t.get("instrumentId") or t.get("instrument_id")
        try:
            iid_int = int(iid) if iid is not None else None
        except (TypeError, ValueError):
            iid_int = None

        is_buy = t.get("isBuy")
        if is_buy is True:
            direction = "LONG"
        elif is_buy is False:
            direction = "SHORT"
        else:
            direction = "—"

        open_rate = t.get("openRate")
        close_rate = t.get("closeRate")
        pnl = t.get("netProfit")

        stock = display_asset_name(_label_for_instrument_id(iid_int))
        rows.append({
            "Stock":      stock,
            "Side":       "Bought" if is_buy is True else ("Sold" if is_buy is False else "—"),
            "Direction":  direction,
            "Open @":     f"{float(open_rate):.5f}" if open_rate is not None else "—",
            "Close @":    f"{float(close_rate):.5f}" if close_rate is not None else "—",
            "P&L":        f"${float(pnl):+,.2f}" if pnl is not None else "—",
            "Invested":   f"${float(t['investment']):,.2f}"
                          if t.get("investment") is not None else "—",
            "Units":      t.get("units"),
            "Opened":     vtables.parse_api_timestamp(t.get("openTimestamp")),
            "Closed":     vtables.parse_api_timestamp(t.get("closeTimestamp")),
            "Position #": t.get("positionId"),
        })
    return pd.DataFrame(rows)


# History floor: never show (or fetch) closed trades before this date.  "All time"
# starts here, and the Custom picker can't go earlier.
ALL_HISTORY_START = datetime(2025, 6, 9).date()


def _etoro_history_close_local_date(t: dict):
    """Close date of an eToro history row in the user's display timezone."""
    return _etoro_row_close_local_date(t)


def _filter_etoro_history_period(
    trades: list[dict],
    mode: str,
    *,
    custom_start=None,
    custom_end=None,
) -> list[dict]:
    """Keep eToro history trades whose close date (display TZ) falls in the period."""
    if mode == "All time":
        return list(trades)
    today = datetime.now(timez.active_tz()).date()
    out: list[dict] = []
    for t in trades:
        d = _etoro_history_close_local_date(t)
        if d is None:
            continue
        if mode == "Today":
            if d != today:
                continue
        elif mode == "Yesterday":
            if d != today - timedelta(days=1):
                continue
        elif mode == "7 days":
            if d < today - timedelta(days=6):
                continue
        elif mode == "30 days":
            if d < today - timedelta(days=29):
                continue
        elif mode == "Custom":
            if custom_start and d < custom_start:
                continue
            if custom_end and d > custom_end:
                continue
        out.append(t)
    return out


def fetch_all_etoro_trade_history(
    min_date: datetime.date,
    *,
    demo: bool,
) -> list[dict]:
    """Load every closed trade from eToro (all pages, deduped)."""
    return client.get_all_trade_history(
        min_date.strftime("%Y-%m-%d"),
        demo=demo,
    )


def _render_open_positions_history(positions: list[dict], is_demo: bool) -> None:
    st.markdown(
        '<p class="pf-section-title">Open positions (not closed yet)</p>',
        unsafe_allow_html=True,
    )
    acct = "Demo" if is_demo else "Real"
    live_rows = [_enrich_position_live(dict(p)) for p in positions]
    if not live_rows:
        st.caption(f"No open positions on eToro {acct} account")
        return

    with st.container(border=True):
        st.html(vtables.open_positions_history_html(live_rows))


def render_unified_history(
    etoro_trades: list[dict],
    open_positions: list[dict],   # kept for signature compatibility; History shows closed only
    *,
    is_demo: bool,
    period_lbl: str = "this period",
) -> None:
    # History tab = closed trades only.  Open positions live in the Portfolio tab.
    if not etoro_trades:
        acct = "demo" if is_demo else "real"
        st.info(f"No closed trades on your eToro {acct} account for **{period_lbl}**.")
    else:
        _render_etoro_history_results(etoro_trades)


def _enrich_history_close_methods(trades: list[dict]) -> None:
    """Attach journal exit reason to each eToro history row (by position id)."""
    import trade_journal

    meta = trade_journal.position_close_meta_map()
    # In-memory closes from this session (before journal reload on another worker).
    for ct in trade_manager.get_closed():
        pid = ct.etoro_position_id
        if pid is not None and str(pid) not in meta:
            meta[str(pid)] = {
                "reason": (ct.reason or "").strip(),
                "strategy": (getattr(ct, "strategy", "") or "").strip(),
            }
    for t in trades:
        pid = t.get("positionId") or t.get("position_id") or t.get("positionID")
        row = meta.get(str(pid)) if pid is not None else None
        t["_close_reason"] = (row or {}).get("reason", "")
        t["_close_strategy"] = (row or {}).get("strategy", "")


def _render_etoro_history_results(trades: list[dict]) -> None:
    # Partial cash-freeing trims share one open line but get distinct position ids —
    # propagate ownership across those clusters before labelling.
    trade_manager.propagate_cluster_owners(trades)
    _enrich_history_close_methods(trades)
    _uuid_to_key = {v: k for k, v in bot_registry.get_all().items()}
    for t in trades:
        _uuid = trade_manager.resolve_history_owner(t)
        _key = _uuid_to_key.get(_uuid) if _uuid else None
        t["_owner"] = _bot_display_name(_key) if _key else "Manual"

    # ── Bot / Manual filter ───────────────────────────────────────────────────
    _bot_n    = sum(1 for t in trades if t.get("_owner") != "Manual")
    _manual_n = len(trades) - _bot_n
    _counts   = {"All": len(trades), "Bots": _bot_n, "Manual": _manual_n}
    _flt = st.segmented_control(
        "Show",
        options=["All", "Bots", "Manual"],
        format_func=lambda o: f"{o} ({_counts[o]})",
        default="All",
        key="hist_owner_filter",
        label_visibility="collapsed",
    )
    shown = trades
    if _flt == "Bots":
        shown = [t for t in trades if t.get("_owner") != "Manual"]
    elif _flt == "Manual":
        shown = [t for t in trades if t.get("_owner") == "Manual"]

    # ── Re-attribute orphaned closed trades ───────────────────────────────────
    if _manual_n:
        if st.button(
            f"Re-attribute {_manual_n} unlabelled trade(s) to their bot",
            key="hist_backfill_owners",
            help="Match each Manual closed trade to the bot that opened it "
                 "(same instrument, direction and open time vs the signal log).",
        ):
            assigned = (
                trade_manager.backfill_all_owners(client) if client else {}
            )
            if assigned:
                st.success(f"Re-attributed {len(assigned)} position(s)/trade(s) to their bot.")
            else:
                st.info("No matches found — remaining trades look genuinely manual.")
            st.rerun()

    with st.container(border=True):
        st.html(vtables.closed_trades_block_html(shown))


def _pnl_indicator(p: dict) -> str:
    pnl = p.get("pnl")
    if pnl is None:
        return "—"
    if pnl > 0.005:
        return f"▲ {_fmt_money(pnl)}"
    if pnl < -0.005:
        return f"▼ {_fmt_money(pnl)}"
    return f"● {_fmt_money(pnl)}"


def fetch_positions_safe(demo: bool, *, force: bool = False) -> list[dict]:
    try:
        return load_etoro_positions(demo, force=force)
    except PermissionError:
        return []
    except Exception:
        return []


def _render_sidebar_system(is_demo: bool) -> None:
    """Sidebar tail — must run after position helpers are defined."""
    with st.sidebar:
        with st.expander("System status", expanded=False):
            # Read ONLY from background-updated caches — never do network I/O on
            # the render path.  The visual-bot health is probed on a daemon
            # thread; the position count comes from the background positions
            # poller.  This keeps tab switches instant (no synchronous GET/REST).
            _ensure_vbot_poller()
            cached   = positions_cache.get_positions()
            _pos_n   = len(cached) if cached else 0
            _vbot_ok = _VBOT_OK
            cfg = trading_engine.get_config()
            iid = st.session_state.get("engine_instrument_id") or (
                cfg.instrument_id if cfg else None
            )
            badge = ws_badge(iid) if iid else "—"
            ui.sidebar_status(
                _vbot_ok,
                _pos_n,
                engine_on=trading_engine.is_running(),
                trading_on=trading_engine.is_trading_active(),
                feed_live=st.session_state.get("feed_live", False),
                ws_badge=badge,
                auto_trade_count=trading_engine.auto_trade_count(),
            )


def fetch_positions_ui(demo: bool) -> list[dict]:
    """Live UI reads shared cache — no extra REST calls."""
    cached = positions_cache.get_positions()
    return cached if cached else fetch_positions_safe(demo)


_PLOTLY_UI_CONFIG = {"displayModeBar": False, "scrollZoom": True}


def positions_for_instrument(positions: list[dict], instrument_id: int) -> list[dict]:
    return [p for p in positions if p.get("instrument_id") == instrument_id]


def _active_bot_uuid() -> str:
    """UUID of the bot the Trading tab is bound to (empty if none)."""
    active_key = _active_bot_key()
    if not active_key:
        return ""
    snap = trading_engine.get_snapshot(bot_id=active_key)
    if snap and snap.bot_uuid:
        return snap.bot_uuid
    return bot_registry.get(active_key) or ""


def _active_bot_open_trade(instrument_id: int):
    """The open PaperTrade held by the bot the Trading tab is bound to, or None.

    Trades are bot-keyed, so this is just that bot's own trade — never a sibling
    or manual position on the same instrument.
    """
    bot_uuid = _active_bot_uuid()
    if not bot_uuid:
        return None
    return trade_manager.get_open(bot_uuid)


def positions_owned_by_active_bot(
    positions: list[dict], instrument_id: int
) -> list[dict]:
    """eToro positions opened by the bot the Trading tab is bound to.

    Positions are per-instrument on eToro, but the Trading tab follows ONE bot —
    so the Open Positions panel must show only the position THAT bot opened, not
    every position on the stock.  We match the bot's tracked trade (by UUID
    ownership) to the live eToro position via etoro_position_id.  Returns [] when
    the active bot holds no position, even if other bots/manual trades exist on
    the same instrument.
    """
    inst = positions_for_instrument(positions, instrument_id)
    if not inst:
        return []

    trade = _active_bot_open_trade(instrument_id)
    if trade is None:
        return []

    pid = trade.etoro_position_id
    if pid is not None:
        match = [p for p in inst if p.get("position_id") == pid]
        return match  # empty if eToro hasn't surfaced it yet — never show a sibling's
    # Position ID not resolved yet (just opened): with one-position-per-instrument
    # tracking, the single instrument position is this bot's.
    return inst[:1]


def _position_status_badge(p: dict) -> tuple[str, str]:
    pnl = p.get("pnl") or 0
    if pnl > 0.005:
        return "gain", "Gaining"
    if pnl < -0.005:
        return "loss", "Losing"
    return "flat", "Flat"


def apply_live_ticks_to_position(p: dict, ask: float, bid: float) -> dict:
    """Recompute Now / P&L from WebSocket ticks (updates every fragment refresh)."""
    out = dict(p)
    amount = float(out.get("amount") or 0)
    open_rate = float(out.get("open_rate") or 0)
    units = float(out.get("units") or 0)
    if not units and amount and open_rate:
        units = amount / open_rate

    is_short = out.get("direction") == "SHORT" or out.get("is_buy") is False
    if is_short and ask and units:
        current_value = units * ask
        pnl = amount - current_value
        out["current_rate"] = ask
        out["current_value"] = current_value
        out["pnl"] = pnl
        if amount:
            out["pnl_pct"] = (pnl / amount) * 100
    elif not is_short and bid and units:
        current_value = units * bid
        pnl = current_value - amount
        out["current_rate"] = bid
        out["current_value"] = current_value
        out["pnl"] = pnl
        if amount:
            out["pnl_pct"] = (pnl / amount) * 100

    pnl_val = out.get("pnl") or 0
    if pnl_val > 0.005:
        out["status"] = "Gaining"
    elif pnl_val < -0.005:
        out["status"] = "Losing"
    else:
        out["status"] = "Flat"
    out["_live_ticks"] = True
    return out


def render_position_card(
    p: dict, *, live: bool = False, show_symbol: bool = False,
) -> None:
    """Compact dark-theme position card."""
    direction = (p.get("direction") or "—").upper()
    side_cls  = "short" if direction == "SHORT" else "long"
    dir_cls   = "short" if direction == "SHORT" else "long"
    badge_cls, badge_lbl = _position_status_badge(p)
    pnl = p.get("pnl")
    pct = p.get("pnl_pct")
    pnl_col = ui.pnl_color(pnl)
    pnl_amt = _fmt_money(pnl) if pnl is not None else "—"
    pct_txt = f"{pct:+.2f}%" if pct is not None else ""
    live_tag = (
        f'<span style="color:{ui.C_LIVE};font-size:0.6rem;margin-left:4px">● live</span>'
        if live or p.get("_live_ticks") else ""
    )
    symbol = (p.get("symbol") or "").strip()
    name = (p.get("name") or "").strip()
    asset_hdr = ""
    if show_symbol and (symbol or name):
        asset_hdr = (
            f'<div style="font-size:0.8rem;font-weight:600;color:{ui.C_TEXT};'
            f'margin-bottom:6px">{symbol or name}'
            f'{f" · {name}" if symbol and name else ""}</div>'
        )

    # Resolve which bot owns this position — show UUID prefix + key + interval + strategy
    iid = p.get("instrument_id")
    bot_tag_html = ""
    if iid:
        owned_trade = trade_manager.find_open_by_position_id(p.get("position_id"))
        if owned_trade and owned_trade.bot_id:
            trade_uuid = owned_trade.bot_id   # now a UUID string
            snap_by_uuid = trading_engine.get_snapshot_by_uuid(trade_uuid)
            if snap_by_uuid:
                interval_short = snap_by_uuid.interval_label.replace(" Minutes", "m").replace(" Minute", "m")
                eng_cfg = trading_engine.get_config(bot_id=snap_by_uuid.bot_id)
                strat_label = (
                    strategies.display_names().get(eng_cfg.strategy_name, eng_cfg.strategy_name.upper())
                    if eng_cfg else "—"
                )
                bot_desc = f"{trade_uuid[:8]} · {snap_by_uuid.bot_id} · {interval_short} · {strat_label}"
            else:
                bot_desc = trade_uuid[:8]
            bot_tag_html = (
                f'<span style="margin-left:8px;padding:1px 6px;border-radius:4px;'
                f'background:{ui.C_ACCENT}22;color:{ui.C_ACCENT};'
                f'font-size:0.65rem;font-weight:600">🤖 {bot_desc}</span>'
            )

    # Trailing stop display — look up from the local PaperTrade if owned
    trail_html = ""
    if iid:
        _local_trade = trade_manager.find_open_by_position_id(p.get("position_id"))
        if _local_trade and _local_trade.peak_pnl > 0:
            _eng_cfg = None
            if owned_trade and owned_trade.bot_id:
                _snap_t = trading_engine.get_snapshot_by_uuid(owned_trade.bot_id)
                if _snap_t:
                    _eng_cfg = trading_engine.get_config(bot_id=_snap_t.bot_id)
            trail_pct = _eng_cfg.trailing_stop_pct if _eng_cfg else 1.5
            if trail_pct > 0:
                trail_price = trade_manager.trailing_stop_trigger_price(_local_trade, trail_pct)
                if trail_price:
                    trail_html = (
                        f'<div class="pos-stat"><div class="lbl">Trail stop</div>'
                        f'<div class="val" style="color:{ui.C_LIVE}">'
                        f'{_fmt_price(trail_price)}</div></div>'
                    )

    invested = f"${p['amount']:,.0f}" if p.get("amount") else "—"
    st.html(
        f"""<div class="pos-card {side_cls}">
          {asset_hdr}
          <div class="pos-head">
            <span class="pos-dir {dir_cls}">{direction}</span>
            <span class="pos-badge {badge_cls}">{badge_lbl}</span>
          </div>
          <div class="pos-grid">
            <div class="pos-stat"><div class="lbl">Open</div>
              <div class="val">{_fmt_price(p.get("open_rate"))}</div></div>
            <div class="pos-stat"><div class="lbl">Now{live_tag}</div>
              <div class="val">{_fmt_price(p.get("current_rate"))}</div></div>
            <div class="pos-stat"><div class="lbl">Invested</div>
              <div class="val">{invested}</div></div>
            <div class="pos-stat"><div class="lbl">Stop</div>
              <div class="val">{_fmt_price(p.get("stop_loss"))}</div></div>
            {trail_html}
          </div>
          <div class="pos-pnl">
            <span class="amt" style="color:{pnl_col}">{pnl_amt}</span>
            <span class="pct">{pct_txt}</span>
          </div>
          <div style="margin-top:8px;font-size:0.68rem;color:{ui.C_LABEL}">
            Position #{p.get("position_id") or "—"}{bot_tag_html}
          </div>
        </div>"""
    )


def render_etoro_positions_panel(
    positions: list[dict],
    is_demo: bool,
    *,
    filter_instrument_id: int | None = None,
    instrument_label: str | None = None,
    compact: bool = False,
    live_ask: float | None = None,
    live_bid: float | None = None,
) -> None:
    """Show open positions from eToro. Filter by instrument when on Charts tab."""
    acct = "Demo" if is_demo else "Real"

    if filter_instrument_id is not None:
        positions = positions_for_instrument(positions, filter_instrument_id)

    n = len(positions)
    if n == 0:
        if filter_instrument_id is not None:
            st.caption(
                f"No open positions for **{instrument_label or filter_instrument_id}** "
                f"on eToro {acct}"
            )
        else:
            st.caption(f"No open positions on eToro {acct} account")
        return

    total_pnl = sum(p.get("pnl") or 0 for p in positions)
    gaining   = sum(1 for p in positions if (p.get("pnl") or 0) > 0.005)
    losing    = sum(1 for p in positions if (p.get("pnl") or 0) < -0.005)

    if not compact:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Positions", n)
        m2.metric("Unrealised P&L", _fmt_money(total_pnl))
        m3.metric("Gaining / Losing", f"{gaining} / {losing}")
        m4.metric("Account", f"eToro {acct}")
    if compact:
        use_ticks = (
            live_ask is not None and live_bid is not None
            and (live_ask > 0 or live_bid > 0)
        )
        display = []
        for p in positions:
            row = (
                apply_live_ticks_to_position(p, live_ask, live_bid)
                if use_ticks else p
            )
            display.append(row)
        total_pnl = sum(r.get("pnl") or 0 for r in display)
        pnl_col = ui.pnl_color(total_pnl)
        live_note = " · tick P&amp;L" if use_ticks else ""
        st.html(
            f'<p style="font-size:0.75rem;color:{ui.C_MUTED};margin:0 0 8px 0">'
            f'<span style="color:{pnl_col};font-weight:600">{_fmt_money(total_pnl)}</span>'
            f" unrealised · {n} open{live_note}</p>"
        )
        for row in display:
            render_position_card(row, live=use_ticks)
        return

    show_instrument_col = filter_instrument_id is None
    rows = []
    for p in positions:
        pct = p.get("pnl_pct")
        row: dict = {
            "Direction":   p.get("direction", "—"),
            "Status":      p.get("status", "—"),
            "Invested":    f"${p['amount']:,.2f}" if p.get("amount") else "—",
            "Open @":      _fmt_price(p.get("open_rate")),
            "Now @":       _fmt_price(p.get("current_rate")),
            "Value now":   _fmt_money(p.get("current_value")),
            "P&L":         _pnl_indicator(p),
            "P&L %":       f"{pct:+.2f}%" if pct is not None else "—",
            "Stop":        _fmt_price(p.get("stop_loss")),
            "Position #":  p.get("position_id") or "—",
        }
        if show_instrument_col:
            row = {"Instrument": _instrument_label(p), **row}
        rows.append(row)

    _render_table(pd.DataFrame(rows))


def _enrich_position_for_display(
    p: dict,
    chart_instrument_id: int | None,
    chart_ask: float,
    chart_bid: float,
) -> dict:
    iid = p.get("instrument_id")
    if chart_instrument_id and iid == chart_instrument_id and (chart_ask or chart_bid):
        return apply_live_ticks_to_position(p, chart_ask, chart_bid)
    quote = tick_manager.get_latest_quote(iid) if iid else None
    if quote:
        return apply_live_ticks_to_position(p, quote[0], quote[1])
    return dict(p)


def render_open_positions_below_chart(
    positions: list[dict],
    is_demo: bool,
    *,
    chart_instrument_id: int | None = None,
    live_ask: float = 0.0,
    live_bid: float = 0.0,
    key_prefix: str = "chart_pos",
    instrument_label: str | None = None,
    show_close: bool = False,
) -> None:
    """Full-width open positions under the chart for the selected instrument only."""
    acct = "Demo" if is_demo else "Real"
    if chart_instrument_id is not None:
        positions = positions_for_instrument(positions, chart_instrument_id)

    n = len(positions)
    if n == 0:
        sym = instrument_label or chart_instrument_id or "this instrument"
        st.caption(f"No open positions for **{sym}** on eToro {acct}")
        return

    use_ticks = live_ask > 0 or live_bid > 0
    display = [
        _enrich_position_for_display(p, chart_instrument_id, live_ask, live_bid)
        for p in positions
    ]
    total_pnl = sum(r.get("pnl") or 0 for r in display)
    pnl_col = ui.pnl_color(total_pnl)
    live_note = " · tick P&amp;L" if use_ticks else ""
    st.html(
        f'<p style="font-size:0.75rem;color:{ui.C_MUTED};margin:0 0 10px 0">'
        f'<span style="color:{pnl_col};font-weight:600">{_fmt_money(total_pnl)}</span>'
        f" unrealised · {n} open{live_note}</p>"
    )

    cols_per_row = min(3, n) if n > 1 else 1
    show_sym = n > 1
    for i in range(0, n, cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        for j, row in enumerate(display[i : i + cols_per_row]):
            pid = row.get("position_id")
            iid = row.get("instrument_id")
            with cols[j]:
                render_position_card(
                    row,
                    live=use_ticks and iid == chart_instrument_id,
                    show_symbol=show_sym,
                )
                if show_close and st.button(
                    "Close position",
                    key=f"{key_prefix}_close_{pid}",
                    width="stretch",
                    type="primary",
                    disabled=not is_demo,
                ):
                    if err := _close_portfolio_position(row, is_demo):
                        st.error(err)
                    else:
                        st.rerun()


_PF_COLS = [2.5, 1.4, 1.15, 1.0, 0.95, 0.9, 1.15, 0.85]


def _session_price_change(iid: int, current: float | None) -> tuple[float | None, float | None]:
    """Delta since the last portfolio refresh (2s), for live price movement."""
    if current is None or not iid:
        return None, None
    key = f"pf_prev_{iid}"
    prev = st.session_state.get(key)
    st.session_state[key] = current
    if prev is None:
        return None, None
    ch = current - prev
    return ch, (ch / prev) * 100 if prev else None


def _enrich_position_live(p: dict) -> dict:
    """Animate portfolio rows from REST rates (no per-position WebSocket)."""
    out = dict(p)
    iid = out.get("instrument_id")
    ch, ch_pct = _session_price_change(iid, out.get("current_rate"))
    if ch is not None:
        out["live_change"] = ch
        out["live_change_pct"] = ch_pct
    return out


def _close_portfolio_position(p: dict, is_demo: bool) -> str | None:
    """Close one eToro demo position and sync local bot trade state."""
    pid = p.get("position_id")
    iid = p.get("instrument_id")
    if pid is None or iid is None:
        return "Missing position or instrument id"
    if not is_demo:
        return "Close is only available on the Demo account via API"

    try:
        pid_int = int(pid)
        iid_int = int(iid)
    except (TypeError, ValueError):
        return "Invalid position id"

    rate = float(p.get("current_rate") or p.get("open_rate") or 0)
    ask = bid = rate

    try:
        # Full close only — omit UnitsToDeduct. Computed units (esp. crypto like
        # XRP) often differ slightly from eToro's internal amount and trigger
        # "unable to partially close" errors.
        client.close_demo_position(pid_int, iid_int)
        # Clear our local tracking for THIS specific position (eToro close already
        # done above, so pass client=None).
        if trade := trade_manager.find_open_by_position_id(pid_int):
            closed = trade_manager.close_manual(trade.bot_id, ask, bid, client=None)
            if closed:
                st.session_state[f"last_close_{iid_int}"] = closed
        # Optimistic cache update — do NOT invalidate()+force refresh here; that
        # blanked the table and blocked the Streamlit server on a slow REST call.
        positions_cache.remove_position(pid_int)
        trading_engine.suppress_adopt(iid_int)
        return None
    except Exception as exc:
        return str(exc)


def _on_portfolio_close(position_id: int, is_demo: bool) -> None:
    """on_click callback for the Portfolio ✕ button — queue only, no REST here.

    Running close_demo_position + a forced portfolio refresh inside on_click
    blocked the whole Streamlit server for several seconds and wiped the cache
    mid-flight, which looked like the tab crashed."""
    st.session_state["_pf_pending_close"] = int(position_id)


def _process_pending_portfolio_close(positions: list[dict], is_demo: bool) -> None:
    """Execute a queued Portfolio close on the render path (after rerun)."""
    pid = st.session_state.pop("_pf_pending_close", None)
    if pid is None:
        return
    p = next((x for x in positions if x.get("position_id") == pid), None)
    if not p:
        trade = trade_manager.find_open_by_position_id(pid)
        iid = trade.instrument_id if trade else None
        p = {"position_id": pid, "instrument_id": iid}
    with st.spinner(f"Closing position #{pid}…"):
        err = _close_portfolio_position(p, is_demo)
    if err:
        st.session_state["_pf_close_error"] = err


def _positions_for_chart_instrument(instrument_id: int, is_demo: bool) -> list[dict]:
    positions = positions_cache.get_positions()
    if not positions:
        positions = load_etoro_positions(is_demo)
    # Only the bound bot's position — the close button must not act on a sibling
    # bot's or a manual position on the same instrument.
    return positions_owned_by_active_bot(positions, instrument_id)


def render_chart_position_close(instrument_id: int, is_demo: bool) -> None:
    """Close control — must live outside auto-refresh fragments or clicks are lost."""
    if not is_demo:
        st.caption("Close is only available on the Demo account via API.")
        return

    positions = _positions_for_chart_instrument(instrument_id, is_demo)
    if not positions:
        return

    if len(positions) == 1:
        p = positions[0]
        sym = (p.get("symbol") or p.get("name") or instrument_id)
        direction = (p.get("direction") or "LONG").upper()
        if st.button(
            f"Close {sym} {direction} on eToro",
            key=f"chart_close_{instrument_id}_{p.get('position_id')}",
            type="primary",
            width="stretch",
        ):
            if err := _close_portfolio_position(p, is_demo):
                st.error(err)
            else:
                st.rerun()
        return

    opts: dict[str, dict] = {}
    for p in positions:
        sym = (p.get("symbol") or p.get("name") or "?").strip()
        direction = (p.get("direction") or "LONG").upper()
        pid = p.get("position_id") or "?"
        opts[f"{sym} · {direction} · #{pid}"] = p
    labels = ["— select position —", *opts.keys()]
    choice = st.selectbox(
        "Close position", labels, key=f"chart_close_pick_{instrument_id}",
        label_visibility="collapsed",
    )
    if st.button(
        "Close selected on eToro",
        key=f"chart_close_btn_{instrument_id}",
        type="primary",
        width="stretch",
        disabled=choice == labels[0],
    ):
        if err := _close_portfolio_position(opts[choice], is_demo):
            st.error(err)
        else:
            st.rerun()


def render_portfolio_close_select(positions: list[dict], is_demo: bool) -> None:
    if not positions or not is_demo:
        return
    opts: dict[str, dict] = {}
    for p in positions:
        sym = (p.get("symbol") or p.get("name") or "?").strip()
        direction = (p.get("direction") or "LONG").upper()
        pid = p.get("position_id") or "?"
        opts[f"{sym} · {direction} · #{pid}"] = p
    labels = ["— select position —", *opts.keys()]
    choice = st.selectbox("Close position", labels, key="pf_close_pick", label_visibility="collapsed")
    if st.button("Close selected", key="pf_close_btn", width="stretch", disabled=choice == labels[0]):
        if err := _close_portfolio_position(opts[choice], is_demo):
            st.error(err)
        else:
            st.rerun()


def _owner_label_for_position(p: dict, uuid_to_key: dict[str, str]) -> str:
    """Return the bot key that owns this eToro position, or 'Manual'.

    A position belongs to a bot when our tracked trade for that instrument links
    to it by etoro_position_id and carries the bot's UUID.  Anything else — a
    sibling/manual position, or one we never opened — is 'Manual'.
    """
    pid = p.get("position_id")
    if pid is None:
        return "Manual"
    # Match the exact position to its owning bot — by the live tracked trade, then
    # the persisted owner map (covers positions opened by a bot in a past session).
    trade = trade_manager.find_open_by_position_id(pid)
    uuid = trade.bot_id if (trade and trade.bot_id) else trade_manager.owner_of_position(pid)
    if uuid:
        key = uuid_to_key.get(uuid)
        return _bot_display_name(key) if key else "Bot"
    return "Manual"


_PF_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _open_dt(p: dict) -> datetime:
    """Parsed open timestamp (UTC) for a portfolio row, or the epoch when absent
    (so undated rows sort to the bottom)."""
    try:
        dt = trade_manager._parse_etoro_open_date(p)
    except Exception:
        dt = None
    return dt if dt is not None else _PF_EPOCH_UTC


def _fmt_open_date(p: dict) -> str:
    """Open timestamp for a portfolio row, rendered in the user's chosen display
    timezone (the Trading-tab selector → display_tz) — same zone as every other
    date in the app.  '—' when the field is absent."""
    dt = _open_dt(p)
    return timez.fmt(dt) if dt is not _PF_EPOCH_UTC else "—"


def render_portfolio_with_close(positions: list[dict], is_demo: bool) -> None:
    """Portfolio table with an inline Close button on every row."""
    acct = "Demo" if is_demo else "Real"

    _process_pending_portfolio_close(positions, is_demo)
    positions = positions_cache.get_positions()

    if _close_err := st.session_state.pop("_pf_close_error", None):
        st.error(f"Close failed: {_close_err}")

    if not positions:
        st.caption(f"No open positions on eToro {acct} account")
        return

    # Sort by Opened — newest first (undated rows fall to the bottom).
    positions = sorted(positions, key=_open_dt, reverse=True)
    live_rows  = [_enrich_position_live(dict(p)) for p in positions]

    # ── Tag each row with its owning bot (or Manual) ──────────────────────────
    _uuid_to_key = {v: k for k, v in bot_registry.get_all().items()}
    for r in live_rows:
        r["_owner"] = _owner_label_for_position(r, _uuid_to_key)

    # ── Bot / Manual filter ───────────────────────────────────────────────────
    _bot_n    = sum(1 for r in live_rows if r["_owner"] != "Manual")
    _manual_n = len(live_rows) - _bot_n
    _counts   = {"All": len(live_rows), "Bots": _bot_n, "Manual": _manual_n}
    # Stable option VALUES (counts only in the label) so the selection survives a
    # changing count between refreshes.
    _flt = st.segmented_control(
        "Show",
        options=["All", "Bots", "Manual"],
        format_func=lambda o: f"{o} ({_counts[o]})",
        default="All",
        key="pf_owner_filter",
        label_visibility="collapsed",
    )
    if _flt == "Bots":
        live_rows = [r for r in live_rows if r["_owner"] != "Manual"]
    elif _flt == "Manual":
        live_rows = [r for r in live_rows if r["_owner"] == "Manual"]

    # ── Re-attribute orphaned positions ───────────────────────────────────────
    # Recover bot ownership for positions that lost it (matched to the bot that
    # logged the entry signal at the same instant).  Only offered when there are
    # unattributed positions to fix.
    if _manual_n:
        if st.button(
            f"Re-attribute {_manual_n} unlabelled position(s) to their bot",
            key="pf_backfill_owners",
            help="Match each Manual position to the bot that opened it "
                 "(same instrument, direction and open time vs the signal log).",
        ):
            assigned = (
                trade_manager.backfill_all_owners(client, demo=is_demo)
                if client else {}
            )
            if assigned:
                st.success(f"Re-attributed {len(assigned)} position(s)/trade(s) to their bot.")
            else:
                st.info("No matches found — remaining positions look genuinely manual.")
            st.rerun()

    if not live_rows:
        st.caption("No positions match this filter.")
        return

    for r in live_rows:
        r["_open_display"] = _fmt_open_date(r)

    n = len(live_rows)
    live_n     = sum(1 for r in live_rows if r.get("current_rate") is not None)
    total_pnl  = sum(r.get("pnl") or 0 for r in live_rows)
    gaining    = sum(1 for r in live_rows if (r.get("pnl") or 0) > 0.005)
    losing     = sum(1 for r in live_rows if (r.get("pnl") or 0) < -0.005)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positions", n)
    m2.metric("Unrealised P&L", _fmt_money(total_pnl))
    m3.metric("Gaining / Losing", f"{gaining} / {losing}")
    live_note = f" · {live_n} live" if live_n else ""
    m4.metric("Account", f"eToro {acct}{live_note}")

    with st.container(border=True):
        if not is_demo:
            st.html(vtables.portfolio_positions_table_html(live_rows))
        else:
            st.html(vtables.portfolio_positions_table_html(live_rows))
            render_portfolio_close_select(live_rows, is_demo)


def render_portfolio_etoro_table(
    positions: list[dict],
    is_demo: bool,
) -> None:
    """eToro-style open positions — single HTML table (fast fragment refresh)."""
    acct = "Demo" if is_demo else "Real"

    n = len(positions)
    if n == 0:
        st.caption(f"No open positions on eToro {acct} account")
        return

    positions = sorted(
        positions,
        key=lambda p: float(p.get("current_value") or p.get("amount") or 0),
        reverse=True,
    )

    live_rows = [_enrich_position_live(dict(p)) for p in positions]
    live_n = sum(1 for r in live_rows if r.get("current_rate") is not None)
    total_pnl = sum(r.get("pnl") or 0 for r in live_rows)
    gaining = sum(1 for r in live_rows if (r.get("pnl") or 0) > 0.005)
    losing = sum(1 for r in live_rows if (r.get("pnl") or 0) < -0.005)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Positions", n)
    m2.metric("Unrealised P&L", _fmt_money(total_pnl))
    m3.metric("Gaining / Losing", f"{gaining} / {losing}")
    live_note = f" · {live_n} live" if live_n else ""
    m4.metric("Account", f"eToro {acct}{live_note}")

    with st.container(border=True):
        st.html(vtables.portfolio_table_html(live_rows))


def ws_badge(instrument_id: int) -> str:
    state = tick_manager.get_state(instrument_id)
    last  = tick_manager.get_last_tick_time(instrument_id)

    if state == State.IDLE:
        if tick_manager.is_running(instrument_id):
            return "🟡 CONNECTING"
        chart = market_data_hub.get_snapshot(instrument_id)
        if (
            chart is not None
            and chart.instrument_id == instrument_id
            and chart.last_tick_time is not None
        ):
            age = (datetime.now(tz=timezone.utc) - chart.last_tick_time).total_seconds()
            if age < tick_manager.LIVE_SEC:
                return "🟢 LIVE"
            return "🟡 STALE"

    if state == State.CONNECTED:
        if last is None:
            return "🟡 WAITING"
        stale = (datetime.now(tz=timezone.utc) - last).total_seconds()
        if stale < tick_manager.LIVE_SEC:
            return "🟢 LIVE"
        return "🟡 STALE"
    if state == State.CONNECTING:
        return "🟡 CONNECTING"
    if state == State.RECONNECTING:
        return "🔄 RECONNECTING"
    if state == State.STOPPED:
        return "🔴 STOPPED"
    return "⚪ IDLE"


def compute_feed_live(instrument_id: int) -> bool:
    """True when background ticks are arriving (or WS is still connecting)."""
    last = tick_manager.get_last_tick_time(instrument_id)
    if last is not None:
        age = (datetime.now(tz=timezone.utc) - last).total_seconds()
        if age < tick_manager.LIVE_SEC:
            return True
    ws_state = tick_manager.get_state(instrument_id)
    if ws_state in (State.CONNECTED, State.CONNECTING, State.RECONNECTING):
        return True
    chart = market_data_hub.get_snapshot(instrument_id)
    if (
        chart is not None
        and chart.instrument_id == instrument_id
        and chart.last_tick_time is not None
    ):
        age = (datetime.now(tz=timezone.utc) - chart.last_tick_time).total_seconds()
        return age < tick_manager.LIVE_SEC
    return False


# ── Instrument list (cached 1 h) ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Loading instruments…")
def load_all_instruments(_k: str, _u: str) -> dict[str, int]:
    c = get_shared_client(_k, _u)
    raw = c.get_instruments()
    opts: dict[str, int] = {}
    for inst in raw.get("instrumentDisplayDatas", []):
        name = inst.get("instrumentDisplayName", "")
        sym  = inst.get("symbolFull", "")
        iid  = inst.get("instrumentID")
        if iid and name:
            opts[f"{name}  ({sym})" if sym else name] = iid
    return opts


try:
    ALL_INSTRUMENTS  = load_all_instruments(api_key, user_key)
    INSTRUMENT_LABELS = list(ALL_INSTRUMENTS.keys())
    INSTRUMENT_ID_TO_LABEL = {iid: lbl for lbl, iid in ALL_INSTRUMENTS.items()}
except Exception as exc:
    st.error(f"Failed to load instrument list: {exc}")
    if st.button("Retry instruments"):
        load_all_instruments.clear()
        st.rerun()
    st.stop()

vtables.configure_label_resolver(_label_for_instrument_id)
_timed("render_sidebar_system", _render_sidebar_system, is_demo)

# ── Background engine boot (multi-instrument) ─────────────────────────────────

# One boot per Python process.  Streamlit re-executes this script on every rerun
# in the SAME module namespace, so a plain `_ENGINES_BOOTED = False` would reset
# the flag each rerun and re-run the (idempotent but costly) boot every time.
# Guard the initialiser so the flag survives reruns and boot truly runs once.
if "_ENGINES_BOOTED" not in globals():
    _ENGINES_BOOTED: bool = False


def _boot_background_engines() -> None:
    """
    Start one trading engine per enabled instrument from instruments.toml.
    Module-level flag ensures this runs exactly once per Python process,
    regardless of how many Streamlit sessions or reruns occur.
    """
    global _ENGINES_BOOTED
    # Primary guard lives in trading_engine (a module imported once, so its state
    # survives Streamlit re-executing this script on every rerun).  Guard on the
    # explicit fleet-boot flag — NOT on engine_count(): a single stray engine
    # created by a UI path before boot (Trading-tab sync / Start callback) once
    # masked a count-based guard and silently left all configured bots unstarted.
    if _ENGINES_BOOTED or trading_engine.fleet_booted():
        _ENGINES_BOOTED = True
        return

    specs = instrument_config.load_specs()
    if not specs:
        return  # transient (missing/empty toml) — retry on the next rerun

    # Resolve instrument_id=0 entries using the live eToro instruments list
    resolved = instrument_config.resolve_ids(specs, ALL_INSTRUMENTS)
    if not resolved:
        return  # transient (instrument list hiccup) — retry on the next rerun

    # Latch only once we're actually registering the fleet.
    _ENGINES_BOOTED = True
    trading_engine.mark_fleet_booted()

    _boot_log = logging.getLogger("app.boot")
    for spec in resolved:
        try:
            trading_engine.start_instrument(
                spec,
                api_key=api_key,
                user_key=user_key,
                is_demo=is_demo,
            )
            _boot_log.info("Boot: started engine for %s (id=%s)", spec.label, spec.instrument_id)
        except Exception as exc:
            _boot_log.error("Boot: failed to start engine for %s: %s", spec.label, exc)

    # Self-heal: drop any persisted disabled-bot keys that aren't real configured
    # bots (e.g. a stray instrument-id "phantom" key like "100003" left by an
    # older build).  All config engines are registered by now, so their keys are
    # the source of truth.
    trading_engine.prune_disabled({s.key for s in resolved})
    bot_ranking.ensure_reviewer()
    # Data moat + weekly walk-forward (both are daemon threads, boot-once).
    try:
        import candle_archive
        candle_archive.ensure_archiver(api_key, user_key)
    except Exception:
        logging.getLogger("app.boot").warning("candle archiver failed to start", exc_info=True)
    try:
        import fleet_scheduler
        fleet_scheduler.ensure_scheduler()
    except Exception:
        logging.getLogger("app.boot").warning("walk-forward scheduler failed to start", exc_info=True)

    # Restore global auto-trade from persisted state (default OFF).  Engine threads
    # only start for bots that are actually ON — avoids dozens of idle tick loops.
    _boot_at = bool(runtime_persist.load().get("auto_trade_active", False))
    trading_engine.restore_auto_trade(_boot_at)
    st.session_state["auto_trade_active"] = _boot_at
    runtime_persist.save(dict(st.session_state))
    _boot_log.info(
        "Boot: global auto-trade=%s, %d engine thread(s), %d bot(s) with auto-trade ON",
        _boot_at, trading_engine.active_engine_count(), trading_engine.auto_trade_count(),
    )

_boot_background_engines()

# ── Normalise REST candles → clean DataFrame ──────────────────────────────────

def normalise_candles(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw)
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if   lc in ("fromdate","date","time","timestamp","datetime"): rename[col] = "time"
        elif lc == "open":           rename[col] = "Open"
        elif lc == "high":           rename[col] = "High"
        elif lc == "low":            rename[col] = "Low"
        elif lc == "close":          rename[col] = "Close"
        elif lc in ("volume","vol"): rename[col] = "Volume"
    df = df.rename(columns=rename)
    keep = [c for c in ("time","Open","High","Low","Close","Volume") if c in df.columns]
    df = df[keep]
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
    return df


# ── Historical candle cache (per session, per instrument+interval) ─────────────

def load_hist_candles(instrument_id: int, api_name: str, count: int) -> pd.DataFrame:
    cache_key = f"hist_{instrument_id}_{api_name}_{count}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        raw   = client.get_candles(instrument_id, "desc", api_name, count)
        outer = extract_list(raw, "candles", "data")
        rows  = outer[0].get("candles", []) if outer else []
        df    = normalise_candles(rows) if rows else pd.DataFrame()
        st.session_state[cache_key] = df
        return df
    except Exception as exc:
        st.error(f"Error loading candles: {exc}")
        return pd.DataFrame()


# ── Non-blocking historical candle load ───────────────────────────────────────
# get_candles is a synchronous REST call (15s timeout × 3 retries + 429 backoff).
# When many bots hammer the eToro API it gets rate-limited, so a blocking call on
# the Streamlit thread freezes the whole UI for tens of seconds ("Loading candles
# hangs").  We fetch on a daemon thread instead and let the chart pick up the
# result on the next rerun — the UI stays responsive throughout.
_hist_lock = threading.Lock()
_hist_async_results: dict[str, "pd.DataFrame"] = {}   # cache_key → df (ready)
_hist_async_inflight: set[str] = set()                # cache_key → fetch running
_hist_async_error: dict[str, str] = {}                # cache_key → last error
_hist_inflight_at: dict[str, float] = {}             # cache_key → monotonic start
_HIST_INFLIGHT_TIMEOUT = 45.0


def load_hist_candles_async(
    instrument_id: int,
    api_name: str,
    count: int,
    *,
    interval_seconds: int | None = None,
    bot_id: str | None = None,
) -> tuple["pd.DataFrame", str]:
    """Return (df, status) where status ∈ {"ready", "loading", "error"}.

    Tries session cache → hub preload → shared client cache before starting a
    background REST fetch.  The caller should mount the poller fragment while
    status == "loading" so the page refreshes once candles arrive."""
    cache_key = f"hist_{instrument_id}_{api_name}_{count}"

    cached = st.session_state.get(cache_key)
    if cached is not None and not cached.empty:
        return cached, "ready"

    if bot_id and interval_seconds:
        hub_df = market_data_hub.get_hist_df(
            bot_id=bot_id, interval_seconds=interval_seconds,
        )
        if not hub_df.empty:
            st.session_state[cache_key] = hub_df
            return hub_df, "ready"

    secs = interval_seconds
    if secs is None and api_name in {v[0] for v in INTERVALS.values()}:
        for _lbl, (an, s) in INTERVALS.items():
            if an == api_name:
                secs = s
                break

    if secs:
        try:
            hit = client.get_hist_candles_cached(instrument_id, secs, count)
            if hit is not None and not hit.empty:
                st.session_state[cache_key] = hit
                return hit, "ready"
        except Exception:
            pass

    with _hist_lock:
        if cache_key in _hist_async_results:               # worker finished
            df = _hist_async_results.pop(cache_key)
            _hist_async_inflight.discard(cache_key)
            _hist_inflight_at.pop(cache_key, None)
            st.session_state[cache_key] = df
            return df, ("ready" if not df.empty else "error")
        if _hist_async_error.pop(cache_key, None):         # worker failed
            _hist_async_inflight.discard(cache_key)
            _hist_inflight_at.pop(cache_key, None)
            return pd.DataFrame(), "error"
        if cache_key in _hist_async_inflight:              # still fetching
            started = _hist_inflight_at.get(cache_key, 0.0)
            if time.monotonic() - started < _HIST_INFLIGHT_TIMEOUT:
                return pd.DataFrame(), "loading"
            _hist_async_inflight.discard(cache_key)
            _hist_inflight_at.pop(cache_key, None)
        _hist_async_inflight.add(cache_key)
        _hist_inflight_at[cache_key] = time.monotonic()

    def _worker(ck: str, iid: int, an: str, cnt: int, interval_secs: int | None) -> None:
        try:
            if interval_secs:
                df = client.get_hist_candles_cached(iid, interval_secs, cnt)
            else:
                raw   = client.get_candles(iid, "desc", an, cnt)
                outer = extract_list(raw, "candles", "data")
                rows  = outer[0].get("candles", []) if outer else []
                df    = normalise_candles(rows) if rows else pd.DataFrame()
            with _hist_lock:
                _hist_async_results[ck] = df
        except Exception as exc:           # noqa: BLE001 — surfaced via status
            with _hist_lock:
                _hist_async_error[ck] = str(exc)

    threading.Thread(
        target=_worker,
        args=(cache_key, instrument_id, api_name, count, secs),
        daemon=True, name="hist-candles",
    ).start()
    return pd.DataFrame(), "loading"


@st.fragment(run_every=1.0)
def _hist_load_poller(cache_key: str) -> None:
    """While candles load in the background, poll once a second and trigger a
    full rerun the moment they're ready so the chart renders without a hang."""
    with _hist_lock:
        done = (cache_key in _hist_async_results) or (cache_key in _hist_async_error)
    if done and st.session_state.get("main_nav") == "Trading":
        st.rerun()


def invalidate_hist(instrument_id: int, api_name: str, count: int) -> None:
    st.session_state.pop(f"hist_{instrument_id}_{api_name}_{count}", None)


# ── TradingView-style chart builder ───────────────────────────────────────────

# ── Signal helper ─────────────────────────────────────────────────────────────

def _prompt_spread_pct(
    asset: str,
    ask: float | None,
    bid: float | None,
) -> float:
    if ask is not None and bid is not None and ask > bid:
        mid = (ask + bid) / 2
        if mid > 0:
            return (ask - bid) / mid * 100
    return prompt_preview.default_spread_pct(asset)


def render_llm_prompt_expander(
    instrument_label: str,
    interval_label: str,
    *,
    ask: float | None = None,
    bid: float | None = None,
    position_type: str = "NONE",
    entry_price: float | None = None,
    key: str = "llm_prompt",
) -> None:
    """Show the Visual Bot system + user prompt with live-filled context."""
    asset = display_asset_name(instrument_label)
    current_price = None
    if ask is not None and bid is not None and ask > bid:
        current_price = (ask + bid) / 2
    spread_pct = _prompt_spread_pct(asset, ask, bid)
    user_prompt = prompt_preview.build_trading_eval_prompt(
        asset=asset,
        timeframe=interval_label,
        current_price=current_price,
        spread_pct=spread_pct,
        position_type=position_type,
        entry_price=entry_price,
    )
    with st.expander("View LLM prompt", expanded=False):
        st.caption(
            "Sent to Visual Bot with the chart image on **Analyse chart now** "
            "and on each candle close (auto-trade / exit review)."
        )
        st.markdown("**System**")
        st.code(prompt_preview.SYSTEM_PROMPT, language="text")
        st.markdown("**User**")
        st.code(user_prompt, language="text")


def _dispatch_manual_signal(
    instrument_id: int,
    df: "pd.DataFrame",
    ask: float,
    bid: float,
    instrument_label: str,
    interval_label: str,
) -> None:
    """Dispatch a manual analysis using the instrument's configured strategy.

    • LLM strategy  → fires the Visual Bot asynchronously (pending state shown).
    • Sync strategy → runs immediately, stores result via set_result_direct.
    """
    strategy_key = _active_strategy_for(instrument_id)
    trigger_at   = timez.now_str("%H:%M:%S")
    _abot        = _active_bot_uuid()   # store under the active bot's key so the panel reads it

    if strategy_key == "llm":
        signal_worker.request_signal(
            df, instrument_id, instrument_label, interval_label,
            trigger_at=trigger_at,
            force=True,
            ask=ask or None,
            bid=bid or None,
            bot_id=_abot,
        )
        return

    # Synchronous strategy: run now, enrich with execution quality, store
    try:
        strategy = strategies.get(strategy_key)
        sig = strategy.generate(df, ask or 0.0, bid or 0.0, instrument_id)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).exception(
            "Manual strategy %r failed for %s", strategy_key, instrument_id
        )
        signal_worker.set_result_direct(
            instrument_id, interval_label,
            {"_error": str(exc), "_status": "done", "_at": trigger_at},
            instrument_label=instrument_label,
            trigger_at=trigger_at,
            bot_id=_abot,
        )
        return

    if sig is None:
        # Strategy returned None (not enough data, etc.) — clear pending state
        signal_worker.set_result_direct(
            instrument_id, interval_label,
            {
                "signal": "HOLD", "confidence": 0,
                "reasoning": f"{strategy.display_name}: insufficient data for a signal",
                "risk_level": "LOW", "observations": [],
                "strategy": strategy_key,
                "_status": "done", "_at": trigger_at,
            },
            instrument_label=instrument_label,
            trigger_at=trigger_at,
            bot_id=_abot,
        )
        return

    from strategies.execution_quality import assess as _eq_assess
    eq     = _eq_assess(df, ask or 0.0, bid or 0.0, strategy_key, sig.confidence, sig.signal)
    result = sig.to_result_dict(trigger_at)
    result["strategy"] = strategy_key          # logged to signal_log + shown in Signals tab
    result.update(eq.to_dict())
    signal_worker.set_result_direct(
        instrument_id, interval_label, result,
        instrument_label=instrument_label,
        trigger_at=trigger_at,
        bot_id=_abot,
    )


def render_signal(result: dict) -> None:
    """Render a signal card in the current Streamlit context."""
    if "_error" in result:
        st.warning(f"Signal bot: {result['_error']}")
        return

    trade_sig = result.get("signal", "HOLD")
    display_sig = result.get("current_signal") or trade_sig
    conf     = result.get("confidence", 0)
    risk     = result.get("risk_level", "—")
    reason   = result.get("reasoning", "") or result.get("reason", "")
    obs      = result.get("observations", [])
    key_lvl  = result.get("key_level")

    col = ui.signal_color(trade_sig)
    css_cls = trade_sig.lower() if trade_sig in ("BUY", "SELL", "HOLD") else "hold"

    st.markdown(
        f"""<div class="signal-card {css_cls}">
          <span class="signal-action" style="color:{col}">{display_sig}</span>
          <span class="signal-meta">
            <b>Confidence:</b> {conf}% &nbsp;|&nbsp;
            <b>Risk:</b> {risk}
            {"&nbsp;|&nbsp;<b>Key level:</b> " + f"{float(key_lvl):.5f}" if key_lvl else ""}
          </span>
        </div>""",
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(reason)

    # ── Execution quality badge ───────────────────────────────────────────────
    viable       = result.get("viable")          # None = old result, no EQ data
    slippage_pct = result.get("slippage_pct")
    net_edge_pct = result.get("net_edge_pct")
    exec_risk    = result.get("exec_risk")
    edge_decay   = result.get("edge_decay")

    if viable is not None and trade_sig != "HOLD":
        risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(exec_risk or "", "⚪")
        decay_txt = f" · decay ×{edge_decay:.2f}" if edge_decay is not None and edge_decay < 0.99 else ""
        if viable:
            st.caption(
                f"📊 **Execution** — "
                f"Slippage: {slippage_pct:.3f}% · "
                f"Net edge: {net_edge_pct:+.3f}% · "
                f"Risk: {risk_icon} {exec_risk}"
                f"{decay_txt}"
            )
        else:
            slippage_str = f"{slippage_pct:.3f}%" if slippage_pct is not None else "?"
            net_str      = f"{net_edge_pct:+.3f}%" if net_edge_pct is not None else "?"
            st.warning(
                f"⚠️ **Trade skipped** — "
                f"net edge {net_str} after {slippage_str} slippage "
                f"({risk_icon} {exec_risk} execution risk)"
            )
            exec_notes = result.get("exec_notes", [])
            if exec_notes:
                with st.expander("Execution detail"):
                    for note in exec_notes:
                        st.caption(f"• {note}")

    if obs:
        with st.expander("Key observations"):
            for o in obs:
                st.markdown(f"- {o}")


def _signal_display_dict(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in ("_status",)}


def _render_entry_signal_content(
    instrument_id: int,
    interval_label: str,
    instrument_label: str,
    *,
    trading_active: bool,
    position_open: bool,
    exit_result: dict | None = None,
    df: "pd.DataFrame | None" = None,
    ask: float = 0.0,
    bid: float = 0.0,
) -> None:
    """Always paint signal slot for the active instrument — avoids stale cards on switch."""
    sym = display_asset_name(instrument_label)
    strategy_key = _active_strategy_for(instrument_id)
    _abot = _active_bot_uuid()

    slot = st.empty()
    with slot:
        if position_open:
            st.caption("Entry signals paused while position is open.")
            render_exit_advice(
                exit_result,
                signal_worker.is_exit_pending(instrument_id, interval_label, _abot),
            )
            return

        if signal_worker.is_pending(instrument_id, interval_label, _abot):
            st.info(f"Analysing **{sym}**…")
            return

        sig_result = signal_worker.get_result(instrument_id, interval_label, _abot)
        if sig_result and sig_result.get("_status") == "done":
            sig = _signal_display_dict(sig_result)
            at = sig.pop("_at", "")
            render_signal(sig)
            if at:
                mode = "auto" if trading_active else "manual"
                st.caption(f"{at} · {mode} · {sym}")
            action = sig.get("signal", "HOLD").upper()
            if action in ("BUY", "SELL", "BUY_LONG", "SELL_SHORT") and not trading_active:
                st.caption(
                    f"⚠ {action} is advisory — enable **Auto-trade** in the sidebar to execute."
                )
            return

        # No signal yet — for non-LLM strategies auto-dispatch immediately so the
        # panel is populated on first load without waiting for the next candle close.
        if strategy_key != "llm" and df is not None and not df.empty:
            _dispatch_manual_signal(
                instrument_id, df, ask, bid, instrument_label, interval_label,
            )
            st.info(f"Reading **{sym}**…")
            return

        hint = (
            "click the **Analyse chart now** button above or wait for the next candle close."
            if strategy_key == "llm"
            else "waiting for the next candle close."
        )
        st.caption(f"No signal yet for **{sym}** · {hint}")


def render_exit_advice(exit_result: dict | None, pending: bool) -> None:
    """Show the LLM's latest HOLD / CLOSE recommendation."""
    if pending:
        st.info("👁 LLM watching chart — reviewing exit on candle close…")
        return
    if not exit_result or exit_result.get("_status") != "done":
        return
    if "_error" in exit_result:
        st.warning(f"Exit advisor: {exit_result['_error']}")
        return

    action = exit_result.get("action", "HOLD")
    trend  = exit_result.get("trend_strength", "—")
    conf   = exit_result.get("confidence", 0)
    reason = exit_result.get("reasoning", "")
    at     = exit_result.get("_at", "")
    col = ui.C_UP if action == "HOLD" else ui.C_HOLD
    css_cls = "hold" if action == "HOLD" else "close"

    st.markdown(
        f"""<div class="exit-card {css_cls}">
          <span style="font-size:0.95rem;font-weight:700;color:{col}">{action}</span>
          <span style="color:{ui.C_MUTED};font-size:0.75rem;margin-left:10px">
            trend: <b style="color:{ui.C_TEXT2}">{trend}</b> &nbsp;·&nbsp; {conf}% conf
            {f" &nbsp;·&nbsp; at {at}" if at else ""}
          </span>
        </div>""",
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(reason)


C_UP   = ui.C_UP
C_DOWN = ui.C_DOWN
C_BG   = ui.C_BG
C_GRID = ui.C_GRID


def build_figure(
    committed: pd.DataFrame,
    forming:   pd.DataFrame,
    instrument: str,
    interval:   str,
    live_label: str = "",
    *,
    live_ask: float = 0.0,
    live_bid: float = 0.0,
    open_trade: "trade_manager.PaperTrade | None" = None,
) -> go.Figure:
    # Localise the time axis to the user's display zone (candles are stored UTC).
    # Convert to naive wall-clock so plotly renders the chosen zone literally
    # instead of re-shifting against the viewer's browser timezone.
    if not committed.empty and "time" in committed.columns:
        committed = committed.copy()
        committed["time"] = timez.localize_series(committed["time"])
    if not forming.empty and "time" in forming.columns:
        forming = forming.copy()
        forming["time"] = timez.localize_series(forming["time"])

    all_df  = pd.concat([committed, forming], ignore_index=True) if not forming.empty else committed.copy()
    has_vol = (
        "Volume" in all_df.columns
        and all_df["Volume"].notna().any()
        and float(all_df["Volume"].sum()) > 0
    )

    if has_vol:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.78, 0.22],
        )
        cr, vr = 1, 2
    else:
        fig = make_subplots(rows=1, cols=1)
        cr, vr = 1, None

    candle_style = dict(whiskerwidth=0, showlegend=False)

    # Committed candles
    if not committed.empty:
        fig.add_trace(go.Candlestick(
            x=committed["time"],
            open=committed["Open"], high=committed["High"],
            low=committed["Low"],  close=committed["Close"],
            increasing=dict(line=dict(color=C_UP,   width=1), fillcolor=C_UP),
            decreasing=dict(line=dict(color=C_DOWN, width=1), fillcolor=C_DOWN),
            name="Price", **candle_style,
        ), row=cr, col=1)

    # Forming candle (semi-transparent — still building)
    if not forming.empty:
        fig.add_trace(go.Candlestick(
            x=forming["time"],
            open=forming["Open"], high=forming["High"],
            low=forming["Low"],  close=forming["Close"],
            increasing=dict(line=dict(color=C_UP,   width=1), fillcolor=ui.C_UP_RGBA),
            decreasing=dict(line=dict(color=C_DOWN, width=1), fillcolor=ui.C_DOWN_RGBA),
            name="Forming", **candle_style,
        ), row=cr, col=1)

    # Current price line + label (mid of live ask/bid when available — matches quote panel)
    if not all_df.empty:
        last_row = all_df.iloc[-1]
        candle_close = float(last_row["Close"])
        if live_ask > 0 and live_bid > 0:
            last_price = (live_ask + live_bid) / 2
            px_color = C_UP if last_price >= float(last_row["Open"]) else C_DOWN
        else:
            last_price = candle_close
            px_color = C_UP if last_price >= float(last_row["Open"]) else C_DOWN
        fig.add_hline(
            y=last_price, line_width=1, line_dash="dot", line_color=px_color,
            row=cr, col=1,
        )
        if live_ask > 0 and live_bid > 0:
            fig.add_hline(
                y=live_bid, line_width=1, line_dash="dash", line_color=ui.C_DOWN,
                opacity=0.55, row=cr, col=1,
            )
            fig.add_hline(
                y=live_ask, line_width=1, line_dash="dash", line_color=ui.C_UP,
                opacity=0.55, row=cr, col=1,
        )
        fig.add_annotation(
            x=1, xref="paper", xanchor="left",
            y=last_price, yref="y" if cr == 1 else f"y{cr}",
            text=f" {(live_bid if live_bid > 0 else last_price):.5f} ",
            showarrow=False,
            font=dict(size=11, color=ui.C_APP),
            bgcolor=px_color, bordercolor=px_color, borderwidth=0,
        )

        # OHLC info line (top-left)
        prev_close = float(all_df.iloc[-2]["Close"]) if len(all_df) > 1 else float(last_row["Open"])
        ref_close = last_price if (live_ask > 0 and live_bid > 0) else candle_close
        chg  = ref_close - prev_close
        pct  = chg / prev_close * 100 if prev_close else 0
        cc   = C_UP if chg >= 0 else C_DOWN
        ohlc = (
            f"O <b>{last_row['Open']:.5f}</b>  "
            f"H <b>{last_row['High']:.5f}</b>  "
            f"L <b>{last_row['Low']:.5f}</b>  "
            f"C <b>{candle_close:.5f}</b>  "
            f"<span style='color:{cc}'>{chg:+.5f} ({pct:+.2f}%)</span>"
        )
        if live_ask > 0 and live_bid > 0:
            ohlc += (
                f"  <span style='color:{ui.C_MUTED}'>· Ask <b>{live_ask:.5f}</b>"
                f" Bid <b>{live_bid:.5f}</b></span>"
        )
        fig.add_annotation(
            x=0.0, y=1.0, xref="paper", yref="paper",
            xanchor="left", yanchor="bottom",
            text=ohlc, showarrow=False,
            font=dict(size=11, color=ui.C_TEXT2),
            bgcolor="rgba(0,0,0,0)",
        )

    # ── Entry price marker (open trade) ──────────────────────────────────────
    if open_trade is not None and not all_df.empty:
        ep     = open_trade.entry_price
        ep_dir = open_trade.direction.upper()        # "LONG" or "SHORT"
        et     = open_trade.entry_time               # datetime (UTC-aware)

        entry_color = C_UP   if ep_dir == "LONG"  else C_DOWN
        dir_label   = "▲ LONG" if ep_dir == "LONG" else "▼ SHORT"
        yref_str    = "y" if cr == 1 else f"y{cr}"

        # Horizontal dashed line at entry price
        fig.add_hline(
            y=ep,
            line_width=1.5, line_dash="dash", line_color=entry_color,
            opacity=0.85, row=cr, col=1,
        )
        # Price label on right side
        fig.add_annotation(
            x=1, xref="paper", xanchor="left",
            y=ep, yref=yref_str,
            text=f" {dir_label} @ {ep:.5f} ",
            showarrow=False,
            font=dict(size=10, color=ui.C_APP),
            bgcolor=entry_color, bordercolor=entry_color, borderwidth=0,
            xshift=2,
        )

        # ── Entry arrow — point at the actual entry TIME on the date axis ──────
        # The x-axis is a datetime axis, so the most accurate placement is the
        # real entry timestamp itself (no candle snapping, no rounding).
        # entry_time is trustworthy when EITHER:
        #   • opened_by_bot          — we stamped it the instant we sent the order
        #   • etoro_open_time_synced — reconciled to eToro's authoritative time
        # Only an *adopted* position whose real open time eToro never provided
        # falls back to a best-effort price search.
        try:
            n = len(all_df)
            # Match the localized (naive wall-clock) time axis used for the candles.
            entry_dt = timez.to_local_naive(et)
            entry_ts = pd.Timestamp(entry_dt)
            t_first  = pd.to_datetime(all_df.iloc[0]["time"])
            t_last   = pd.to_datetime(all_df.iloc[-1]["time"])

            reliable = bool(
                getattr(open_trade, "etoro_open_time_synced", False)
                or getattr(open_trade, "opened_by_bot", False)
            )

            arrow_x = None
            if reliable:
                # Clamp to the visible window so the arrow is always drawable;
                # otherwise use the exact entry moment.
                if entry_ts < t_first:
                    arrow_x = all_df.iloc[0]["time"]
                elif entry_ts > t_last:
                    arrow_x = all_df.iloc[-1]["time"]
                else:
                    arrow_x = entry_dt
            else:
                # Adopted, real open time unknown → best-effort: most recent
                # candle whose High-Low range contains the entry price.
                for _k in range(n - 1, -1, -1):
                    _row = all_df.iloc[_k]
                    _lo = float(_row.get("Low") or 0)
                    _hi = float(_row.get("High") or 0)
                    if _lo > 0 and _hi >= _lo and _lo <= ep <= _hi:
                        arrow_x = all_df.iloc[_k]["time"]
                        break
                if arrow_x is None:
                    arrow_x = (
                        entry_dt if t_first <= entry_ts <= t_last
                        else all_df.iloc[-1]["time"]
                    )

            # Tail is offset in pixels; arrowhead lands exactly on entry_price line.
            # Blue (accent) on purpose: a green/red arrow disappears against the
            # candles — blue is the one color the chart never uses elsewhere.
            ay_shift = 40 if ep_dir == "LONG" else -40   # positive = tail below tip
            fig.add_annotation(
                x=arrow_x,
                y=ep,
                xref="x", yref=yref_str,
                text="",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=2.5,
                arrowcolor=ui.C_ACCENT,
                ax=0,
                ay=ay_shift,
            )
        except Exception:
            pass   # arrow placement is best-effort — never crash the chart

    # Volume subplot
    if has_vol and vr:
        for df_part, opacity in [(committed, 0.6), (forming, 0.35)]:
            if not df_part.empty and "Volume" in df_part.columns:
                colors = [C_UP if c >= o else C_DOWN
                          for c, o in zip(df_part["Close"], df_part["Open"])]
                fig.add_trace(go.Bar(
                    x=df_part["time"], y=df_part["Volume"],
                    marker_color=colors, marker_opacity=opacity,
                    showlegend=False,
                ), row=vr, col=1)

    # Axis styling
    axis_kw = dict(showgrid=True, gridcolor=C_GRID, gridwidth=1,
                   color=ui.C_MUTED, showline=False, zeroline=False)
    fig.update_xaxes(**axis_kw, rangeslider_visible=False)
    fig.update_yaxes(**axis_kw, side="right")

    title_text = (
        f"<b>{instrument}</b>"
        f"  <span style='color:{ui.C_MUTED};font-size:12px'>{interval}</span>"
        + (f"  <span style='color:{ui.C_LIVE};font-size:11px'>● {live_label}</span>" if live_label else "")
    )
    fig.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        margin=dict(l=0, r=72, t=48, b=0),
        height=580,
        hovermode="x unified",
        hoverlabel=dict(bgcolor=ui.C_SURFACE2, font_color=ui.C_TEXT2, font_size=11),
        title=dict(text=title_text, font=dict(size=13, color=ui.C_TEXT2), x=0.0, xanchor="left"),
        showlegend=False,
    )
    return fig


# ── Background trading engine (runs regardless of active tab) ─────────────────

def _sync_global_light() -> None:
    """Cheap per-rerun housekeeping — safe on every tab.

    Keeps the positions poller and engine supervisor alive without pushing
    Trading-tab instrument config or writing runtime state (that belongs in
    sync_background_engine on the Trading tab only).
    """
    positions_cache.start_background_poller(client, st.session_state.get("is_demo", True))
    trading_engine.set_desired_live(True)
    if trading_engine.consume_portfolio_bump():
        bump_portfolio_cache()


def sync_background_engine() -> None:
    """Push UI settings to the module-level supervisor (engine always on when configured)."""
    _sync_global_light()
    _persist_widget_state()

    iid = st.session_state.get("engine_instrument_id")
    running_cfg = trading_engine.get_config()
    if not iid and running_cfg:
        iid = running_cfg.instrument_id
        st.session_state.setdefault("engine_instrument_id", iid)
        st.session_state.setdefault("engine_selected_label", running_cfg.instrument_label)
        st.session_state.setdefault("engine_interval_label", running_cfg.interval_label)
        st.session_state.setdefault("engine_interval_seconds", running_cfg.interval_seconds)
        st.session_state.setdefault("engine_candle_count", running_cfg.candle_count)

    if not iid:
        instruments = globals().get("ALL_INSTRUMENTS") or {}
        default_lbl = _default_instrument_label()
        if default_lbl and default_lbl in instruments:
            iid = instruments[default_lbl]
            st.session_state.setdefault("engine_instrument_id", iid)
            st.session_state.setdefault("engine_selected_label", default_lbl)
            st.session_state.setdefault("engine_interval_label", "1 Minute")
            st.session_state.setdefault("engine_interval_seconds", 60)
            st.session_state.setdefault("engine_candle_count", 300)

    if not iid:
        st.session_state["feed_live"] = False
        return

    trading_engine.set_desired_live(True)
    # Keep the background positions poller's demo flag in sync with the
    # sidebar selector — idempotent, just updates the flag if the poller
    # is already running.
    positions_cache.start_background_poller(client, st.session_state.get("is_demo", True))
    st.session_state["feed_live"] = compute_feed_live(iid)

    api_name = st.session_state.get("engine_api_name", "OneMinute")
    candle_count = st.session_state.get("engine_candle_count", 300)
    hist_cache_key = f"hist_{iid}_{api_name}_{candle_count}"
    hist_df = st.session_state.get(hist_cache_key, pd.DataFrame())

    # Route to the bot the user is currently viewing.  NEVER leave this empty: an
    # empty bot_id makes the engine fall back to an instrument-id-keyed "phantom"
    # engine (e.g. "100003") that duplicates a real config bot, keeps the feed
    # warm even when every bot is off, and pollutes the persisted disabled set.
    # When no bot is explicitly bound, bind to the first configured bot for THIS
    # instrument so the Trading tab always drives a real bot.
    _active_bot = _active_bot_key()
    if not _active_bot:
        _active_bot = next(
            (o["key"] for o in _trading_bot_options() if o["instrument_id"] == iid),
            "",
        )

    config = EngineConfig(
        instrument_id=iid,
        instrument_label=st.session_state.get("engine_selected_label", ""),
        interval_label=st.session_state.get("engine_interval_label", "1 Minute"),
        interval_seconds=st.session_state.get("engine_interval_seconds", 60),
        candle_count=candle_count,
        trading_active=trading_engine.is_auto_trade(iid, bot_id=_active_bot or None),
        demo_amount=st.session_state.get("demo_trade_amount", 1000.0),
        is_demo=st.session_state.get("is_demo", True),
        api_key=api_key,
        user_key=user_key,
        strategy_name=trading_engine.get_strategy(iid, bot_id=_active_bot or None),
        bot_id=_active_bot,
    )
    trading_engine.update_from_ui(config, hist_df)

    # NOTE: do NOT force-sync auto-trade ON here from session_state.
    # The per-bot toggle in the Bots tab is the authoritative control; forcing
    # ON every Streamlit rerun would override a user's per-bot OFF action.
    # Session-wide auto-trade restore happens once at boot via
    # _boot_background_engines, and the sidebar's Start/Stop button handles
    # explicit all-on / all-off transitions.

    if trading_engine.consume_portfolio_bump():
        bump_portfolio_cache()

    if closed := trading_engine.pop_last_close(iid):
        st.session_state[f"last_close_{iid}"] = closed

    if err := trading_engine.get_trade_error(iid):
        st.session_state[f"trade_error_{iid}"] = err

    runtime_persist.save(dict(st.session_state))


# ── Recent-signal feed helper ─────────────────────────────────────────────────

def _render_recent_signals(
    iid: int,
    interval_label: str,
    limit: int = 6,
    bot_id: str = "",
) -> None:
    """Compact live feed of the most recent auto-trade signals for one bot.

    Filtered by bot_id (when provided) so each bot card only shows its own
    signals.  Falls back to interval-only filtering for the Trading tab where
    bot_id is not known.
    """
    records = signal_log.load(
        instrument_ids=[iid],
        interval=interval_label,
        bot_id=bot_id or None,
        limit=limit,
    )
    if not records:
        st.caption("No signals yet — waiting for the next candle close.")
        return

    _dnames = strategies.display_names()
    for rec in records:
        sig_type      = rec.get("type", "entry")
        # Entry → signal field; exit → action field
        if sig_type == "exit":
            decision  = (rec.get("action") or "HOLD").upper()
        else:
            decision  = (rec.get("signal") or "HOLD").upper()
        conf          = rec.get("confidence") or 0
        strategy_key  = rec.get("strategy", "llm")
        strat_label   = _dnames.get(strategy_key, strategy_key.upper())
        trigger_at    = rec.get("trigger_at", "")
        reasoning_raw = rec.get("reasoning") or ""
        reasoning     = reasoning_raw[:70] + ("…" if len(reasoning_raw) > 70 else "")
        viable        = rec.get("viable")
        net_edge      = rec.get("net_edge_pct")

        icon     = {"BUY": "🟢", "SELL": "🔴", "CLOSE": "🟠", "HOLD": "⚪"}.get(decision, "⚪")
        type_tag = " · EXIT" if sig_type == "exit" else ""
        skipped  = " ⛔ skipped" if viable is False else ""
        edge_txt = f" · edge {net_edge:+.2f}%" if net_edge is not None else ""

        st.caption(
            f"{icon} **{decision}** {conf}%  ·  {strat_label}"
            f"  ·  {trigger_at}{type_tag}{edge_txt}{skipped}"
        )
        if reasoning:
            st.caption(f"  ↳ {reasoning}")


# Refresh cadence — tightened after the signal-log cache, mtime-cached TOML,
# and background positions poller landed.  None of these now do REST/disk in
# the render path, so we can push the UI to a more "live" feel.
#
#   LIVE_REFRESH_SEC    → live chart + quote + side panels (Plotly render)
#   PORTFOLIO_REFRESH_SEC → eToro portfolio table (reads positions cache)
#   SIGNALS_REFRESH_SEC   → signal log feed (reads in-memory cache)
LIVE_REFRESH_SEC      = 4
QUOTE_REFRESH_SEC     = LIVE_REFRESH_SEC
CHART_REFRESH_SEC     = LIVE_REFRESH_SEC
PORTFOLIO_REFRESH_SEC = 10
SIGNALS_REFRESH_SEC   = 10


def _live_trading_active() -> bool:
    """True when Trading tab should render live chart + quote fragments."""
    return (
        st.session_state.get("main_nav", "Trading") == "Trading"
        and st.session_state.get("live_feed", False)  # UI: live chart toggle
        and st.session_state.get("engine_instrument_id") is not None
    )


def _render_live_chart_column(chart: market_data_hub.ChartSnapshot) -> None:
    iid = chart.instrument_id
    selected_label = st.session_state.get("engine_selected_label", "")
    interval_label = st.session_state.get("engine_interval_label", "1 Minute")
    badge = ws_badge(iid)
    live_label = badge if "LIVE" in badge else ""

    # Entry marker — ONLY for the bound bot's own trade, not a sibling/manual
    # position on the same instrument.
    open_trade = _active_bot_open_trade(iid)

    fig = build_figure(
        chart.committed, chart.forming,
        selected_label, interval_label, live_label,
        live_ask=chart.latest_ask,
        live_bid=chart.latest_bid,
        open_trade=open_trade,
    )
    st.plotly_chart(fig, width="stretch", key="live_chart", config=_PLOTLY_UI_CONFIG)

    if chart.last_tick_time:
        age = (datetime.now(tz=timezone.utc) - chart.last_tick_time).total_seconds()
        ui.feed_status(
            badge, age, chart.tick_count,
            len(chart.committed) + (0 if chart.forming.empty else 1),
            engine=True,
        )


def _render_live_positions_panel(chart: market_data_hub.ChartSnapshot) -> None:
    iid = chart.instrument_id
    is_demo = st.session_state.get("is_demo", True)
    selected_label = st.session_state.get("engine_selected_label", "")

    # Read from cache (kept fresh by positions_cache background poller) so this
    # fragment never blocks on a REST round-trip during refresh.  Show ONLY the
    # position the bound bot opened — not every position on the instrument.
    positions = positions_owned_by_active_bot(positions_cache.get_positions(), iid)

    with st.container(border=True):
        ui.panel_title("Open Positions")
        render_open_positions_below_chart(
            positions, is_demo,
            chart_instrument_id=iid,
            live_ask=chart.latest_ask,
            live_bid=chart.latest_bid,
            key_prefix="live_pos",
            instrument_label=selected_label,
        )


def _render_static_positions_panel(instrument_id: int, selected_label: str) -> None:
    is_demo = st.session_state.get("is_demo", True)
    # Read from the background-polled cache — no REST in the render path.  Show
    # ONLY the bound bot's position, not every position on the instrument.
    positions = positions_owned_by_active_bot(
        positions_cache.get_positions(), instrument_id
    )
    quote = tick_manager.get_latest_quote(instrument_id)
    live_ask = quote[0] if quote else 0.0
    live_bid = quote[1] if quote else 0.0

    with st.container(border=True):
        ui.panel_title("Open Positions")
        render_open_positions_below_chart(
            positions, is_demo,
            chart_instrument_id=instrument_id,
            live_ask=live_ask,
            live_bid=live_bid,
            key_prefix="static_pos",
            instrument_label=selected_label,
        )


def _render_static_analyse_button(
    instrument_id: int,
    df: "pd.DataFrame",
    last_close: float,
    selected_label: str,
    interval_label: str,
    position_open: bool,
) -> None:
    """Analyse button for static-chart mode — outside the auto-refresh fragment.

    Streamlit drops button clicks that occur inside fragments with run_every
    when the timer fires at the same moment.  Placing the button here (in the
    main render path) guarantees the click is always delivered.
    """
    _s_key = _active_strategy_for(instrument_id)
    if _s_key != "llm":
        return  # non-LLM strategies run automatically — no manual trigger shown
    if st.button(
        "Analyse chart now",
        key=f"sig_static_{instrument_id}",
        width="stretch",
        disabled=position_open,
        type="primary" if not position_open else "secondary",
    ):
        quote = tick_manager.get_latest_quote(instrument_id)
        ask = quote[0] if quote else last_close
        bid = quote[1] if quote else last_close
        _dispatch_manual_signal(instrument_id, df, ask, bid, selected_label, interval_label)
        st.rerun()  # force immediate fragment re-render so signal appears at once


def _render_live_analyse_button() -> None:
    """Outside fragments — Streamlit drops button clicks inside auto-refresh fragments."""
    iid = st.session_state.get("engine_instrument_id")
    if not iid:
        return
    chart = _active_chart_snapshot(iid)
    trade = trading_engine.get_snapshot(bot_id=_active_bot_key()) or trading_engine.get_snapshot(iid)
    if chart is None or chart.instrument_id != iid:
        return
    position_open = trade.position_open if trade else False
    selected_label = st.session_state.get("engine_selected_label", "")
    interval_label = st.session_state.get("engine_interval_label", "1 Minute")
    _strat_key = _active_strategy_for(iid)
    if _strat_key != "llm":
        return  # non-LLM strategies run automatically — no manual trigger shown
    if st.button(
        "Analyse chart now", key=f"sig_live_manual_{iid}", width="stretch",
        disabled=position_open,
        type="primary" if not position_open else "secondary",
    ):
        _dispatch_manual_signal(
            iid, chart.chart_data,
            chart.latest_ask or 0.0, chart.latest_bid or 0.0,
            selected_label, interval_label,
        )
        st.rerun()  # force immediate fragment re-render so signal appears at once


def _render_live_side_panels(chart: market_data_hub.ChartSnapshot) -> None:
    iid = chart.instrument_id
    demo_trade_amount = st.session_state.get("demo_trade_amount", 1000.0)
    selected_label = st.session_state.get("engine_selected_label", "")
    interval_label = st.session_state.get("engine_interval_label", "1 Minute")

    _abot = _active_bot_key()
    trade = trading_engine.get_snapshot(bot_id=_abot) or trading_engine.get_snapshot(iid)
    badge = ws_badge(iid)
    trading_active = trading_engine.is_auto_trade(iid, bot_id=_abot)
    position_open = trade.position_open if trade else False

    exit_result = signal_worker.get_exit_result(iid, interval_label, _active_bot_uuid())

    trade_err = st.session_state.get(f"trade_error_{iid}")
    if trade_err:
        st.error(f"eToro order failed: {trade_err}")

    with st.container(border=True):
        ui.panel_title("Live Quote")
        if chart.latest_ask or chart.latest_bid:
            ui.quote_strip(
                chart.latest_ask, chart.latest_bid,
                chart.latest_ask - chart.latest_bid,
            )
        else:
            st.caption(
                f"{badge} — connecting to eToro WebSocket…"
                if "CONNECT" in badge or "WAIT" in badge
                else f"{badge} — waiting for ticks…"
            )

    with st.container(border=True):
        ui.panel_title("Auto-Trade")
        if position_open and trading_active:
            ui.status_banner(
                "manage",
                "Position open — LLM exit on each candle close · "
                "new entries resume automatically after close",
            )
        elif position_open:
            ui.status_banner(
                "manage",
                "Position open — LLM tracking exit on each candle close · "
                "enable auto-trade in the sidebar to hunt new entries after close",
            )
        elif trading_active:
            ui.status_banner(
                "hunt",
                f"Hunting signals · ${demo_trade_amount:.0f} per entry",
            )
            _render_recent_signals(iid, interval_label, limit=1)
        else:
            ui.status_banner(
                "off",
                "Entries off — enable auto-trade in the sidebar to open positions on signals",
            )

    pos_type = "NONE"
    entry_px = None
    if position_open:
        open_trade = _active_bot_open_trade(iid)
        chart_positions = positions_owned_by_active_bot(
            positions_cache.get_positions(), iid,
        )
        etoro_pos = chart_positions[0] if chart_positions else {}
        if open_trade:
            pos_type = open_trade.direction
            entry_px = open_trade.entry_price
        elif etoro_pos:
            pos_type = str(etoro_pos.get("direction") or "LONG").upper()
            entry_px = etoro_pos.get("open_rate")

    _lv_key   = _active_strategy_for(iid)
    _lv_label = strategies.display_names().get(_lv_key, _lv_key)
    _lv_title = "AI Signal" if _lv_key == "llm" else f"Signal · {_lv_label}"
    with st.container(border=True):
        ui.panel_title(_lv_title)
        if _lv_key == "llm":
            render_llm_prompt_expander(
                selected_label,
                interval_label,
                ask=chart.latest_ask or None,
                bid=chart.latest_bid or None,
                position_type=pos_type,
                entry_price=float(entry_px) if entry_px is not None else None,
                key=f"llm_prompt_live_{iid}",
            )
        _render_entry_signal_content(
                iid,
                interval_label,
                selected_label,
                trading_active=trading_active,
                position_open=position_open,
                exit_result=exit_result,
                df=chart.chart_data,
                ask=chart.latest_ask or 0.0,
                bid=chart.latest_bid or 0.0,
            )


@st.fragment(run_every=LIVE_REFRESH_SEC)
def live_trading_chart_fragment() -> None:
    """Live chart + positions (left column only).

    Split from the side panel so the right rail — General Stats, Analyse, Live
    Quote, Auto-Trade — renders as ONE cohesive column (built by the caller in
    col_side) instead of General Stats floating in a separate row above it.
    """
    if not _live_trading_active():
        return
    iid = st.session_state["engine_instrument_id"]
    try:
        chart = _active_chart_snapshot(iid)   # the active bot's own interval
        if chart is None or chart.instrument_id != iid:
            st.caption(f"{ws_badge(iid)} — starting chart builder…")
            return
        _render_live_chart_column(chart)
        _render_live_positions_panel(chart)
    except Exception as exc:
        st.warning(f"Live trading paused — {exc}")


@st.fragment(run_every=LIVE_REFRESH_SEC)
def live_trading_side_fragment() -> None:
    """Live Quote + Auto-Trade panels (right rail, rendered below General Stats).

    Reads its own snapshot; a sub-second drift from the chart fragment is
    imperceptible for a quote/status panel.
    """
    if not _live_trading_active():
        return
    iid = st.session_state["engine_instrument_id"]
    try:
        chart = _active_chart_snapshot(iid)
        if chart is None or chart.instrument_id != iid:
            return
        _render_live_side_panels(chart)
    except Exception as exc:
        st.warning(f"Live panel paused — {exc}")


@st.fragment(run_every=QUOTE_REFRESH_SEC)
def trading_feed_badge_fragment(
    instrument_id: int,
    selected_label: str,
    interval_label: str,
) -> None:
    """Toolbar feed badge — must refresh; static render showed IDLE while WS connected."""
    if st.session_state.get("main_nav") != "Trading":
        return
    feed_live = compute_feed_live(instrument_id)
    st.session_state["feed_live"] = feed_live
    ui.toolbar_hint(selected_label, interval_label, ws_badge(instrument_id))


@st.fragment(run_every=QUOTE_REFRESH_SEC)
def trading_engine_status_fragment(
    instrument_id: int,
    live_mode: bool,
) -> None:
    if st.session_state.get("main_nav") != "Trading":
        return
    feed_live = compute_feed_live(instrument_id)
    st.caption(
        f"Engine **{'live' if feed_live else 'connecting…'}** · "
        f"chart **{'live' if live_mode else 'static'}**"
    )


@st.fragment(run_every=CHART_REFRESH_SEC)
def static_positions_fragment() -> None:
    if st.session_state.get("main_nav", "Trading") != "Trading":
        return
    if st.session_state.get("live_feed", False):
        return
    iid = st.session_state.get("engine_instrument_id")
    if not iid:
        return
    try:
        _render_static_positions_panel(
            iid, st.session_state.get("engine_selected_label", ""),
        )
    except Exception as exc:
        st.warning(f"Positions paused — {exc}")


@st.fragment(run_every=LIVE_REFRESH_SEC)
def static_right_panel_fragment(
    instrument_id: int,
    selected_label: str,
    interval_label: str,
    df_high: float,
    df_low: float,
    last_close: float,
    df_show: pd.DataFrame,
) -> None:
    """
    Auto-refreshing right-column panels for static (non-live) chart mode.
    Runs every LIVE_REFRESH_SEC so engine signals and auto-trade status appear
    without needing the live chart toggle to be on.
    """
    if st.session_state.get("main_nav") != "Trading":
        return
    if st.session_state.get("live_feed", False):
        return  # live_trading_sync_fragment handles this path

    trading_active    = trading_engine.is_auto_trade(instrument_id, bot_id=_active_bot_key())
    demo_trade_amount = st.session_state.get("demo_trade_amount", 1000.0)
    position_open     = _active_bot_open_trade(instrument_id) is not None

    # ── Session OHLC ──────────────────────────────────────────────────────────
    with st.container(border=True):
        ui.panel_title("Session OHLC")
        st.metric("Period high", f"{df_high:.5f}")
        st.metric("Period low",  f"{df_low:.5f}")
        st.caption(f"Loaded {timez.now_str('%H:%M:%S')} {timez.abbrev()}")

    # ── Auto-Trade status ─────────────────────────────────────────────────────
    with st.container(border=True):
        ui.panel_title("Auto-Trade")
        if position_open and trading_active:
            ui.status_banner(
                "manage",
                "Position open — LLM exit on each candle close · "
                "new entries resume automatically after close",
            )
        elif trading_active:
            ui.status_banner(
                "hunt",
                f"Hunting signals · ${demo_trade_amount:.0f} per entry",
            )
            _render_recent_signals(instrument_id, interval_label, limit=1)
        else:
            ui.status_banner(
                "off",
                "Entries off — enable auto-trade in the sidebar to open positions on signals",
            )

    # ── Signal panel (all strategies) ─────────────────────────────────────────
    # NOTE: the "Analyse now" button lives outside this fragment; auto-refresh
    # fragments drop button clicks when the timer fires during the click.
    _s_key   = _active_strategy_for(instrument_id)
    _s_label = strategies.display_names().get(_s_key, _s_key)
    _s_title = "AI Signal" if _s_key == "llm" else f"Signal · {_s_label}"
    with st.container(border=True):
        ui.panel_title(_s_title)
        quote = tick_manager.get_latest_quote(instrument_id)
        ask = quote[0] if quote else last_close
        bid = quote[1] if quote else last_close
        if _s_key == "llm":
            render_llm_prompt_expander(
                selected_label, interval_label,
                ask=ask, bid=bid,
                key=f"llm_prompt_static_{instrument_id}",
            )
        _render_entry_signal_content(
                instrument_id,
                interval_label,
                selected_label,
                trading_active=trading_active,
                position_open=position_open,
                df=df_show,
                ask=ask,
                bid=bid,
            )


HISTORY_REFRESH_SEC = 60


@st.fragment(run_every=HISTORY_REFRESH_SEC)
def history_tab_fragment(
    is_demo: bool,
    min_date: datetime.date,
    *,
    period_mode: str = _DEFAULT_PERIOD,
    custom_start=None,
    custom_end=None,
    period_lbl: str = "",
) -> None:
    """Reload eToro trade history on tab open and every minute while History is active."""
    if st.session_state.get("main_nav") != "History":
        return

    hist_err = st.session_state.get("hist_load_error")
    if hist_err == "permission":
        permission_error("Trade History", scope_hint="Trade History read")
        st.info(
            "Trade history is only available on **Demo** for most API keys. "
            "Switch **Environment → Demo** in the sidebar, or enable Trade History read for real."
        )
        return
    if hist_err:
        st.error(f"Trade history error: {hist_err}")
        return

    # Refetch only when actually needed.  The page handler clears the cache on
    # tab-open / Refresh / range change, so a missing-or-stale cache is our
    # signal to fetch.  Otherwise reuse the cached trades — fetching all
    # trade-history pages (up to 25 REST calls) on every fragment pass made
    # opening the tab (and the post-switch rerun) hang.  run_every keeps it
    # fresh: once HISTORY_REFRESH_SEC has elapsed we refetch on the next pass.
    fetch_key   = _history_fetch_key(is_demo, min_date)
    cached_ok   = (
        st.session_state.get("etoro_hist_key") == fetch_key
        and st.session_state.get("etoro_hist_trades") is not None
    )
    last_fetch  = st.session_state.get("_hist_fetched_at", 0.0)
    stale       = (time.time() - last_fetch) >= HISTORY_REFRESH_SEC
    need_fetch  = (not cached_ok) or stale

    show_spinner = st.session_state.pop("_hist_show_spinner", False)
    if need_fetch and show_spinner:
        with st.spinner("Loading trade history…"):
            trades = _ensure_history_trades(is_demo, min_date, force=True)
        st.session_state["_hist_fetched_at"] = time.time()
    elif need_fetch:
        trades = _ensure_history_trades(is_demo, min_date, force=True)
        st.session_state["_hist_fetched_at"] = time.time()
    else:
        trades = st.session_state.get("etoro_hist_trades")
    if trades is None:
        return

    shown = _filter_etoro_history_period(
        trades,
        period_mode,
        custom_start=custom_start,
        custom_end=custom_end,
    )
    # History shows closed trades only — no need to force-fetch open positions.
    render_unified_history(shown, [], is_demo=is_demo, period_lbl=period_lbl)


BOTS_REFRESH_SEC = 15


def _on_bot_toggle(bot_key: str, iid: int, current_at: bool) -> None:
    """on_click callback — runs atomically before the rerun, immune to fragment-timer races.

    Using on_click= instead of checking `if st.button(...)` prevents the Streamlit
    fragment timer from racing with the button press and dropping the click.

    Turning OFF: disables auto-trade AND stops all background workers (engine
    thread, tick manager, market-data hub slice).
    Turning ON: re-enables auto-trade AND restarts the engine thread if stop_bot
    previously halted it.
    """
    new_state = not current_at
    trading_engine.set_auto_trade(iid, new_state, bot_id=bot_key)
    if not new_state:
        # Bot toggled OFF — halt engine thread + tick/hub workers
        trading_engine.stop_bot(bot_key)
    else:
        # Bot toggled ON — restart engine thread in case stop_bot halted it
        specs = instrument_config.load_specs()
        spec = next((s for s in specs if s.key == bot_key), None)
        if spec is not None:
            resolved_list = instrument_config.resolve_ids(
                [spec], globals().get("ALL_INSTRUMENTS") or {}
            )
            if resolved_list:
                trading_engine.start_instrument(
                    resolved_list[0],
                    api_key=api_key,
                    user_key=user_key,
                    is_demo=st.session_state.get("is_demo", True),
                )


def _on_bots_bulk_toggle(bot_pairs: list[tuple[str, int]], turn_on: bool) -> None:
    """on_click callback — turn auto-trade ON/OFF for a SET of bots at once.

    `bot_pairs` is the list of (bot_key, instrument_id) currently visible under
    the Bots-page filter, so the bulk action only affects the filtered bots.
    Mirrors _on_bot_toggle's per-bot behaviour (engine thread + workers).
    """
    is_demo = st.session_state.get("is_demo", True)
    _ak, _uk = api_key, user_key
    all_instruments = globals().get("ALL_INSTRUMENTS") or {}

    # 1) Flip the auto-trade flag for every bot NOW — this is cheap (in-memory +
    #    one small file write) and makes the UI reflect the new state instantly.
    for bot_key, iid in bot_pairs:
        trading_engine.set_auto_trade(iid, turn_on, bot_id=bot_key)

    # 2) Do the heavy lifting (engine threads, websocket feeds, hist fetches) OFF
    #    the Streamlit thread.  Starting/stopping ~30 bots synchronously here would
    #    block the rerun and hang the UI; a background worker (lightly staggered to
    #    avoid an eToro REST/WS thundering herd) keeps the page responsive.
    def _apply_bulk() -> None:
        specs = instrument_config.load_specs()
        for bot_key, iid in bot_pairs:
            try:
                if not turn_on:
                    trading_engine.stop_bot(bot_key)
                else:
                    spec = next((s for s in specs if s.key == bot_key), None)
                    if spec is not None:
                        resolved_list = instrument_config.resolve_ids([spec], all_instruments)
                        if resolved_list:
                            trading_engine.start_instrument(
                                resolved_list[0],
                                api_key=_ak, user_key=_uk, is_demo=is_demo,
                            )
            except Exception:
                logging.getLogger("app").warning(
                    "Bulk toggle failed for bot %s", bot_key, exc_info=True
                )
            time.sleep(0.1)   # stagger feed/REST startup across bots

    threading.Thread(target=_apply_bulk, daemon=True, name="bulk-bot-toggle").start()


@st.fragment(run_every=BOTS_REFRESH_SEC)
def bots_live_fragment() -> None:
    """Auto-refreshing overview of all running bots (one card per instruments.toml entry)."""
    if st.session_state.get("main_nav") != "Bots":
        return

    specs = instrument_config.load_specs()
    # get_all_snapshots now returns dict[str, ...] keyed by bot_id (spec.key)
    all_snaps = trading_engine.get_all_snapshots()
    all_hub   = market_data_hub.get_all_snapshots()

    if not specs:
        st.info(
            "No instruments configured. "
            "Add entries to `instruments.toml` and rebuild the container."
        )
        return

    st.subheader("Running Bots")
    st.caption(
        "Each row is an independent bot — separate candle interval and strategy. "
        "Multiple bots for the same asset share one WebSocket feed."
    )

    resolved = instrument_config.resolve_ids(
        specs, globals().get("ALL_INSTRUMENTS") or {}
    )

    # ── Filters (strategy / stock / frequency) + bulk on-off ──────────────────
    # The bulk buttons act ONLY on the bots matching the current filter, so the
    # user can e.g. turn every XRP bot — or every Supertrend bot — on/off at once.
    _strat_disp     = strategies.display_names()
    _present_strats = sorted({s.strategy for s in resolved})
    _present_labels = sorted({s.label for s in resolved})
    _freq_order     = ("scalp", "intraday", "swing", "daily", "weekly")
    _present_freqs  = sorted(
        {_freq_bucket(s.interval_secs) for s in resolved},
        key=lambda b: _freq_order.index(b) if b in _freq_order else 99,
    )

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        _strat_filter = st.selectbox(
            "Filter by strategy",
            options=["__all__"] + _present_strats,
            format_func=lambda k: "All strategies" if k == "__all__" else _strat_disp.get(k, k),
            key="bots_filter_strategy",
        )
    with fc2:
        _stock_filter = st.selectbox(
            "Filter by stock",
            options=["__all__"] + _present_labels,
            format_func=lambda k: "All stocks" if k == "__all__" else k,
            key="bots_filter_stock",
        )
    with fc3:
        _freq_filter = st.selectbox(
            "Filter by trading frequency",
            options=["__all__"] + _present_freqs,
            format_func=lambda k: _FREQ_FILTER_LABELS.get(k, k),
            key="bots_filter_frequency",
        )
    with fc4:
        _state_filter = st.selectbox(
            "Filter by state",
            options=["__all__", "on", "off"],
            format_func=lambda k: {"__all__": "All states", "on": "🟢 On (auto-trade)",
                                   "off": "⭕ Off"}[k],
            key="bots_filter_state",
        )

    def _bot_is_on(key: str) -> bool:
        snap = all_snaps.get(key)
        return bool(snap and snap.trading_active)

    visible = [
        s for s in resolved
        if (_strat_filter == "__all__" or s.strategy == _strat_filter)
        and (_stock_filter == "__all__" or s.label == _stock_filter)
        and (_freq_filter == "__all__" or _freq_bucket(s.interval_secs) == _freq_filter)
        and (_state_filter == "__all__" or _bot_is_on(s.key) == (_state_filter == "on"))
    ]
    _visible_pairs = [(s.key, s.instrument_id) for s in visible]

    bc1, bc2, bc3 = st.columns([1, 1, 2], vertical_alignment="center")
    with bc1:
        st.button(
            "Turn ON all", key="bots_bulk_on", type="primary",
            use_container_width=True, disabled=not _visible_pairs,
            help="Enable auto-trade for every bot matching the current filter",
            on_click=_on_bots_bulk_toggle, args=(_visible_pairs, True),
        )
    with bc2:
        st.button(
            "Turn OFF all", key="bots_bulk_off",
            use_container_width=True, disabled=not _visible_pairs,
            help="Disable auto-trade for every bot matching the current filter",
            on_click=_on_bots_bulk_toggle, args=(_visible_pairs, False),
        )
    with bc3:
        st.caption(f"{len(visible)} of {len(resolved)} bot(s) shown")

    if not visible:
        st.info("No bots match the current filter.")
        return

    # Group bots by instrument label for visual separation
    seen_labels: set[str] = set()

    # ── Per-bot rows ──────────────────────────────────────────────────────────
    for spec in visible:
        iid      = spec.instrument_id
        bot_key  = spec.key                    # unique toml section key e.g. "btc_15m"
        interval = spec.interval

        snap      = all_snaps.get(bot_key)
        chart     = all_hub.get(bot_key)
        # Stable UUID for this bot — used for signal filtering and trade ownership.
        # OFF bots have no live snapshot, so fall back to the persistent registry
        # rather than dropping the UUID (which would break position attribution).
        bot_uuid  = (snap.bot_uuid if snap else None) or bot_registry.get(bot_key) or ""

        # Show a divider / header when we first encounter a new instrument label
        if spec.label not in seen_labels:
            seen_labels.add(spec.label)
            st.markdown(f"##### {spec.label}")

        # Effective ON = user enabled AND market open.  User intent is kept so
        # bots auto-resume when the session reopens without a manual toggle.
        is_on = trading_engine.is_auto_trade(iid, bot_id=bot_key)
        user_on = trading_engine.is_user_auto_trade_enabled(iid, bot_id=bot_key)

        # Feed status.  The tick stream is a SHARED per-instrument resource: a
        # sibling bot on the same asset — or that asset being open in the Trading
        # chart — keeps it live regardless of THIS bot's state.  Showing the raw
        # shared feed on an OFF bot made it read "live even though off" (and made
        # XRP look live while BTC idled simply because XRP was the charted asset).
        # So the badge reflects the BOT: only an effectively ON bot shows the live feed.
        if not user_on:
            feed_badge = "⚪ off"
        elif not is_on:
            feed_badge = "🌙 MARKET CLOSED"
        else:
            ws_state  = tick_manager.get_state(iid)
            last_tick = tick_manager.get_last_tick_time(iid)
            if last_tick:
                age = (datetime.now(tz=timezone.utc) - last_tick).total_seconds()
                if age < tick_manager.LIVE_SEC:
                    feed_badge = "🟢 LIVE"
                else:
                    feed_badge = f"🟡 STALE ({int(age)}s)"
            elif ws_state.value == "connected":
                feed_badge = "🟡 WAITING"
            elif ws_state.value in ("connecting", "reconnecting"):
                feed_badge = "🔄 " + ws_state.value.upper()
            else:
                feed_badge = "⚪ IDLE"

        # Position — THIS bot's own trade (the store is bot-keyed, so a sibling
        # bot's position on the same instrument never shows here).
        open_trade = trade_manager.get_open(bot_uuid) if bot_uuid else None
        if open_trade is not None:
            ask = chart.latest_ask if chart else 0.0
            bid = chart.latest_bid if chart else 0.0
            _, pnl_d, pnl_pct, _ = trade_manager.dollar_unrealised_pnl(
                open_trade.direction, open_trade.entry_price, ask, bid, trade=open_trade,
            )
            pnl_color = "🟢" if pnl_d >= 0 else "🔴"
            pos_text = (
                f"{open_trade.direction} @ {open_trade.entry_price:.4f} · "
                f"{pnl_color} {pnl_pct:+.2f}%"
            )
        else:
            pos_text = "—"

        # Last strategy signal for THIS bot (keyed by its UUID)
        sig  = signal_worker.get_result(iid, interval, bot_uuid)
        esig = signal_worker.get_exit_result(iid, interval, bot_uuid)
        if esig and esig.get("_status") == "done" and "_error" not in esig:
            last_sig = f"EXIT · {esig.get('action','?')} {esig.get('confidence','?')}%"
        elif sig and sig.get("_status") == "done" and "_error" not in sig:
            last_sig = f"{sig.get('signal','?')} {sig.get('confidence','?')}%"
        elif sig and sig.get("_status") == "pending":
            last_sig = "⏳ pending…"
        else:
            last_sig = "—"

        # Creation time from snapshot (populated once the engine ticks at least once)
        started_at = snap.started_at if snap else None
        started_str = (
            timez.fmt(started_at, "%H:%M:%S") if started_at else "starting…"
        )

        # Card: info on left, controls stacked on right
        with st.container(border=True):
            info_col, ctrl_col = st.columns([5, 1], vertical_alignment="center")

            with info_col:
                n1, n2, n3 = st.columns([3, 2, 3])
                with n1:
                    st.markdown(f"**{interval}** bot")
                    uuid_short = bot_uuid[:8] if bot_uuid else "—"
                    st.caption(f"id: `{uuid_short}` · key: `{bot_key}` · started {started_str}")
                    _bleed_cap = bot_ranking.card_caption(
                        bot_uuid,
                        strategy=spec.strategy,
                        interval=spec.interval,
                        instrument_label=spec.label,
                        instrument_id=iid,
                    )
                    if _bleed_cap:
                        st.caption(_bleed_cap)
                with n2:
                    st.caption("Feed")
                    st.markdown(feed_badge)
                with n3:
                    st.caption("Position")
                    st.markdown(pos_text)

                # Strategy selector — keyed by bot_key to avoid widget collisions
                _strat_names  = strategies.display_names()
                _strat_keys   = list(_strat_names.keys())
                _strat_labels = list(_strat_names.values())
                _cur_key      = trading_engine.get_strategy(iid, bot_id=bot_key)
                _cur_label    = _strat_names.get(_cur_key, "")
                _cur_idx      = _strat_keys.index(_cur_key) if _cur_key in _strat_keys else 0
                _bot_track_key = f"_bot_strat_last_set_{bot_key}"
                _bot_last_set  = st.session_state.get(_bot_track_key, _cur_key)
                if _cur_key != _bot_last_set:
                    st.session_state[f"bot_strategy_{bot_key}"] = _cur_label
                    st.session_state[_bot_track_key] = _cur_key
                _new_label = st.selectbox(
                    "Strategy",
                    options=_strat_labels,
                    index=_cur_idx,
                    key=f"bot_strategy_{bot_key}",
                    label_visibility="collapsed",
                    help="Strategy that drives auto-trade signals for this bot",
                )
                _new_key = _strat_keys[_strat_labels.index(_new_label)]
                if _new_key != _cur_key:
                    trading_engine.set_strategy(iid, _new_key, bot_id=bot_key)
                    st.session_state[_bot_track_key] = _new_key
                    st.rerun()

                st.caption(f"Last signal: {last_sig}")

            with ctrl_col:
                current_at = user_on
                at_label   = "🤖 ON" if is_on else "🤖 OFF"
                at_type    = "primary" if is_on else "secondary"
                _at_help = (
                    "Toggle auto-trade for this bot.  Shows OFF while the market "
                    "is closed; resumes automatically at the next session if left "
                    "enabled."
                    if (user_on and not is_on)
                    else "Toggle auto-trade for this bot"
                )
                # Use on_click= so the action fires in a callback (before rerun)
                # rather than being checked inside the fragment body.  Callbacks
                # are immune to the fragment timer firing at the same instant as
                # the button press, which would otherwise drop the click.
                st.button(
                    at_label, key=f"bot_at_{bot_key}",
                    type=at_type, use_container_width=True,
                    help=_at_help,
                    on_click=_on_bot_toggle,
                    args=(bot_key, iid, current_at),
                )
                if st.button(
                    "View →", key=f"bot_view_{bot_key}",
                    use_container_width=True,
                    help="Open this bot's chart in the Trading tab",
                ):
                    # Bind the Trading tab to this bot.  The Trading toolbar reads
                    # engine_active_bot_key and derives instrument / interval /
                    # strategy / chart from the bot's spec, so this single key is
                    # all that's needed to switch the whole tab to this bot.
                    st.session_state["engine_active_bot_key"] = bot_key
                    st.session_state["main_nav"] = "Trading"
                    st.rerun()

                # Delete — only allowed when auto-trade is OFF and no owned position
                _has_owned_pos = bool(
                    bot_uuid and trade_manager.get_open(bot_uuid) is not None
                )
                _can_delete = not current_at and not _has_owned_pos
                _delete_help = (
                    "Disable auto-trade first." if current_at
                    else "Close the open position first." if _has_owned_pos
                    else "Permanently stop and remove this bot."
                )
                if st.button(
                    "Delete", key=f"bot_del_{bot_key}",
                    use_container_width=True,
                    type="secondary",
                    disabled=not _can_delete,
                    help=_delete_help,
                ):
                    ok, reason = trading_engine.delete_bot(bot_key)
                    if ok:
                        st.success(f"Bot `{bot_key}` deleted.")
                    else:
                        st.error(reason)
                    st.rerun()

    # ── Session P&L summary (per instrument, not per bot) ─────────────────────
            st.divider()
    st.caption("Session P&L (closed trades this run — per instrument)")
    # Deduplicate by iid so we don't show the same instrument twice
    seen_iids: dict[int, str] = {}
    for spec in resolved:
        if spec.instrument_id not in seen_iids:
            seen_iids[spec.instrument_id] = spec.label
    pnl_cols = st.columns(len(seen_iids) or 1)
    for i, (iid, label) in enumerate(seen_iids.items()):
        closed_trades = trade_manager.get_closed(iid)
        total_pnl     = trade_manager.total_realised_pnl(iid)
        with pnl_cols[i]:
            st.metric(
                label,
                f"${total_pnl:+.2f}",
                f"{len(closed_trades)} trade{'s' if len(closed_trades) != 1 else ''} closed",
            )


@st.fragment(run_every=SIGNALS_REFRESH_SEC)
def signals_live_fragment() -> None:
    """Auto-refreshing LLM signal log page (polls every 30 s)."""
    if st.session_state.get("main_nav") != "Signals":
        return

    total = signal_log.total_count()
    st.subheader("Signal Log")
    st.caption(
        f"{total} signal{'s' if total != 1 else ''} logged · "
        "every entry and exit analysis from all strategies, newest first"
    )

    if total == 0:
        st.info(
            "**No signals yet.** No strategy has been run this session.  \n"
            "Go to the **Trading** tab, pick an instrument, and either:\n"
            "- Enable **Auto-trade** — signals fire automatically on each candle close\n"
            "- Click the **Run / Analyse** button to analyse the current chart manually"
        )
        return

    # ── filters — use the full session instrument list, not just logged ones ──
    all_inst_labels = sorted(ALL_INSTRUMENTS.keys()) if ALL_INSTRUMENTS else []

    default_inst: list[str] = []
    current_lbl = st.session_state.get("engine_selected_label")
    if current_lbl and current_lbl in ALL_INSTRUMENTS:
        default_inst = [current_lbl]

    # Bot options: current configured bots that have a UUID (i.e. can have signals).
    _key_to_uuid = bot_registry.get_all()                       # {key: uuid}
    _cfg_keys    = {s.key for s in instrument_config.load_specs()}
    _bot_keys    = sorted(k for k in _key_to_uuid if k in _cfg_keys)

    f1, f2, f3, f4 = st.columns([2.5, 1.8, 1.6, 2.3])
    with f1:
        sel_instruments = st.multiselect(
            "Instrument",
            all_inst_labels,
            default=default_inst,
            placeholder="All instruments",
            key="sig_filter_inst",
            label_visibility="collapsed",
        )
    with f2:
        sel_decisions = st.multiselect(
            "Decision", ["BUY", "SELL", "HOLD", "CLOSE"],
            placeholder="All decisions",
            key="sig_filter_decision",
            label_visibility="collapsed",
        )
    with f3:
        sel_type = st.selectbox(
            "Type", ["All", "Entry only", "Exit only"],
            key="sig_filter_type",
            label_visibility="collapsed",
        )
    with f4:
        sel_bot = st.selectbox(
            "Bot", ["All bots"] + _bot_keys,
            format_func=lambda k: "All bots" if k == "All bots" else _bot_display_name(k),
            key="sig_filter_bot",
            label_visibility="collapsed",
        )

    iids     = [ALL_INSTRUMENTS[l] for l in sel_instruments if l in ALL_INSTRUMENTS] or None
    stype    = {"Entry only": "entry", "Exit only": "exit"}.get(sel_type or "All")
    bot_uuid = _key_to_uuid.get(sel_bot) if sel_bot and sel_bot != "All bots" else None
    records = signal_log.load(
        instrument_ids=iids,
        signal_type=stype,
        decisions=sel_decisions or None,
        bot_id=bot_uuid,
        limit=100,
    )

    if not records:
        st.info("No signals match the current filters.")
    else:
        st.caption(f"Showing {len(records)} of {total} total (capped at 100)")
        for rec in records:
            _render_signal_record(rec)


@st.fragment(run_every=PORTFOLIO_REFRESH_SEC)
def portfolio_live_fragment() -> None:
    if st.session_state.get("main_nav") != "Portfolio":
        return
    is_demo = st.session_state.get("is_demo", True)
    try:
        # Read the background-polled eToro positions (refreshed every ~4s) — keeps
        # the heavy REST + rates enrichment OFF the render path so the tab stays
        # responsive.  Manual closes update the cache optimistically on success.
        positions = positions_cache.get_positions()
        if not positions:
            # Never block the render path on a synchronous portfolio REST call —
            # the background poller fills the cache every ~4s.  Use Refresh for
            # an immediate pull.
            st.caption("Loading positions… (background poller, usually under 4s)")
            return
        render_portfolio_with_close(positions, is_demo)
    except PermissionError:
        permission_error("Portfolio")
    except Exception as exc:
        st.error(f"Could not load positions from eToro: {exc}")


# ── Tabs ──────────────────────────────────────────────────────────────────────


def _refresh_global_feed_status() -> None:
    """Header feed pill on every tab — derive from tick manager, not Trading tab only."""
    iid = st.session_state.get("engine_instrument_id")
    if not iid:
        cfg = trading_engine.get_config()
        if cfg:
            iid = cfg.instrument_id
    if iid:
        st.session_state["feed_live"] = compute_feed_live(iid)


def _show_engine_notifications() -> None:
    for note in engine_notify.drain():
        if note.kind == "trade_error":
            st.error(note.message)
        elif note.kind == "trade_open":
            st.toast(note.message, icon="📈")
        elif note.kind == "trade_close":
            st.toast(note.message, icon="✅")
        else:
            st.toast(note.message)


# ══════════════════════════════════════════════════════════════════════════════
# Performance & Lessons — evidence from the durable trade journal
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_pf(pf: float) -> str:
    return "∞" if pf == float("inf") else f"{pf:.2f}"


_PERF_TABLE_CSS = """
<style>
.perf-table-wrap{max-width:100%;overflow-x:auto;margin:0.15rem 0 0.6rem;}
table.perf-table{width:100%;border-collapse:collapse;font-size:0.86rem;margin:0;}
table.perf-table th{text-align:right;padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.18);
  color:#9aa4b2;font-weight:600;white-space:nowrap;}
table.perf-table th:first-child,table.perf-table td:first-child{text-align:left;}
table.perf-table td{text-align:right;padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.06);
  white-space:nowrap;}
table.perf-table tr:hover td{background:rgba(255,255,255,0.03);}
</style>
"""


def _render_table(df: "pd.DataFrame", *, empty_msg: str = "No data yet.") -> None:
    """Render a small summary DataFrame as a static HTML table.

    st.dataframe (the glide-data-grid component) intermittently paints blank in
    this Streamlit/pyarrow build — the data and Arrow schema are valid and the
    render completes without error, yet the grid shows nothing.  A plain HTML
    table always renders, so the Performance tables use this instead."""
    if df is None or getattr(df, "empty", True):
        st.caption(empty_msg)
        return
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(
            f"<td>{'—' if (v is None or (isinstance(v, float) and pd.isna(v))) else v}</td>"
            for v in row
        )
        body_rows.append(f"<tr>{cells}</tr>")
    html = (
        # Wrapper contains the nowrap table inside its st.column: wide content
        # scrolls horizontally WITHIN the column instead of bleeding across
        # into the neighbouring table (the side-by-side gains/losses overlap).
        "<div class='perf-table-wrap'>"
        "<table class='perf-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _bucket_df(group: dict) -> "pd.DataFrame":
    """Turn a trade_journal group-aggregate dict into a sorted display table."""
    rows = []
    for name, s in group.items():
        if not s.get("n"):
            continue
        rows.append({
            "Bucket":   name,
            "Trades":   s["n"],
            "Win %":    round(s["win_rate"] * 100, 1),
            "Net $":    round(s["total_pnl"], 2),
            "Exp. $/trade": round(s["expectancy"], 2),
            "PF":       _fmt_pf(s["profit_factor"]),
            "Avg hold (m)": round(s["avg_hold_min"], 1),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Net $", ascending=False).reset_index(drop=True)
    return df


def _strategy_winrate_df(by_strategy: dict) -> "pd.DataFrame":
    """Win rate per strategy — includes EVERY strategy your bots are configured
    to use, even ones with 0 trades, so it's obvious which haven't fired yet."""
    configured = {s.strategy for s in instrument_config.load_specs()}
    names = strategies.display_names()
    rows = []
    for k in sorted(configured | set(by_strategy.keys())):
        agg = by_strategy.get(k)
        nn = agg["n"] if agg else 0
        rows.append({
            "Strategy":     names.get(k, k),
            "Trades":       nn,
            "Win %":        round(agg["win_rate"] * 100, 1) if nn else None,
            "Net $":        round(agg["total_pnl"], 2) if nn else 0.0,
            "Exp. $/trade": round(agg["expectancy"], 2) if nn else None,
            "PF":           _fmt_pf(agg["profit_factor"]) if nn else "—",
            "Avg hold (m)": round(agg["avg_hold_min"], 1) if nn else None,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Trades", "Net $"], ascending=False).reset_index(drop=True)
    return df


def _etoro_dt(val) -> "datetime | None":
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _etoro_trade_to_journal_record(t: dict) -> "dict | None":
    """Map one eToro closed-trade dict (History API) to a trade-journal record.

    eToro history lacks our decision context (strategy/confidence/exec-risk), so
    those are left blank — but direction, prices, P&L, holding time and hour are
    all present and drive the analysis.
    """
    iid = t.get("instrumentId") or t.get("instrument_id")
    try:
        iid_int = int(iid) if iid is not None else 0
    except (TypeError, ValueError):
        iid_int = 0

    is_buy = t.get("isBuy")
    direction = "LONG" if is_buy is True else ("SHORT" if is_buy is False else "")

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    entry = _f(t.get("openRate"))
    exit_ = _f(t.get("closeRate"))
    pnl_usd = _f(t.get("netProfit"))
    invested = _f(t.get("investment")) or 0.0
    pid = t.get("positionId") or t.get("position_id") or t.get("positionID")
    open_dt = _etoro_dt(t.get("openTimestamp"))
    close_dt = _etoro_dt(t.get("closeTimestamp"))

    if pnl_usd is None or entry is None:
        return None  # not enough to analyse

    holding_min = (
        (close_dt - open_dt).total_seconds() / 60.0 if (open_dt and close_dt) else 0.0
    )
    pnl_pct = (pnl_usd / invested * 100.0) if invested else 0.0
    if direction == "SHORT" and exit_ is not None:
        pnl_abs = entry - exit_
    elif exit_ is not None:
        pnl_abs = exit_ - entry
    else:
        pnl_abs = 0.0
    label = INSTRUMENT_ID_TO_LABEL.get(iid_int, "") if iid_int else ""

    return {
        "ts":            close_dt.isoformat() if close_dt else "",
        "entry_time":    open_dt.isoformat() if open_dt else "",
        "exit_time":     close_dt.isoformat() if close_dt else "",
        "instrument_id": iid_int,
        "instrument_label": label,
        "bot_id":        "",
        "direction":     direction,
        "strategy":      "etoro",      # decision context unknown for history trades
        "signal":        "",
        "confidence":    0,
        "entry_price":   entry,
        "exit_price":    exit_ or 0.0,
        "entry_spread":  0.0,
        "slippage_pct":  0.0,
        "trade_amount":  invested,
        "exec_risk":     "",
        "net_edge_pct":  0.0,
        "reason":        "etoro",
        "holding_min":   round(holding_min, 2),
        "pnl_abs":       round(pnl_abs, 6),
        "pnl_dollars":   round(pnl_usd, 4),
        "pnl_pct":       round(pnl_pct, 4),
        "win":           pnl_usd > 0,
        "hour":          (open_dt or close_dt).hour if (open_dt or close_dt) else 0,
        "etoro_position_id": pid,
    }


def _backfill_journal_from_etoro(is_demo: bool) -> int:
    """Fetch eToro closed-trade history and ingest it into the trade journal."""
    trades = fetch_all_etoro_trade_history(ALL_HISTORY_START, demo=is_demo)
    records = [r for t in trades if (r := _etoro_trade_to_journal_record(t))]
    return trade_journal.add_external_records(records)


_PERF_HIGHLIGHT_N = 10


def _perf_short_label(label: str) -> str:
    text = (label or "").strip()
    if "  (" in text:
        return text.split("  (")[0].strip()
    return text or "—"


def _perf_trade_rows_df(records: list[dict]) -> "pd.DataFrame":
    """Format journal records for Performance standout-trade tables."""
    if not records:
        return pd.DataFrame()
    names = strategies.display_names()
    rows = []
    for r in records:
        exit_dt = _parse_journal_dt(r.get("exit_time") or r.get("ts"))
        closed = timez.fmt(exit_dt, "%b %d %H:%M") if exit_dt else "—"
        strat_key = (r.get("strategy") or "").strip()
        rows.append({
            "Instrument": _perf_short_label(r.get("instrument_label") or ""),
            "Strategy":   names.get(strat_key, strat_key or "—"),
            "Dir":        r.get("direction") or "—",
            "P&L $":      round(float(r.get("pnl_dollars") or 0), 2),
            "P&L %":      round(float(r.get("pnl_pct") or 0), 2),
            "Hold (m)":   round(float(r.get("holding_min") or 0), 1),
            "Exit":       r.get("reason") or "—",
            "Closed":     closed,
        })
    return pd.DataFrame(rows)


def _perf_top_trades_df(
    records: list[dict],
    *,
    sort_key: str,
    reverse: bool = False,
    limit: int = _PERF_HIGHLIGHT_N,
) -> "pd.DataFrame":
    ranked = sorted(
        records,
        key=lambda r: float(r.get(sort_key) or 0),
        reverse=reverse,
    )
    return _perf_trade_rows_df(ranked[:limit])


def render_performance_lessons() -> None:
    """Dashboard view: what the bot has learned from its own closed trades."""
    _is_demo = st.session_state.get("is_demo", True)
    st.markdown(_PERF_TABLE_CSS, unsafe_allow_html=True)
    head, ctrl = st.columns([3, 1.5], vertical_alignment="center")
    with head:
        st.subheader("Performance & Lessons")
        st.caption(
            "Evidence from the durable trade journal — drives the LLM's inline "
            "memory and the evidence-based entry guard. Survives restarts."
        )
    with ctrl:
        if st.button(
            "⬇ Import eToro history", key="perf_import", use_container_width=True,
            help="Backfill this view from your eToro closed-trade history "
                 f"({'demo' if _is_demo else 'real'} account).",
        ):
            with st.spinner("Importing eToro trade history…"):
                try:
                    n = _backfill_journal_from_etoro(_is_demo)
                    if n:
                        st.success(f"Imported {n} trade(s) from eToro.")
                    else:
                        st.info("No new trades to import (already up to date).")
                except PermissionError:
                    permission_error("Trade History")
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    total = trade_journal.total_count()
    if total == 0:
        ui.empty_state(
            "🧠",
            "No closed trades journaled yet",
            "Each time a bot closes a trade it is recorded here with full decision "
            "context, and the LLM starts learning from its own track record. To see "
            "data right now, click **⬇ Import eToro history** above to backfill from "
            "your existing eToro closed trades.",
        )
        return

    st.caption(
        f"Close dates in **{timez.active_name()}** ({timez.abbrev()}) — "
        "same periods as the P&L tab."
    )
    pf_head, pf_ctrl = st.columns([1.2, 3.8], vertical_alignment="center")
    with pf_head:
        st.caption("Period")
    with pf_ctrl:
        pfc1, pfc2 = st.columns([3.2, 2.0], vertical_alignment="center")
        with pfc1:
            perf_period = st.segmented_control(
                "Period",
                options=list(_PNL_PERIOD_OPTIONS),
                default=_DEFAULT_PERIOD,
                key="perf_period_mode",
                label_visibility="collapsed",
            )
        with pfc2:
            perf_custom_start = perf_custom_end = None
            if (perf_period or st.session_state.get("perf_period_mode", _DEFAULT_PERIOD)) == "Custom":
                pfd1, pfd2 = st.columns(2)
                with pfd1:
                    perf_custom_start = st.date_input(
                        "From",
                        value=datetime.now(timez.active_tz()).date() - timedelta(days=7),
                        min_value=ALL_HISTORY_START,
                        key="perf_custom_start",
                        label_visibility="collapsed",
                    )
                with pfd2:
                    perf_custom_end = st.date_input(
                        "To",
                        value=datetime.now(timez.active_tz()).date(),
                        min_value=ALL_HISTORY_START,
                        key="perf_custom_end",
                        label_visibility="collapsed",
                    )

    _perf_mode = perf_period or st.session_state.get("perf_period_mode", _DEFAULT_PERIOD)
    period_lbl = _journal_period_label(
        _perf_mode,
        custom_start=perf_custom_start,
        custom_end=perf_custom_end,
    )
    period_rows = _filter_journal_period(
        trade_journal.closed_records(),
        _perf_mode,
        custom_start=perf_custom_start,
        custom_end=perf_custom_end,
    )

    if not period_rows:
        st.info(f"No journaled trades in this period ({period_lbl}).")
        return

    # Optional instrument filter (instruments that traded in the selected period)
    labels = sorted({
        r.get("instrument_label", "") for r in period_rows
        if r.get("instrument_label")
    })
    sel = st.selectbox(
        "Instrument", ["All instruments"] + labels, key="perf_instrument_filter",
    )
    iid_filter = None
    if sel != "All instruments":
        iid_filter = ALL_INSTRUMENTS.get(sel)

    stats = trade_journal.performance_stats(instrument_id=iid_filter, rows=period_rows)
    o = stats["overall"]

    if not o["n"]:
        st.info(f"No journaled trades for this instrument in {period_lbl}.")
        return

    # ── Headline metrics ──────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f"Trades ({period_lbl})", o["n"])
    m2.metric("Win rate", f"{o['win_rate']*100:.0f}%")
    m3.metric("Profit factor", _fmt_pf(o["profit_factor"]))
    m4.metric("Expectancy", f"${o['expectancy']:+.2f}/trade")
    m5.metric("Net P&L", f"${o['total_pnl']:+.2f}")

    # ── Standout trades in the selected period ────────────────────────────────
    st.markdown(f"##### Standout trades · {period_lbl}")
    st.caption(
        f"Top {_PERF_HIGHLIGHT_N} biggest gains, biggest losses, and longest holds "
        f"(close time in **{timez.active_name()}**)."
    )
    sg1, sg2 = st.columns(2)
    with sg1:
        st.caption("Biggest gains")
        _render_table(
            _perf_top_trades_df(period_rows, sort_key="pnl_dollars", reverse=True),
            empty_msg="No trades in this period.",
        )
    with sg2:
        st.caption("Biggest losses")
        _render_table(
            _perf_top_trades_df(period_rows, sort_key="pnl_dollars", reverse=False),
            empty_msg="No trades in this period.",
        )
    st.caption("Longest holds")
    _render_table(
        _perf_top_trades_df(period_rows, sort_key="holding_min", reverse=True),
        empty_msg="No trades in this period.",
    )

    # ── Win rate by strategy (every configured strategy, incl. 0-trade) ───────
    st.markdown("##### Win rate by strategy")
    st.caption(
        "Every strategy your bots use. **0 trades** = that strategy hasn't opened "
        "a position yet (trend strategies like Supertrend/MA-cross only fire on a "
        "flip/cross, so they trade rarely)."
    )
    _render_table(_strategy_winrate_df(stats["by_strategy"]))

    # ── Lessons (exactly what the LLM is told) ────────────────────────────────
    mem = trade_journal.llm_memory_block(
        iid_filter if iid_filter is not None else (
            ALL_INSTRUMENTS.get(labels[0]) if labels else 0
        ),
        strategy="llm",
    )
    with st.container(border=True):
        st.markdown("##### 🧠 Lessons fed to the LLM")
        st.caption(
            "Always from the full journal (what the bot actually sees at entry) — "
            f"not limited to **{period_lbl}**."
        )
        if mem:
            st.text(mem)
        else:
            st.caption(
                "Not enough history on a single instrument/strategy yet to surface "
                "a memory block (needs a handful of trades). Aggregate stats below "
                "are already accumulating."
            )

    # ── Breakdowns ────────────────────────────────────────────────────────────
    st.markdown(f"##### Winning vs losing patterns · {period_lbl}")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("By strategy")
        _render_table(_bucket_df(stats["by_strategy"]))
        st.caption("By confidence (low <55 · mid · high ≥70)")
        _render_table(_bucket_df(stats["by_confidence"]))
        st.caption("By exit reason")
        _render_table(_bucket_df(stats["by_reason"]))
    with c2:
        st.caption("By direction")
        _render_table(_bucket_df(stats["by_direction"]))
        st.caption("By holding time")
        _render_table(_bucket_df(stats["by_holding"]))
        st.caption("By execution-risk tier at entry")
        _render_table(_bucket_df(stats["by_exec_risk"]))

    with st.expander("By hour of day (UTC)"):
        _render_table(_bucket_df(stats["by_hour"]))


# ══════════════════════════════════════════════════════════════════════════════
# Logs — live container logs (dashboard + visual-bot)
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_visual_bot_logs(
    level: str, query: str, limit: int, *, force: bool = False,
) -> list[dict]:
    """Pull recent visual-bot logs from its /logs endpoint."""
    now = time.time()
    cache = st.session_state.get("_vbot_logs_cache")
    if not force and cache and now - cache[0] < LOGS_REFRESH_SEC:
        raw = cache[1]
    else:
        try:
            resp = http_requests.get(
                f"{VISUAL_BOT_URL}/logs", params={"limit": 2000}, timeout=2.5,
            )
            raw = resp.json().get("records", []) if resp.ok else []
        except Exception:
            raw = []
        st.session_state["_vbot_logs_cache"] = (now, raw)

    thr   = log_buffer._LEVEL_ORDER.get(level, 0) if level and level != "ALL" else 0
    q     = (query or "").lower()
    out: list[dict] = []
    for r in raw:
        if thr and log_buffer._LEVEL_ORDER.get(r.get("level", ""), 0) < thr:
            continue
        if q and q not in r.get("msg", "").lower() and q not in r.get("logger", "").lower():
            continue
        try:
            ts = datetime.fromisoformat(r["ts"])
        except Exception:
            ts = datetime.now(timezone.utc)
        out.append({
            "ts": ts, "level": r.get("level", ""), "logger": r.get("logger", ""),
            "msg": r.get("msg", ""), "source": "visual-bot",
        })
    return out[-limit:]


def _format_log_lines(records: list[dict]) -> str:
    return "\n".join(
        f"{timez.fmt(r['ts'], '%H:%M:%S')} "
        f"{r.get('source', ''):<10} {r['level']:<7} {r['logger']}: {r['msg']}"
        for r in records
    )


LOGS_REFRESH_SEC = 3.0


@st.fragment(run_every=LOGS_REFRESH_SEC)
def logs_fragment() -> None:
    slot = st.empty()
    if st.session_state.get("main_nav") != "Logs":
        slot.empty()
        return

    with slot.container():
        _render_logs_tab_body()


def _render_logs_tab_body() -> None:
    st.subheader("Container logs")
    st.caption(
        f"Live logs from the dashboard and visual-bot containers "
        f"(auto-refresh every {LOGS_REFRESH_SEC:.0f}s)."
    )

    c1, c2, c3, c4 = st.columns([1.1, 1, 1.6, 0.8])
    with c1:
        source = st.selectbox("Source", ["Both", "Dashboard", "Visual-bot"], key="_logs_source")
    with c2:
        level = st.selectbox(
            "Min level", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"], index=2, key="_logs_level",
        )
    with c3:
        query = st.text_input("Filter", key="_logs_query", placeholder="search message or logger…")
    with c4:
        max_lines = st.selectbox("Lines", [200, 500, 1000, 2000], index=1, key="_logs_max")

    records: list[dict] = []
    if source in ("Dashboard", "Both"):
        for r in log_buffer.get_records(level=level, query=query, limit=max_lines):
            records.append({**r, "source": "dashboard"})
    if source in ("Visual-bot", "Both"):
        records.extend(
            _fetch_visual_bot_logs(level, query, max_lines, force=True),
        )

    records.sort(key=lambda r: r["ts"])
    records = records[-max_lines:]
    st.session_state["_logs_last_refresh"] = datetime.now(timez.active_tz())
    st.caption(
        f"{len(records)} line(s) · last refresh "
        f"{st.session_state['_logs_last_refresh'].strftime('%H:%M:%S')} "
        f"({timez.abbrev()})"
    )

    bcol1, bcol2, _ = st.columns([1, 1, 4])
    with bcol1:
        text = _format_log_lines(records)
        st.download_button(
            "Download", text or "", file_name="etoro-logs.txt",
            mime="text/plain", use_container_width=True,
        )
    with bcol2:
        if st.button("Clear dashboard buffer", use_container_width=True):
            log_buffer.clear()
            st.rerun()

    st.code(text or "No log records captured yet.", language="log")


_timed("_sync_global_light", _sync_global_light)
_timed("refresh_global_feed_status", _refresh_global_feed_status)
_timed("show_engine_notifications", _show_engine_notifications)

ui.render_header(
    is_demo,
    trading_active=trading_engine.is_trading_active(),
    live_connected=st.session_state.get("feed_live", False),
)

_NAV_OPTIONS = [
    "Trading", "Bots", "Portfolio", "History", "P&L", "Performance",
    "Signals", "Strategies", "Settings", "Watchlists", "Logs",
]
page = st.segmented_control(
    "Section",
    options=_NAV_OPTIONS,
    default="Trading",
    key="main_nav",
    label_visibility="collapsed",
)
if not page:
    page = st.session_state.get("main_nav", "Trading")

_prev_nav = st.session_state.get("_main_nav_prev")
st.session_state["_main_nav_prev"] = page

# Tab switches no longer trigger an extra st.rerun() — that doubled every click
# and made navigation feel hung.  Fragments already no-op when main_nav != tab.

# ══════════════════════════════════════════════════════════════════════════════
# Trading (only mounted when active — prevents hidden tab fragments)
# ══════════════════════════════════════════════════════════════════════════════

_nav_body = st.empty()
with _nav_body.container():
    if page == "Trading":
        _restore_trading_toolbar_widgets()

        # ── Bot selector — the Trading tab is bound to exactly ONE bot ─────────────
        # Everything below (chart interval, strategy, signal panel, prompt) is driven
        # by the selected bot.  First visit defaults to the first bot; the Bots-tab
        # "View →" button selects a specific one.  Switching here moves the whole tab.
        _bots = _trading_bot_options()
        if not _bots:
            ui.empty_state("🤖", "No bots configured",
                           "Add bots to instruments.toml and rebuild.")
            st.stop()
        _bot_keys    = [b["key"] for b in _bots]
        _display_for = {b["key"]: b["display"] for b in _bots}

        # Resolve the active bot: prior selection / View→, else default to the first.
        _active = st.session_state.get("engine_active_bot_key")
        if _active not in _bot_keys:
            _active = _bot_keys[0]
            st.session_state["engine_active_bot_key"] = _active
        # Mirror the active bot into the widget BEFORE it renders so an external
        # change (Bots-tab View →) is always shown.  The widget stores the STABLE bot
        # key (format_func renders the label), so changing a bot's strategy never
        # orphans the selection.  User picks flow back via the on_change callback.
        st.session_state["_trading_bot_select"] = _active

        with st.container(border=True):
            t1, t2, t3 = st.columns([3.4, 1, 1.6])
            with t1:
                st.selectbox(
                    "Bot", _bot_keys,
                    format_func=lambda k: _display_for.get(k, k),
                    key="_trading_bot_select",
                    on_change=_on_trading_bot_pick,
                    label_visibility="collapsed",
                    help="The Trading tab follows this bot — its instrument, interval, "
                         "strategy, chart and signals.",
                )
            with t2:
                candle_count = st.number_input(
                    "Candles", min_value=20, max_value=1000, step=50,
                    key="_candle_count", label_visibility="collapsed",
                )
            with t3:
                _tz_opts = list(timez.COMMON_ZONES)
                _cur_tz  = st.session_state.get("display_tz", "Asia/Dubai")
                if _cur_tz not in _tz_opts:
                    _tz_opts.insert(0, _cur_tz)
                st.session_state.setdefault("_tz_select", _cur_tz)
                _picked_tz = st.selectbox(
                    "Timezone", _tz_opts, key="_tz_select",
                    format_func=lambda z: f"🕒 {z}",
                    label_visibility="collapsed",
                    help="Display timezone — every date/time in the app is shown in this "
                         "zone. Stored/recorded times stay in UTC.",
                )
                if _picked_tz != _cur_tz:
                    st.session_state["display_tz"] = _picked_tz
                    timez.set_active(_picked_tz)
                    runtime_persist.save(dict(st.session_state))
                    st.rerun()

        # engine_active_bot_key is authoritative — the on_change callback already
        # updated it on a user pick.  NEVER override it from the widget's return value
        # (a cleared/stale widget would otherwise revert a View → selection).
        _active = st.session_state["engine_active_bot_key"]
        if _active not in _bot_keys:
            _active = _bot_keys[0]
            st.session_state["engine_active_bot_key"] = _active

        _bot = _bots[_bot_keys.index(_active)]
        instrument_id    = _bot["instrument_id"]
        selected_label   = _bot["label"]
        interval_label   = _bot["interval"]
        interval_seconds = _bot["interval_secs"]
        api_name         = INTERVALS[interval_label][0] if interval_label in INTERVALS else "OneMinute"

        live_mode = st.session_state.get("live_feed", False)
        st.session_state.update({
            "engine_instrument_id":   instrument_id,
            "engine_selected_label":  selected_label,
            "engine_interval_label":  interval_label,
            "engine_interval_seconds": interval_seconds,
            "engine_api_name":        api_name,
            "engine_candle_count":    int(candle_count),
        })

        if not instrument_id:
            ui.empty_state("🔍", "Bot not resolved",
                           "This bot's instrument could not be resolved from eToro.")
            st.stop()

        sync_background_engine()
        feed_live = compute_feed_live(instrument_id)
        st.session_state["feed_live"] = feed_live

        a1, a2 = st.columns([1, 2])
        with a1:
            trading_engine_status_fragment(instrument_id, live_mode)
        with a2:
            trading_feed_badge_fragment(instrument_id, selected_label, interval_label)

        # ── Historical candles — auto-load on instrument / interval / count change
        hist_cache_key = f"hist_{instrument_id}_{api_name}_{int(candle_count)}"
        hist_df = st.session_state.get(hist_cache_key, pd.DataFrame())
        if hist_df.empty:
            hist_df, _hist_status = load_hist_candles_async(
                instrument_id, api_name, int(candle_count),
                interval_seconds=interval_seconds, bot_id=_active,
            )
            if _hist_status == "loading":
                hub_ready = market_data_hub.get_snapshot(bot_id=_active) is not None
                if not (live_mode and hub_ready):
                    st.caption(
                        f"Fetching {selected_label} candle history from eToro "
                        f"({interval_label})… usually a few seconds."
                    )
                _hist_load_poller(hist_cache_key)   # self-reruns when ready
            elif _hist_status == "error":
                st.warning(
                    "Couldn't load historical candles right now (the API may be "
                    "rate-limited) — the live feed will still fill the chart."
                )
        if not hist_df.empty:
            trading_engine.set_hist(hist_df, instrument_id, bot_id=_active)

        # ── LIVE mode — chart on the left, the FULL right rail (General Stats,
        # Analyse, Live Quote, Auto-Trade) stacked in ONE column.  General Stats and
        # the Analyse button stay OUTSIDE the fragment so the run_every timer can't
        # drop the button click; the live Quote/Auto-Trade panels are their own
        # fragment below them.
        if live_mode:
            col_chart, col_side = st.columns([11, 5], gap="medium")
            with col_chart:
                live_trading_chart_fragment()
                render_chart_position_close(instrument_id, is_demo)
            with col_side:
                _render_live_analyse_button()
                live_trading_side_fragment()

        elif not hist_df.empty:
            df_show = hist_df.tail(int(candle_count))
            fig = build_figure(
                df_show, pd.DataFrame(), selected_label, interval_label,
                open_trade=_active_bot_open_trade(instrument_id),
            )
            r = df_show.iloc[-1]
            prev_c = float(df_show.iloc[-2]["Close"]) if len(df_show) > 1 else float(r["Open"])
            pct_ch = (float(r["Close"]) - prev_c) / prev_c * 100 if prev_c else 0

            col_c, col_s = st.columns([11, 5], gap="medium")
            with col_c:
                st.plotly_chart(fig, width="stretch", key="chart_static")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Open",  f"{r['Open']:.5f}")
                m2.metric("High",  f"{r['High']:.5f}")
                m3.metric("Low",   f"{r['Low']:.5f}")
                m4.metric("Close", f"{r['Close']:.5f}", delta=f"{pct_ch:+.2f}%")
                static_positions_fragment()
                render_chart_position_close(instrument_id, is_demo)

            with col_s:
                _render_static_analyse_button(
                    instrument_id, df_show, float(r["Close"]),
                    selected_label, interval_label,
                    position_open=_active_bot_open_trade(instrument_id) is not None,
                )
                static_right_panel_fragment(
                    instrument_id,
                    selected_label,
                    interval_label,
                    float(df_show["High"].max()),
                    float(df_show["Low"].min()),
                    float(r["Close"]),
                    df_show,
                )

        else:
            ui.empty_state(
                "📉", "No data",
                "No candles returned for this instrument and interval.",
            )


    # ══════════════════════════════════════════════════════════════════════════════
    # Portfolio (only mounted when active)
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Bots":
        bots_live_fragment()

    elif page == "Portfolio":
        p_h1, p_h2 = st.columns([4, 1])
        with p_h1:
            st.subheader("Portfolio")
            st.caption(
                f"Open positions fetched live from your eToro **{'demo' if is_demo else 'real'}** "
                "account — this is exactly what eToro reports."
            )
        with p_h2:
            if st.button("Refresh", key="refresh_pos", width="stretch"):
                bump_portfolio_cache()
                positions_cache.refresh_if_stale(client, is_demo, force=True)

        portfolio_live_fragment()

        # ── Debug: see exactly what eToro returns ─────────────────────────────────
        # If a position looks like a phantom, open this to compare the raw eToro
        # payload against what's displayed — pinpoints eToro-side vs parsing issues.
        with st.expander("🔧 Debug — raw eToro portfolio response"):
            if st.button("Fetch raw eToro response", key="pf_debug_fetch"):
                try:
                    raw = client.get_portfolio_raw(demo=is_demo)
                    shown = client.get_open_positions(demo=is_demo)
                    st.caption(
                        f"eToro returned this payload for the **{'demo' if is_demo else 'real'}** "
                        f"account. The Portfolio shows **{len(shown)}** genuine open position(s)."
                    )
                    st.markdown("**Displayed positions (id · instrument · units · amount):**")
                    st.write([
                        {
                            "position_id": p.get("position_id"),
                            "instrument_id": p.get("instrument_id"),
                            "symbol": p.get("symbol") or p.get("name"),
                            "direction": p.get("direction"),
                            "units": p.get("units"),
                            "amount": p.get("amount"),
                        }
                        for p in shown
                    ])
                    st.markdown("**Raw eToro payload:**")
                    st.json(raw)
                except Exception as exc:
                    st.error(f"Raw fetch failed: {exc}")


    # ══════════════════════════════════════════════════════════════════════════════
    # Trade History
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "History":
        st.subheader("Trade History")
        st.caption(
            f"Closed trades on eToro **{'demo' if is_demo else 'real'}** · "
            f"close dates in **{timez.active_name()}** ({timez.abbrev()}) · "
            f"refreshes on open and every {HISTORY_REFRESH_SEC}s"
        )

        h_head, h_ctrl = st.columns([1.2, 3.8], vertical_alignment="center")
        with h_head:
            st.caption("Period")
        with h_ctrl:
            hc1, hc2, hc3 = st.columns([3.2, 2.0, 0.55], vertical_alignment="center")
            with hc1:
                hist_period = st.segmented_control(
                    "Period",
                    options=list(_PNL_PERIOD_OPTIONS),
                    default=_DEFAULT_PERIOD,
                    key="hist_period_mode",
                    label_visibility="collapsed",
                )
            with hc2:
                hist_custom_start = hist_custom_end = None
                if (hist_period or st.session_state.get("hist_period_mode", _DEFAULT_PERIOD)) == "Custom":
                    hd1, hd2 = st.columns(2)
                    with hd1:
                        hist_custom_start = st.date_input(
                            "From",
                            value=datetime.now(timez.active_tz()).date() - timedelta(days=7),
                            min_value=ALL_HISTORY_START,
                            key="hist_custom_start",
                            label_visibility="collapsed",
                        )
                    with hd2:
                        hist_custom_end = st.date_input(
                            "To",
                            value=datetime.now(timez.active_tz()).date(),
                            min_value=ALL_HISTORY_START,
                            key="hist_custom_end",
                            label_visibility="collapsed",
                        )
            with hc3:
                refresh_hist = st.button(
                    "Refresh", key="hist_refresh", use_container_width=True,
                )

        _hist_mode = hist_period or st.session_state.get("hist_period_mode", _DEFAULT_PERIOD)
        period_lbl = _journal_period_label(
            _hist_mode,
            custom_start=hist_custom_start,
            custom_end=hist_custom_end,
        )
        min_date = _period_min_fetch_date(_hist_mode, custom_start=hist_custom_start)

        st.session_state["hist_min_date"] = min_date
        fetch_key = _history_fetch_key(is_demo, min_date)
        if refresh_hist or st.session_state.get("etoro_hist_key") != fetch_key:
            st.session_state.pop("etoro_hist_trades", None)
            st.session_state.pop("etoro_hist_key", None)
            st.session_state.pop("hist_load_error", None)

        if not is_demo:
            st.caption("Tip: select **Demo** in the sidebar — demo trade history is available for auto-traded positions.")

        st.session_state["_hist_show_spinner"] = refresh_hist
        history_tab_fragment(
            is_demo,
            min_date,
            period_mode=_hist_mode,
            custom_start=hist_custom_start,
            custom_end=hist_custom_end,
            period_lbl=period_lbl,
        )


    # ══════════════════════════════════════════════════════════════════════════════
    # P&L
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "P&L":
        st.subheader("P&L Summary")
        st.caption(
            f"Account performance · eToro **{'demo' if is_demo else 'real'}** · "
            f"times in **{timez.active_name()}** ({timez.abbrev()})"
        )

        _render_today_bot_pnl_banner()

        st.markdown("#### Closed Trades")
        p_head, p_ctrl = st.columns([1.2, 3.8], vertical_alignment="center")
        with p_head:
            st.caption("Period")
        with p_ctrl:
            pc1, pc2 = st.columns([3.2, 2.0], vertical_alignment="center")
            with pc1:
                pnl_period = st.segmented_control(
                    "Period",
                    options=list(_PNL_PERIOD_OPTIONS),
                    default=_DEFAULT_PERIOD,
                    key="pnl_period_mode",
                    label_visibility="collapsed",
                )
            with pc2:
                pnl_custom_start = pnl_custom_end = None
                if (pnl_period or st.session_state.get("pnl_period_mode", _DEFAULT_PERIOD)) == "Custom":
                    cd1, cd2 = st.columns(2)
                    with cd1:
                        pnl_custom_start = st.date_input(
                            "From",
                            value=datetime.now(timez.active_tz()).date() - timedelta(days=7),
                            min_value=ALL_HISTORY_START,
                            key="pnl_custom_start",
                            label_visibility="collapsed",
                        )
                    with cd2:
                        pnl_custom_end = st.date_input(
                            "To",
                            value=datetime.now(timez.active_tz()).date(),
                            min_value=ALL_HISTORY_START,
                            key="pnl_custom_end",
                            label_visibility="collapsed",
                        )

        _pnl_mode = pnl_period or st.session_state.get("pnl_period_mode", _DEFAULT_PERIOD)
        if st.session_state.get("_pnl_period_prev") != _pnl_mode:
            st.session_state["_bot_trades_shown"] = _BOT_TRADES_PAGE
            st.session_state["_pnl_period_prev"] = _pnl_mode

        render_bot_session_trades(
            period_mode=_pnl_mode,
            custom_start=pnl_custom_start,
            custom_end=pnl_custom_end,
        )

        st.divider()
        st.markdown("#### eToro Account P&L")
        st.caption("Authoritative account figures pulled live from eToro.")

        if st.button("Refresh account P&L", type="primary"):
            with st.spinner("Loading account P&L…"):
                try:
                    st.session_state["_pnl_data"] = client.get_pnl(demo=is_demo)
                    st.session_state["_pnl_err"]  = None
                except PermissionError:
                    st.session_state["_pnl_err"] = "permission"
                except Exception as exc:
                    st.session_state["_pnl_err"] = str(exc)

        _pnl_err = st.session_state.get("_pnl_err")
        if _pnl_err == "permission":
            permission_error("P&L")
        elif _pnl_err:
            st.error(f"P&L error: {_pnl_err}")

        _pnl_data = st.session_state.get("_pnl_data")
        if _pnl_data:
            cp        = (_pnl_data or {}).get("clientPortfolio", {}) or {}
            positions = cp.get("positions", []) or []
            cash      = float(cp.get("credit") or 0.0)
            bonus     = float(cp.get("bonusCredit") or 0.0)
            unreal    = float(cp.get("unrealizedPnL") or 0.0)
            invested  = sum(float(p.get("amount") or 0) for p in positions)
            equity    = cash + bonus + invested + unreal

            a1, a2, a3 = st.columns(3)
            a1.metric("Equity", f"${equity:,.2f}",
                      help="Total account value: cash + invested + unrealized P&L")
            a2.metric("Available cash", f"${cash:,.2f}",
                      help="Free credit not tied up in open positions")
            a3.metric("Bonus credit", f"${bonus:,.2f}")

            b1, b2, b3 = st.columns(3)
            b1.metric("Invested", f"${invested:,.2f}",
                      help=f"Capital across {len(positions)} open position(s)")
            b2.metric("Unrealized P&L", f"${unreal:,.2f}", delta=f"{unreal:+,.2f}",
                      help="Mark-to-market on all open positions")
            b3.metric("Open positions", f"{len(positions)}")

            with st.expander("Raw JSON"):
                st.json(_pnl_data)
        else:
            st.caption("Click **Refresh account P&L** to load.")


    # ══════════════════════════════════════════════════════════════════════════════
    # Signals log
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Performance":
        _timed("render_performance_lessons", render_performance_lessons)


    elif page == "Signals":
        signals_live_fragment()


    # ══════════════════════════════════════════════════════════════════════════════
    # Strategy Guide
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Strategies":
        from views.strategy_guide import render as _render_strategy_guide
        _render_strategy_guide()
        # Backtest lives at the end of the guide: read what a strategy does,
        # then immediately replay it over real history with the live settings.
        st.markdown("---")
        from views.backtest import render as _render_backtest
        _render_backtest()


    # ══════════════════════════════════════════════════════════════════════════════
    # Settings
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Settings":
        from views.settings import render as _render_settings
        _render_settings()


    # ══════════════════════════════════════════════════════════════════════════════
    # Watchlists
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Watchlists":
        st.subheader("Watchlists")
        st.caption("Saved instrument lists from your eToro account")

        if st.button("Load watchlists", type="primary", key="wl_fetch_btn"):
            st.session_state["_wl_fetch_pending"] = True
            st.rerun()

        if st.session_state.pop("_wl_fetch_pending", False):
            with st.spinner("Loading watchlists…"):
                try:
                    wl_data = client.get_watchlists()
                    watchlists = extract_list(wl_data, "watchlists", "data", "Watchlists")
                    st.session_state["_wl_cache"] = {
                        "raw": wl_data,
                        "watchlists": watchlists,
                    }
                    st.session_state.pop("_wl_contents", None)
                except PermissionError:
                    st.session_state["_wl_cache"] = {"error": "permission"}
                except Exception as exc:
                    st.session_state["_wl_cache"] = {"error": str(exc)}
            st.rerun()

        _wl = st.session_state.get("_wl_cache")
        if _wl:
            if _wl.get("error") == "permission":
                permission_error("Watchlists")
            elif _wl.get("error"):
                st.error(f"Watchlists error: {_wl['error']}")
            elif not _wl.get("watchlists"):
                st.info("No watchlists found.")
                with st.expander("Raw response"):
                    st.json(_wl.get("raw"))
            else:
                df_wl = pd.json_normalize(_wl["watchlists"])
                id_col = next(
                    (c for c in df_wl.columns if c.lower().endswith("id")),
                    df_wl.columns[0],
                )
                name_col = next(
                    (c for c in df_wl.columns if "name" in c.lower()), id_col,
                )
                _render_table(df_wl)

                wl_opts = {
                    str(row[name_col]): str(row[id_col]) for _, row in df_wl.iterrows()
                }
                chosen = st.selectbox("Open watchlist", list(wl_opts.keys()), key="wl_pick")

                if st.button("Load contents", key="wl_contents_btn"):
                    st.session_state["_wl_contents_pending"] = (
                        wl_opts.get(chosen), chosen,
                    )
                    st.rerun()

                _pending = st.session_state.pop("_wl_contents_pending", None)
                if _pending:
                    wl_id, wl_name = _pending
                    with st.spinner(f"Loading {wl_name}…"):
                        try:
                            detail = client.get_watchlist(wl_id)
                            insts = extract_list(
                                detail, "instruments", "items", "data", "Instruments",
                            )
                            st.session_state["_wl_contents"] = {
                                "detail": detail, "insts": insts,
                            }
                        except Exception as exc:
                            st.session_state["_wl_contents"] = {"error": str(exc)}
                    st.rerun()

                _contents = st.session_state.get("_wl_contents")
                if _contents:
                    if _contents.get("error"):
                        st.error(f"Watchlist contents error: {_contents['error']}")
                    elif _contents.get("insts"):
                        _render_table(pd.json_normalize(_contents["insts"]))
                    else:
                        st.json(_contents.get("detail"))

    # ══════════════════════════════════════════════════════════════════════════════
    # Logs
    # ══════════════════════════════════════════════════════════════════════════════

    elif page == "Logs":
        logs_fragment()



# Logs tab — fragment timer only runs while this tab is mounted.

