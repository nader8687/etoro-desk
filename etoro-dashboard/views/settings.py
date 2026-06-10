"""Settings tab — user-editable exit, risk, trading, learning, and display parameters."""

from __future__ import annotations



import pandas as pd

import streamlit as st



import exit_profiles

import instrument_config

import runtime_persist

import timez

import trading_engine

import user_settings

from strategies import display_names



_KIND_LABELS = {

    "trend": "Trend / momentum",

    "mean_revert": "Mean-reverting / oscillator",

    "arb": "Arbitrage",

    "llm": "LLM",

}



_STRATEGY_BY_KIND: dict[str, list[str]] = {}

for _key, _prof in exit_profiles.PROFILES.items():

    _STRATEGY_BY_KIND.setdefault(_prof.kind, []).append(_key)





def _sync_session(*, demo_amount: float | None = None, display_tz: str | None = None) -> None:

    """Keep sidebar / Trading tab in sync after Settings saves."""

    if demo_amount is not None:

        st.session_state["demo_trade_amount"] = float(demo_amount)

    if display_tz is not None:

        st.session_state["display_tz"] = display_tz

        timez.set_active(display_tz)

    try:

        runtime_persist.save(dict(st.session_state))

    except Exception:

        pass





def _exit_profile_fields(kind: str, data: dict) -> dict:

    label = _KIND_LABELS.get(kind, kind)

    strategies = ", ".join(sorted(_STRATEGY_BY_KIND.get(kind, [])))

    with st.expander(f"**{label}** — {strategies}", expanded=(kind == "trend")):

        st.caption(

            "Stop-loss floor % is scaled by asset class (crypto ×1.0, stock ×0.5, …). "

            "Set take-profit to **0** to disable hard take-profit for that class."

        )

        c1, c2, c3 = st.columns(3)

        trail = c1.number_input(

            "Trailing stop %",

            min_value=0.0, max_value=20.0, step=0.1,

            value=float(data.get("trailing_stop_pct", 0.0)),

            key=f"set_exit_{kind}_trail",

            help="Legacy %-from-peak trail: closes when price pulls back this % "
                 "from the best price since entry (only after the trade has been "
                 "in profit). 0 = off.  When live ATR is available the chandelier "
                 "trail (ATR section) takes over instead.",

        )

        tp = c2.number_input(

            "Take-profit %",

            min_value=0.0, max_value=20.0, step=0.1,

            value=float(data.get("take_profit_pct", 0.0)),

            key=f"set_exit_{kind}_tp",

            help="Hard take-profit: close automatically once unrealised gain "
                 "reaches this % above entry. 0 = disabled (let winners run with "
                 "trailing stop only).  Checked every tick before trailing.",

        )

        sl = c3.number_input(

            "Stop-loss floor %",

            min_value=0.1, max_value=20.0, step=0.1,

            value=float(data.get("stop_loss_min_pct", 2.5)),

            key=f"set_exit_{kind}_sl",

            help="Minimum hard stop distance as % of entry price.  The actual stop "
                 "may be wider when ATR volatility sizing is active (see ATR stops). "
                 "Also scaled down for stocks vs crypto.",

        )

    return {

        "trailing_stop_pct": round(trail, 3),

        "take_profit_pct": round(tp, 3),

        "stop_loss_min_pct": round(sl, 3),

    }





def render() -> None:

    st.subheader("Settings")

    st.caption(

        "Changes are saved to the persistent data volume and apply on the next engine "

        "tick (risk, sizing, learning) or immediately to running bots (exit params). "

        "Rebuild is **not** required.  Hover the **?** next to any field for what it does."

    )



    cfg = user_settings.load()



    # ── Exit profiles ─────────────────────────────────────────────────────────

    st.markdown("#### Exit profiles")

    st.caption(

        "Per **strategy class** — applies to every bot using that class unless a "

        "per-bot override is set below."

    )

    with st.form("settings_exit_profiles", border=True):

        exit_out: dict[str, dict] = {}

        for kind in ("trend", "mean_revert", "arb", "llm"):

            exit_out[kind] = _exit_profile_fields(kind, cfg["exit_profiles"].get(kind, {}))

        if st.form_submit_button("Save exit profiles", type="primary"):

            user_settings.save(exit_profiles=exit_out)

            trading_engine.refresh_all_exit_params()

            st.success("Exit profiles saved — running bots updated.")

            st.rerun()

    # ── ATR stops (volatility exits) ──────────────────────────────────────────
    st.markdown("#### ATR stops (volatility exits)")
    st.caption(
        "Stops sized from live volatility — **k × ATR(14)**, Wilder smoothing. "
        "Entry stop multipliers are per strategy class (research: trend needs "
        "2.5–3×, mean-reversion ~2×, arb ~1.5×); the **chandelier trailing** "
        "stop ratchets from the peak at the golden-rule 2× and never widens. "
        "Applies to NEW entries and to every trailing check immediately; an "
        "existing position's server-side hard stop keeps its entry value."
    )
    atr_cfg = cfg.get("atr", {})
    with st.form("settings_atr", border=True):
        a1, a2 = st.columns(2)
        m_trend = a1.number_input(
            "Entry stop — trend (× ATR)", min_value=0.5, max_value=6.0, step=0.25,
            value=float(atr_cfg.get("stop_mult_trend", 2.5)),
            help="Multiplier on ATR(14) for the initial hard stop on trend/momentum "
                 "bots (supertrend, ma_crossover, macd, adx, ichimoku, orb, donchian). "
                 "Stop distance = this × current ATR%.  Higher = more room before stop-out.",
        )
        m_llm = a1.number_input(
            "Entry stop — LLM (× ATR)", min_value=0.5, max_value=6.0, step=0.25,
            value=float(atr_cfg.get("stop_mult_llm", 2.5)),
            help="ATR multiplier for LLM-driven bots.  The model decides exits each "
                 "candle; this sets how far the mechanical hard stop sits below/above "
                 "entry when the position opens.",
        )
        m_mr = a1.number_input(
            "Entry stop — mean-revert (× ATR)", min_value=0.5, max_value=6.0, step=0.25,
            value=float(atr_cfg.get("stop_mult_mean_revert", 2.0)),
            help="ATR multiplier for mean-reversion / oscillator bots (rsi, stoch_rsi, "
                 "bollinger_squeeze, mean_reversion, candlestick).  Usually tighter "
                 "than trend because bounces are expected to resolve quickly.",
        )
        m_arb = a1.number_input(
            "Entry stop — arb (× ATR)", min_value=0.5, max_value=6.0, step=0.25,
            value=float(atr_cfg.get("stop_mult_arb", 1.5)),
            help="ATR multiplier for arbitrage bots (stat_arb, rate_arb).  Smallest "
                 "multiplier — arb edges are tiny so stops must stay tight.",
        )
        m_trail = a2.number_input(
            "Chandelier trail (× ATR) — golden rule 2.0",
            min_value=0.5, max_value=6.0, step=0.25,
            value=float(atr_cfg.get("trail_mult", 2.0)),
            help="Chandelier trailing stop: distance from the best price since entry "
                 "= this × ATR%.  Active from entry, ratchets only in your favour, "
                 "recomputed each candle.  Fires as 'trailing_stop' in the journal.",
        )
        nfloor = a2.number_input(
            "Noise floor (min stop %)", min_value=0.01, max_value=2.0, step=0.01,
            value=float(atr_cfg.get("noise_floor_pct", 0.10)),
            help="Minimum stop width in calm markets.  ATR-sized stops cannot be "
                 "tighter than this % — prevents getting stopped out by spread/noise "
                 "when volatility reads very low.",
        )
        widen = a2.number_input(
            "Panic cap (× fixed floor)", min_value=1.0, max_value=6.0, step=0.5,
            value=float(atr_cfg.get("widen_max", 3.0)),
            help="Upper cap when volatility spikes.  ATR stop may widen in panics; "
                 "it will never exceed the strategy's fixed stop-loss floor × this "
                 "value (prevents runaway wide stops).",
        )
        if st.form_submit_button("Save ATR stops", type="primary"):
            user_settings.save(atr={
                "stop_mult_trend": float(m_trend),
                "stop_mult_llm": float(m_llm),
                "stop_mult_mean_revert": float(m_mr),
                "stop_mult_arb": float(m_arb),
                "trail_mult": float(m_trail),
                "noise_floor_pct": float(nfloor),
                "widen_max": float(widen),
            })
            st.success("ATR stop settings saved — applied on the next tick.")
            st.rerun()

    # ── Cash freeing ──────────────────────────────────────────────────────────
    st.markdown("#### Cash freeing")
    st.caption(
        "Guardrails for funding a strong signal when spendable cash is short: "
        "first the reserve floor relaxes (closes nothing), then the weakest "
        "open positions are partially trimmed — edge-gated and rate-limited."
    )
    cf = cfg.get("cash_freeing", {})
    with st.form("settings_cash_freeing", border=True):
        c1, c2 = st.columns(2)
        cf_edge = c1.number_input(
            "Min signal edge to free cash", min_value=0.1, max_value=1.5, step=0.05,
            value=float(cf.get("min_edge_to_free", 0.50)),
            help="Edge = strategy performance × confidence (~0–1.4). Below this, "
                 "a cash-short signal is skipped rather than touching the book.",
        )
        cf_margin = c1.number_input(
            "Edge margin over victim", min_value=0.0, max_value=1.0, step=0.05,
            value=float(cf.get("edge_margin", 0.15)),
            help="The new signal must beat a trimmed position's forward edge by "
                 "at least this much.",
        )
        cf_age = c1.number_input(
            "Min position age before trim (s)", min_value=0.0, max_value=3600.0, step=30.0,
            value=float(cf.get("min_position_age_sec", 120.0)),
            help="A position must be open at least this many seconds before it can "
                 "be chosen as a cash-freeing trim victim.  Stops brand-new entries "
                 "from being closed immediately to fund another signal.",
        )
        cf_cool = c2.number_input(
            "Trim cooldown per position (s)", min_value=0.0, max_value=3600.0, step=30.0,
            value=float(cf.get("trim_cooldown_sec", 120.0)),
            help="After a position is partially trimmed, it cannot be trimmed again "
                 "until this cooldown elapses.  Reduces churn from repeated partial closes.",
        )
        cf_frac = c2.number_input(
            "Max trim fraction of a position", min_value=0.05, max_value=0.95, step=0.05,
            value=float(cf.get("max_trim_fraction", 0.75)),
            help="Largest share of a victim position that cash-freeing may close in "
                 "one trim (e.g. 0.75 = up to 75% of units).  The rest stays open.",
        )
        cf_keep = c2.number_input(
            "Min $ left in a trimmed position", min_value=50.0, max_value=2000.0, step=50.0,
            value=float(cf.get("keep_min_usd", 200.0)),
            help="After a partial trim, at least this much notional must remain in the "
                 "position.  Works with max trim fraction to avoid dust positions.",
        )
        if st.form_submit_button("Save cash freeing", type="primary"):
            user_settings.save(cash_freeing={
                "min_edge_to_free": float(cf_edge),
                "edge_margin": float(cf_margin),
                "min_position_age_sec": float(cf_age),
                "trim_cooldown_sec": float(cf_cool),
                "max_trim_fraction": float(cf_frac),
                "keep_min_usd": float(cf_keep),
            })
            st.success("Cash-freeing settings saved — applied on the next signal.")
            st.rerun()

    st.markdown("---")



    # ── Risk manager ──────────────────────────────────────────────────────────

    st.markdown("#### Portfolio risk manager")

    risk = cfg["risk"]

    with st.form("settings_risk", border=True):

        r1, r2 = st.columns(2)

        enabled = r1.toggle(
            "Risk manager enabled",
            value=bool(risk.get("enabled", True)),
            help="Master switch for portfolio risk checks before each new entry. "
                 "When off, bots may open trades without position-count, exposure, "
                 "or drawdown limits (not recommended).",
        )

        max_pos = r1.number_input(

            "Max concurrent positions",

            min_value=1, max_value=50, step=1,

            value=int(risk.get("max_concurrent_positions", 12)),

            help="Maximum number of open bot positions at once across the whole "
                 "account.  New entries are blocked when this cap is reached.",

        )

        max_gross = r2.number_input(

            "Max gross exposure % of equity",

            min_value=10.0, max_value=100.0, step=5.0,

            value=float(risk.get("max_gross_exposure_pct", 60.0)),

            help="Sum of all open position notionals (long + short) cannot exceed "
                 "this % of account equity.  Limits total capital deployed.",

        )

        max_heat = r2.number_input(

            "Max portfolio heat % of equity",

            min_value=1.0, max_value=30.0, step=0.5,

            value=float(risk.get("max_portfolio_heat_pct", 6.0)),

            help="Maximum total $ at risk if every open stop-loss hits at once, "
                 "as a % of equity.  Keeps worst-case loss bounded.",

        )

        r3, r4 = st.columns(2)

        cl_gross = r3.number_input(

            "Max cluster gross %",

            min_value=10.0, max_value=100.0, step=5.0,

            value=float(risk.get("max_cluster_gross_pct", 45.0)),

            help="Per asset cluster (e.g. all EUR/USD bots): combined long+short "
                 "notional cannot exceed this % of equity.  Stops over-concentration.",

        )

        cl_net = r3.number_input(

            "Max cluster net %",

            min_value=5.0, max_value=50.0, step=1.0,

            value=float(risk.get("max_cluster_net_pct", 25.0)),

            help="Per cluster: net directional exposure (longs minus shorts) cap "
                 "as % of equity.  Limits one-sided bets on a single instrument.",

        )

        same_dir = r4.number_input(

            "Max same-direction per cluster",

            min_value=1, max_value=20, step=1,

            value=int(risk.get("max_same_dir_per_cluster", 6)),

            help="How many open positions in the same direction (all long or all "
                 "short) are allowed on one instrument cluster at once.",

        )

        per_asset = r4.number_input(

            "Max positions per asset",

            min_value=1, max_value=10, step=1,

            value=int(risk.get("max_positions_per_asset", 4)),

            help="Hard cap on open positions per instrument ID, regardless of "
                 "direction.  Prevents too many bots stacking on one symbol.",

        )

        r5, r6 = st.columns(2)

        block_hedge = r5.toggle(

            "Block internal hedge",

            value=bool(risk.get("block_internal_hedge", False)),

            help="Don't open opposite a larger same-asset net position.",

        )

        dd_halt = r6.number_input(

            "Daily drawdown halt %",

            min_value=1.0, max_value=20.0, step=0.5,

            value=float(risk.get("daily_drawdown_halt_pct", 5.0)),

            help="Halt NEW entries if today's realised P&L ≤ −this % of equity.",

        )

        if st.form_submit_button("Save risk limits", type="primary"):

            user_settings.save(risk={

                "enabled": enabled,

                "max_concurrent_positions": int(max_pos),

                "max_gross_exposure_pct": float(max_gross),

                "max_portfolio_heat_pct": float(max_heat),

                "max_cluster_gross_pct": float(cl_gross),

                "max_cluster_net_pct": float(cl_net),

                "max_same_dir_per_cluster": int(same_dir),

                "max_positions_per_asset": int(per_asset),

                "block_internal_hedge": block_hedge,

                "daily_drawdown_halt_pct": float(dd_halt),

            })

            st.success("Risk limits saved — apply on the next new-entry check.")

            st.rerun()



    st.markdown("---")



    # ── Trading & sizing ──────────────────────────────────────────────────────

    st.markdown("#### Trading & sizing")

    trading = cfg["trading"]

    with st.form("settings_trading", border=True):

        t1, t2 = st.columns(2)

        max_trade = t1.number_input(

            "Max trade size ($)",

            min_value=50.0, max_value=50000.0, step=50.0,

            value=float(trading.get("max_trade_usd", 1000.0)),

            help="Absolute ceiling per new position from dynamic sizing.",

        )

        demo_amt = t2.number_input(

            "Demo trade amount ($)",

            min_value=10.0, max_value=50000.0, step=10.0,

            value=float(trading.get("demo_trade_amount", 1000.0)),

            help="Fallback / config cap when account snapshot is unavailable; "

                 "also shown in the sidebar.",

        )

        t3, t4 = st.columns(2)

        min_trade = t3.number_input(

            "Min trade size ($)",

            min_value=10.0, max_value=5000.0, step=10.0,

            value=float(trading.get("min_trade_usd", 200.0)),

            help="Below this, skip the trade (dust / eToro minimums).",

        )

        risk_pct = t4.number_input(

            "Risk % per trade",

            min_value=0.1, max_value=5.0, step=0.05,

            value=float(trading.get("risk_pct_per_trade", 0.75)),

            help="% of equity risked at the stop for one new position.",

        )

        t5, t6 = st.columns(2)

        max_pos_pct = t5.number_input(

            "Max position % of equity",

            min_value=1.0, max_value=25.0, step=0.5,

            value=float(trading.get("max_position_pct", 6.0)),

            help="Hard cap: one position cannot exceed this % of account equity.",

        )

        cash_reserve = t6.number_input(

            "Cash reserve %",

            min_value=0.0, max_value=50.0, step=1.0,

            value=float(trading.get("cash_reserve_pct", 10.0)),

            help="% of free cash kept untouched before sizing new trades.",

        )

        reserve_hard = st.number_input(

            "Reserve hard floor %",

            min_value=0.0, max_value=50.0, step=1.0,

            value=float(trading.get("reserve_hard_pct", 5.0)),

            help="Cash-freeing may relax the reserve down to this % for strong signals.",

        )

        st.caption("**Per-bot strategy** is on the Bots tab.")

        if st.form_submit_button("Save trading & sizing", type="primary"):

            user_settings.save(trading={

                "max_trade_usd": float(max_trade),

                "demo_trade_amount": float(demo_amt),

                "min_trade_usd": float(min_trade),

                "risk_pct_per_trade": float(risk_pct),

                "max_position_pct": float(max_pos_pct),

                "cash_reserve_pct": float(cash_reserve),

                "reserve_hard_pct": float(reserve_hard),

            })

            _sync_session(demo_amount=float(demo_amt))

            st.success("Trading & sizing saved — applies on the next entry.")

            st.rerun()



    st.markdown("---")



    # ── Learning / journal guard ──────────────────────────────────────────────

    st.markdown("#### Learning & entry guidance")

    learning = cfg["learning"]

    with st.form("settings_learning", border=True):

        guidance_on = st.toggle(

            "Entry guidance enabled",

            value=bool(learning.get("entry_guidance_enabled", True)),

            help="When off, the trade journal never vetoes new entries "

                 "(\"Historically weak setup\" blocks are disabled).",

        )

        l1, l2, l3 = st.columns(3)

        min_bucket = l1.number_input(

            "Min bucket trades",

            min_value=3, max_value=50, step=1,

            value=int(learning.get("min_bucket_n", 8)),

            help="Minimum closed trades in a bucket before guidance can block.",

        )

        lose_wr = l2.number_input(

            "Losing win-rate max",

            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",

            value=float(learning.get("lose_winrate_max", 0.40)),

            help="Block when win rate is at or below this (e.g. 0.40 = 40%).",

        )

        lose_pf = l3.number_input(

            "Losing profit-factor max",

            min_value=0.0, max_value=2.0, step=0.05, format="%.2f",

            value=float(learning.get("lose_profit_factor_max", 0.75)),

            help="Block when gross-wins / gross-losses is below this.",

        )

        if st.form_submit_button("Save learning settings", type="primary"):

            user_settings.save(learning={

                "entry_guidance_enabled": guidance_on,

                "min_bucket_n": int(min_bucket),

                "lose_winrate_max": float(lose_wr),

                "lose_profit_factor_max": float(lose_pf),

            })

            st.success("Learning settings saved — applies on the next entry check.")

            st.rerun()



    st.markdown("---")



    # ── Bot ranking (BLEEDING advisory) ───────────────────────────────────────

    st.markdown("#### Bot ranking (BLEEDING advisory)")

    ranking = cfg.get("ranking") or {}

    st.caption(

        "Per-bot advisory flags on the Bots tab — each bot is judged on its own "

        "closed-trade history (strategy + interval + asset). Does **not** stop trading."

    )

    with st.form("settings_ranking", border=True):

        k1, k2, k3 = st.columns(3)

        min_trades = k1.number_input(

            "Min trades before BLEEDING",

            min_value=5, max_value=50, step=1,

            value=int(ranking.get("min_trades", 13)),

            help="Closed trades on that bot before the advisory flag can appear.",

        )

        pf_flag = k2.number_input(

            "BLEEDING profit-factor threshold",

            min_value=0.1, max_value=1.5, step=0.05, format="%.2f",

            value=float(ranking.get("pf_flag", 0.75)),

            help="Flag when rolling profit factor falls below this.",

        )

        pf_recover = k3.number_input(

            "Recovery profit-factor threshold",

            min_value=0.5, max_value=2.0, step=0.05, format="%.2f",

            value=float(ranking.get("pf_recover", 1.0)),

            help="Clear the flag when profit factor recovers to this (hysteresis).",

        )

        k4, k5 = st.columns(2)

        window = k4.number_input(

            "Rolling trade window",

            min_value=10, max_value=100, step=5,

            value=int(ranking.get("window", 40)),

            help="How many recent closed trades per bot feed the profit-factor calc.",

        )

        review_min = k5.number_input(

            "Background review interval (minutes)",

            min_value=5, max_value=120, step=5,

            value=int(float(ranking.get("review_sec", 1800.0)) / 60.0),

            help="How often the advisory reviewer re-checks all bots.",

        )

        if st.form_submit_button("Save ranking settings", type="primary"):

            user_settings.save(ranking={

                "min_trades": int(min_trades),

                "pf_flag": float(pf_flag),

                "pf_recover": float(pf_recover),

                "window": int(window),

                "review_sec": float(review_min) * 60.0,

            })

            st.success("Ranking settings saved — BLEEDING badges update immediately.")

            st.rerun()



    st.markdown("---")



    # ── Behavior ──────────────────────────────────────────────────────────────

    st.markdown("#### Behavior")

    behavior = cfg["behavior"]

    with st.form("settings_behavior", border=True):

        regime_on = st.toggle(

            "Market regime filter enabled",

            value=bool(behavior.get("regime_filter_enabled", True)),

            help="When off, mean-reversion and trend bots are not suppressed by "

                 "the live regime classifier (ATR% still used for stop sizing).",

        )

        recovery_on = st.toggle(

            "Recovery exit enabled",

            value=bool(behavior.get("recovery_exit_enabled", True)),

            help="Close at breakeven when a trade has been underwater (never "

                 "meaningfully green) for a long time and P&L crosses back to ≥ $0.",

        )

        recovery_mult = st.number_input(

            "Recovery hold multiplier (× strategy avg hold)",

            min_value=1.5, max_value=5.0, step=0.5,

            value=float(behavior.get("recovery_hold_mult", 2.5)),

            help="How long underwater before arming: e.g. 2.5× means if the "

                 "strategy's avg hold is 40 min, act at ≥$0 after ~100 min red.",

            disabled=not recovery_on,

        )

        recovery_be = st.toggle(

            "Breakeven stop instead of closing (recommended)",

            value=bool(behavior.get("recovery_breakeven_stop", True)),

            help="At recovery to ≥$0: raise the stop to the entry price and keep "

                 "the position open — a breakout keeps running into the take-profit "

                 "/ ATR trail, a roll-over closes at ~no loss ('breakeven_stop' in "

                 "the journal).  Off = close immediately at ≥$0 (legacy).",

            disabled=not recovery_on,

        )

        spread_rec_mult = st.number_input(

            "Spread-recovery zone (× entry spread cost)",

            min_value=0.5, max_value=6.0, step=0.5,

            value=float(behavior.get("spread_recovery_mult", 2.0)),

            help="A loss within this × the spread cost is 'just the spread' — the "

                 "LLM never closes there, and 'meaningfully green' for recovery "

                 "arming is measured against it.",

        )

        llm_cut_conf = st.number_input(

            "LLM loss-cut minimum confidence (%)",

            min_value=50, max_value=95, step=5,

            value=int(behavior.get("llm_loss_cut_min_conf", 70)),

            help="An LLM CLOSE on a real losing position only executes at or above "

                 "this confidence; below it the trade rides to its mechanical stop.",

        )

        if st.form_submit_button("Save behavior", type="primary"):

            user_settings.save(behavior={

                "regime_filter_enabled": regime_on,

                "recovery_exit_enabled": recovery_on,

                "recovery_hold_mult": float(recovery_mult),

                "recovery_breakeven_stop": bool(recovery_be),

                "spread_recovery_mult": float(spread_rec_mult),

                "llm_loss_cut_min_conf": int(llm_cut_conf),

            })

            st.success("Behavior settings saved — applies on the next signal.")

            st.rerun()



    st.markdown("---")



    # ── Display ───────────────────────────────────────────────────────────────

    st.markdown("#### Display")

    display = cfg["display"]

    _tz_opts = list(timez.COMMON_ZONES)

    _cur_tz = str(display.get("display_tz", "UTC") or "UTC")

    if _cur_tz not in _tz_opts:

        _tz_opts.insert(0, _cur_tz)

    with st.form("settings_display", border=True):

        picked_tz = st.selectbox(

            "Display timezone",

            _tz_opts,

            index=_tz_opts.index(_cur_tz),

            format_func=lambda z: f"🕒 {z}",

            help="Every date/time in the app is shown in this zone. "

                 "Stored times stay in UTC. Also changeable on the Trading tab.",

        )

        if st.form_submit_button("Save display", type="primary"):

            user_settings.save(display={"display_tz": picked_tz})

            _sync_session(display_tz=picked_tz)

            st.success("Display timezone saved.")

            st.rerun()



    st.markdown("---")



    # ── Per-bot overrides ───────────────────────────────────────────────────────

    st.markdown("#### Per-bot exit overrides")

    st.caption("Leave blank to use the strategy-class defaults above.")

    specs = instrument_config.load_specs()

    rows = []

    overrides = cfg.get("bot_overrides") or {}

    for spec in sorted(specs, key=lambda s: s.key):

        ov = overrides.get(spec.key, {})

        prof = exit_profiles.profile(spec.strategy, spec.label)

        rows.append({

            "bot": spec.key,

            "strategy": display_names().get(spec.strategy, spec.strategy),

            "class": prof.kind,

            "trailing %": ov.get("trailing_stop_pct"),

            "take-profit %": ov.get("take_profit_pct"),

        })



    df = pd.DataFrame(rows)

    edited = st.data_editor(

        df,

        column_config={

            "bot": st.column_config.TextColumn(
                "Bot", disabled=True,
                help="Bot key from instruments.toml — one row per configured bot.",
            ),

            "strategy": st.column_config.TextColumn(
                "Strategy", disabled=True,
                help="Strategy assigned to this bot (determines default exit class).",
            ),

            "class": st.column_config.TextColumn(
                "Class", disabled=True,
                help="Exit behaviour class: trend, mean_revert, arb, or llm — maps to "
                     "the Exit profiles section above.",
            ),

            "trailing %": st.column_config.NumberColumn(

                "Trailing %", min_value=0.0, max_value=20.0, step=0.1, format="%.1f",

                help="Override trailing stop % for this bot only.  Blank = use the "
                     "strategy-class default from Exit profiles above.",

            ),

            "take-profit %": st.column_config.NumberColumn(

                "Take-profit %", min_value=0.0, max_value=20.0, step=0.1, format="%.1f",

                help="Override hard take-profit % for this bot only.  Blank = use "
                     "the strategy-class default.  0 disables take-profit for that bot.",

            ),

        },

        hide_index=True,

        use_container_width=True,

        key="settings_bot_overrides_editor",

    )



    if st.button("Save per-bot overrides", type="primary"):

        bot_out: dict[str, dict] = {}

        for _, row in edited.iterrows():

            entry: dict[str, float] = {}

            if pd.notna(row["trailing %"]):

                entry["trailing_stop_pct"] = float(row["trailing %"])

            if pd.notna(row["take-profit %"]):

                entry["take_profit_pct"] = float(row["take-profit %"])

            if entry:

                bot_out[str(row["bot"])] = entry

        user_settings.save(bot_overrides=bot_out)

        trading_engine.refresh_all_exit_params()

        st.success("Per-bot overrides saved.")

        st.rerun()



    with st.expander("How exits are applied (reference)"):

        st.markdown(

            """

**Every tick (~1s):** stop-loss → take-profit → trailing stop (at most one fires).



**Every candle close:** strategy reversal (rule bots) or LLM exit — reversal only

closes **in profit** above spread costs.



**History** shows which method closed each trade in the **Close method** column.

            """

        )


