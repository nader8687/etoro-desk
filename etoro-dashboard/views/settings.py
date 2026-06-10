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

            help="Pullback from peak profit before closing (0 = off).",

        )

        tp = c2.number_input(

            "Take-profit %",

            min_value=0.0, max_value=20.0, step=0.1,

            value=float(data.get("take_profit_pct", 0.0)),

            key=f"set_exit_{kind}_tp",

            help="Hard +% target on entry (0 = off).",

        )

        sl = c3.number_input(

            "Stop-loss floor %",

            min_value=0.1, max_value=20.0, step=0.1,

            value=float(data.get("stop_loss_min_pct", 2.5)),

            key=f"set_exit_{kind}_sl",

            help="Minimum stop distance as % of entry (may widen with volatility).",

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

        "Rebuild is **not** required."

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



    st.markdown("---")



    # ── Risk manager ──────────────────────────────────────────────────────────

    st.markdown("#### Portfolio risk manager")

    risk = cfg["risk"]

    with st.form("settings_risk", border=True):

        r1, r2 = st.columns(2)

        enabled = r1.toggle("Risk manager enabled", value=bool(risk.get("enabled", True)))

        max_pos = r1.number_input(

            "Max concurrent positions",

            min_value=1, max_value=50, step=1,

            value=int(risk.get("max_concurrent_positions", 12)),

        )

        max_gross = r2.number_input(

            "Max gross exposure % of equity",

            min_value=10.0, max_value=100.0, step=5.0,

            value=float(risk.get("max_gross_exposure_pct", 60.0)),

        )

        max_heat = r2.number_input(

            "Max portfolio heat % of equity",

            min_value=1.0, max_value=30.0, step=0.5,

            value=float(risk.get("max_portfolio_heat_pct", 6.0)),

        )

        r3, r4 = st.columns(2)

        cl_gross = r3.number_input(

            "Max cluster gross %",

            min_value=10.0, max_value=100.0, step=5.0,

            value=float(risk.get("max_cluster_gross_pct", 45.0)),

        )

        cl_net = r3.number_input(

            "Max cluster net %",

            min_value=5.0, max_value=50.0, step=1.0,

            value=float(risk.get("max_cluster_net_pct", 25.0)),

        )

        same_dir = r4.number_input(

            "Max same-direction per cluster",

            min_value=1, max_value=20, step=1,

            value=int(risk.get("max_same_dir_per_cluster", 6)),

        )

        per_asset = r4.number_input(

            "Max positions per asset",

            min_value=1, max_value=10, step=1,

            value=int(risk.get("max_positions_per_asset", 4)),

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

                 "strategy's avg hold is 40 min, close at ≥$0 after ~100 min red.",

            disabled=not recovery_on,

        )

        if st.form_submit_button("Save behavior", type="primary"):

            user_settings.save(behavior={

                "regime_filter_enabled": regime_on,

                "recovery_exit_enabled": recovery_on,

                "recovery_hold_mult": float(recovery_mult),

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

            "bot": st.column_config.TextColumn("Bot", disabled=True),

            "strategy": st.column_config.TextColumn("Strategy", disabled=True),

            "class": st.column_config.TextColumn("Class", disabled=True),

            "trailing %": st.column_config.NumberColumn(

                "Trailing %", min_value=0.0, max_value=20.0, step=0.1, format="%.1f",

            ),

            "take-profit %": st.column_config.NumberColumn(

                "Take-profit %", min_value=0.0, max_value=20.0, step=0.1, format="%.1f",

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


