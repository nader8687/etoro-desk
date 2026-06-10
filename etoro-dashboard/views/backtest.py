"""Backtest page — replay strategies over eToro history with the LIVE exit config."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backtester
import strategies as strategies_mod
import trading_engine
from etoro_client import get_shared_client

_INTERVALS = {
    "1 Minute": 60,
    "5 Minutes": 300,
    "15 Minutes": 900,
    "1 Hour": 3600,
    "4 Hours": 14400,
    "1 Day": 86400,
}


def _rule_strategy_keys() -> list[str]:
    out = []
    for key in strategies_mod.display_names():
        try:
            if not getattr(strategies_mod.get(key), "is_async", False):
                out.append(key)
        except Exception:
            continue
    return out


def render() -> None:
    st.subheader("Backtest")
    st.caption(
        "Replays eToro history through the **live strategy code and your current "
        "Settings** (2×ATR entry stops, chandelier trail, per-class take-profits). "
        "No lookahead: signals fire on closed candles and fill at the next bar's "
        "open; spread is paid on both sides."
    )

    instruments = trading_engine.configured_instruments()
    if not instruments:
        st.info("No configured instruments yet — start the engines first.")
        return

    c1, c2, c3, c4 = st.columns([2, 1.4, 1, 1])
    label = c1.selectbox("Asset", sorted(instruments.keys()), key="bt_asset")
    interval_label = c2.selectbox("Interval", list(_INTERVALS.keys()), index=2, key="bt_interval")
    candles = c3.number_input("Candles", min_value=150, max_value=1000, value=600, step=50, key="bt_candles")
    amount = c4.number_input("$/trade", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0, key="bt_amount")

    c5, c6 = st.columns([1, 3])
    spread_pct = c5.number_input(
        "Spread (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f",
        key="bt_spread",
        help="Round-trip cost model: half on entry, half on exit. Crypto ≈ 0.01–0.10, "
             "liquid stocks ≈ 0.02.",
    )
    all_keys = _rule_strategy_keys()
    picked = c6.multiselect(
        "Strategies (LLM excluded — not replayable)", all_keys, default=all_keys, key="bt_strats",
    )

    if st.button("Run backtest", type="primary", key="bt_run"):
        iid = instruments[label]
        secs = _INTERVALS[interval_label]
        client = get_shared_client(
            os.environ.get("ETORO_API_KEY", ""), os.environ.get("ETORO_USER_KEY", ""),
        )
        with st.spinner(f"Fetching {int(candles)} × {interval_label} candles for {label}…"):
            df = client.get_hist_candles(iid, secs, int(candles))
        if df is None or df.empty or len(df) < backtester.WARMUP_BARS + 10:
            st.error(
                f"Not enough history returned ({0 if df is None else len(df)} candles). "
                "Try a smaller count or a different interval."
            )
            return
        results = {}
        prog = st.progress(0.0, text="Backtesting…")
        for k_i, key in enumerate(picked):
            prog.progress(k_i / max(1, len(picked)), text=f"Backtesting {key}…")
            res = backtester.run_backtest(
                df, key, label, iid, secs,
                amount=float(amount), spread_pct=float(spread_pct),
            )
            if res is not None:
                results[key] = res
        prog.progress(1.0, text="Done")
        st.session_state["bt_results"] = results
        st.session_state["bt_meta"] = {
            "label": label, "interval": interval_label, "bars": len(df),
            "span": f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}",
        }

    results = st.session_state.get("bt_results") or {}
    meta = st.session_state.get("bt_meta") or {}
    if not results:
        st.info("Pick an asset/interval and hit **Run backtest**.")
        return

    st.markdown(
        f"**{meta.get('label','')} · {meta.get('interval','')}** — "
        f"{meta.get('bars',0)} candles ({meta.get('span','')})"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for key, res in results.items():
        s = res.summary()
        ins, oos = res.oos_split()
        rows.append({
            "Strategy": key,
            "Trades": s["n"],
            "Win %": round(s["win_rate"] * 100, 1),
            "PF": s["pf"] if s["pf"] != float("inf") else 99.0,
            "P&L $": s["pnl"],
            "Expectancy $": s["expectancy"],
            "Avg win $": s["avg_win"],
            "Avg loss $": s["avg_loss"],
            "Max DD $": s["max_dd"],
            "Hold (bars)": s["avg_hold_bars"],
            "OOS PF": oos["pf"] if oos["pf"] != float("inf") else 99.0,
            "OOS n": oos["n"],
        })
    table = pd.DataFrame(rows).sort_values("P&L $", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(
        "**OOS** = out-of-sample: last 30% of the window, untouched by the first 70%. "
        "A strategy whose PF holds up OOS is far more trustworthy than one that doesn't."
    )

    # ── Drill-down ────────────────────────────────────────────────────────────
    pick = st.selectbox("Inspect strategy", list(results.keys()), key="bt_inspect")
    res = results[pick]
    if not res.trades:
        st.caption("No trades for this strategy in the window.")
        return

    eq_x = [t.exit_time for t in res.trades]
    fig = go.Figure(go.Scatter(x=eq_x, y=res.equity_curve, mode="lines+markers",
                               line=dict(width=2), name="equity"))
    fig.add_hline(y=0, line_dash="dot", line_width=1)
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=30, b=10),
        title=f"{pick} — cumulative P&L ($)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="bt_equity")

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**By exit reason**")
        br = res.by_reason()
        st.dataframe(pd.DataFrame(
            [{"Reason": k, "n": v["n"], "P&L $": v["pnl"]} for k, v in
             sorted(br.items(), key=lambda kv: kv[1]["pnl"])],
        ), use_container_width=True, hide_index=True)
    with r2:
        ins, oos = res.oos_split()
        st.markdown("**In-sample vs out-of-sample**")
        st.dataframe(pd.DataFrame([
            {"Segment": "first 70%", "n": ins["n"], "P&L $": ins["pnl"], "PF": ins["pf"] if ins["pf"] != float("inf") else 99.0},
            {"Segment": "last 30%", "n": oos["n"], "P&L $": oos["pnl"], "PF": oos["pf"] if oos["pf"] != float("inf") else 99.0},
        ]), use_container_width=True, hide_index=True)

    with st.expander(f"All {len(res.trades)} trades"):
        st.dataframe(pd.DataFrame([{
            "Dir": t.direction, "Entry": t.entry_time, "Exit": t.exit_time,
            "Entry px": t.entry_price, "Exit px": t.exit_price,
            "P&L $": t.pnl_dollars, "P&L %": t.pnl_pct,
            "Reason": t.reason, "Conf": t.confidence,
        } for t in res.trades]), use_container_width=True, hide_index=True)

    with st.expander("Model assumptions & limitations"):
        st.markdown(
            "- Signals on closed candles → **filled at next bar open** (no lookahead).\n"
            "- Stops/trail/TP tested intrabar against the **previous** bar's levels, "
            "worst-case ordering (stop → trail → TP) when several are touched.\n"
            "- Exit params (2×ATR stop, chandelier mult, TP) come from your **live "
            "Settings** at run time.\n"
            "- Spread paid half on each side; LLM strategy and recovery/breakeven "
            "exits not simulated.\n"
            "- eToro history caps the window — treat 1m results as hours of data, "
            "not a verdict."
        )
