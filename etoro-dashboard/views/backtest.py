"""Backtest section (Strategies page) — replay ONE strategy over eToro history
with the live exit config, and SHOW every entry/exit on the price chart."""
from __future__ import annotations

import os
from datetime import date, timedelta

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

C_UP, C_DOWN, C_WIN, C_LOSS = "#3dba9c", "#e36b6b", "#3dba9c", "#e36b6b"


def _clip_candles_to_range(
    df: pd.DataFrame,
    from_d: date,
    to_d: date,
) -> tuple[pd.DataFrame, str | None]:
    """Keep only candles whose timestamps fall in [from_d, to_d] (inclusive days).

    Returns (clipped_df, error_message).  error_message is set when the clip
    yields an empty frame (includes the fetched span for debugging)."""
    if df is None or df.empty:
        return df, "no history"
    _t = pd.to_datetime(df["time"])
    _lo = pd.Timestamp(from_d)
    _hi = pd.Timestamp(to_d) + pd.Timedelta(days=1)
    _series_tz = getattr(_t.dt, "tz", None)
    if _series_tz is not None:
        _lo = _lo.tz_localize(_series_tz)
        _hi = _hi.tz_localize(_series_tz)
    clipped = df[(_t >= _lo) & (_t < _hi)].reset_index(drop=True)
    if clipped.empty:
        fetched_span = f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}"
        return clipped, (
            f"no candles in {from_d} → {to_d} (fetched {fetched_span})"
        )
    return clipped, None


def _live_window_bars(label: str, interval_label: str) -> int:
    """The candle_count the LIVE bot for this (asset, interval) actually sees —
    signals in the replay are computed on a rolling window of exactly this size."""
    try:
        import instrument_config
        for s in instrument_config.load_specs():
            if s.label == label and s.interval == interval_label:
                return int(s.candle_count)
    except Exception:
        pass
    return 300 if _INTERVALS.get(interval_label, 900) < 900 else 200


def _rule_strategy_keys() -> list[str]:
    out = []
    for key in strategies_mod.display_names():
        try:
            if not getattr(strategies_mod.get(key), "is_async", False):
                out.append(key)
        except Exception:
            continue
    return out


def _chart_with_trades(df: pd.DataFrame, trades: list) -> go.Figure:
    """Candlestick chart with colour-vision-safe trade markers.

    Shape carries the meaning (▲ long / ▼ short / ★ winning close / ✕ losing
    close); colours are Okabe-Ito with dark outlines — shared by the main
    backtest chart and the optimizer's best-parameters chart."""
    fig = go.Figure(go.Candlestick(
        x=df["time"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color=C_UP, decreasing_line_color=C_DOWN,
        showlegend=False, name="price",
    ))
    CB_LONG, CB_SHORT, CB_WIN, CB_LOSS = "#E69F00", "#56B4E9", "#F0E442", "#CC79A7"
    longs  = [t for t in trades if t.direction == "LONG"]
    shorts = [t for t in trades if t.direction == "SHORT"]
    wins   = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]
    if longs:
        fig.add_trace(go.Scatter(
            x=[t.entry_time for t in longs], y=[t.entry_price for t in longs],
            mode="markers", name="▲ LONG entry",
            marker=dict(symbol="triangle-up", size=14, color=CB_LONG,
                        line=dict(width=1.5, color="#0e1117")),
            hovertemplate="LONG entry @ %{y:.5f}<br>%{x}<extra></extra>",
        ))
    if shorts:
        fig.add_trace(go.Scatter(
            x=[t.entry_time for t in shorts], y=[t.entry_price for t in shorts],
            mode="markers", name="▼ SHORT entry",
            marker=dict(symbol="triangle-down", size=14, color=CB_SHORT,
                        line=dict(width=1.5, color="#0e1117")),
            hovertemplate="SHORT entry @ %{y:.5f}<br>%{x}<extra></extra>",
        ))
    if wins:
        fig.add_trace(go.Scatter(
            x=[t.exit_time for t in wins], y=[t.exit_price for t in wins],
            mode="markers", name="★ close (profit)",
            marker=dict(symbol="star", size=13, color=CB_WIN,
                        line=dict(width=1, color="#0e1117")),
            customdata=[[t.reason, t.pnl_dollars, t.pnl_pct, t.direction] for t in wins],
            hovertemplate="close %{customdata[3]} (%{customdata[0]})<br>"
                          "@ %{y:.5f} → $%{customdata[1]:+.2f} (%{customdata[2]:+.2f}%)"
                          "<extra></extra>",
        ))
    if losses:
        fig.add_trace(go.Scatter(
            x=[t.exit_time for t in losses], y=[t.exit_price for t in losses],
            mode="markers", name="✕ close (loss)",
            marker=dict(symbol="x", size=11, color=CB_LOSS,
                        line=dict(width=1, color="#0e1117")),
            customdata=[[t.reason, t.pnl_dollars, t.pnl_pct, t.direction] for t in losses],
            hovertemplate="close %{customdata[3]} (%{customdata[0]})<br>"
                          "@ %{y:.5f} → $%{customdata[1]:+.2f} (%{customdata[2]:+.2f}%)"
                          "<extra></extra>",
        ))
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


_FLEET_PATH = "/app/data/fleet_opt.json"


def _load_fleet_saved() -> dict | None:
    try:
        import json
        with open(_FLEET_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("rows") else None
    except Exception:
        return None


def _save_fleet(rows: list, *, meta: dict | None = None) -> None:
    try:
        import json
        from datetime import datetime, timezone
        payload: dict = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="minutes"),
            "rows": rows,
        }
        if meta:
            payload["meta"] = meta
        with open(_FLEET_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:
        pass


def _fleet_range_label(meta: dict | None) -> str:
    if not meta:
        return "full fetched window (~1000 candles per plan)"
    if meta.get("use_range"):
        return f"{meta.get('from')} → {meta.get('to')} ({meta.get('candles', 1000)} candles fetched)"
    return f"full history ({meta.get('candles', 1000)} candles per plan)"


_MARKER_CAPTION = (
    "Shapes carry the meaning: **▲ = long entry** (orange) · **▼ = short entry** "
    "(sky blue) · **★ = profitable close** (yellow) · **✕ = losing close** "
    "(magenta). Hover any marker for the exit reason and exact P&L."
)

_BT_TABLE_CSS = """
<style>
.bt-table-wrap{max-height:640px;overflow:auto;margin:0.2rem 0 0.5rem;}
table.bt-table{width:100%;border-collapse:collapse;font-size:0.86rem;}
table.bt-table th{text-align:right;padding:6px 10px;border-bottom:1px solid rgba(255,255,255,0.18);
  color:#9aa4b2;font-weight:600;white-space:nowrap;position:sticky;top:0;
  background:rgba(14,17,23,0.95);}
table.bt-table th:first-child,table.bt-table td:first-child{text-align:left;}
table.bt-table td{text-align:right;padding:5px 10px;border-bottom:1px solid rgba(255,255,255,0.06);
  white-space:nowrap;}
table.bt-table tr:hover td{background:rgba(255,255,255,0.03);}
</style>
"""


def _fmt_table_cell(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        if abs(v) >= 100 or abs(v) < 0.01:
            return f"{v:,.2f}"
        return f"{v:.2f}"
    return str(v)


def _filter_fleet_rows(
    rows: list[dict],
    *,
    strategies: list[str] | None,
    assets: list[str] | None,
    intervals: list[str] | None,
    min_pnl: float,
    min_oos_pf: float,
    min_trades: int,
    min_oos_n: int,
) -> list[dict]:
    """Apply UI filters to fleet optimization result rows (Status == ok)."""
    out = list(rows)
    if strategies:
        strat_set = set(strategies)
        out = [r for r in out if r.get("Strategy") in strat_set]
    if assets:
        asset_set = set(assets)
        out = [r for r in out if r.get("Asset") in asset_set]
    if intervals:
        iv_set = set(intervals)
        out = [r for r in out if r.get("Interval") in iv_set]
    if min_pnl > 0:
        out = [r for r in out if float(r.get("P&L $") or 0) >= min_pnl]
    if min_oos_pf > 0:
        out = [r for r in out if float(r.get("OOS PF") or 0) >= min_oos_pf]
    if min_trades > 0:
        out = [r for r in out if int(r.get("Trades") or 0) >= min_trades]
    if min_oos_n > 0:
        out = [r for r in out if int(r.get("OOS n") or 0) >= min_oos_n]
    return out


def _render_df_table(df: pd.DataFrame, *, empty_msg: str = "No data yet.") -> None:
    """Static HTML table — st.dataframe (glide-data-grid) paints blank after a
    progress bar in the same expander on this Streamlit build."""
    if df is None or getattr(df, "empty", True):
        st.caption(empty_msg)
        return
    st.markdown(_BT_TABLE_CSS, unsafe_allow_html=True)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{_fmt_table_cell(v)}</td>" for v in row)
        body_rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        "<div class='bt-table-wrap'><table class='bt-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_fleet_optimization(
    instruments: dict,
    names: dict,
    amount: float,
    spread_pct: float,
) -> None:
    """Fleet exit sweep — always rendered below single-strategy backtest charts."""
    with st.expander("🏁 Fleet optimization — best exit parameters for EVERY plan, ranked by P&L"):
        st.caption(
            "Sweeps the 196-combo exit grid for every configured (strategy × asset "
            "× interval) plan at its live window size, replays each plan's best "
            "out-of-sample combo over the chosen window, and ranks the results by "
            "P&L.  Optionally clip to a date range (same as single backtest).  "
            "Takes a few minutes — one full signal replay per plan."
        )
        fl_assets = st.multiselect(
            "Assets", sorted(instruments.keys()), default=sorted(instruments.keys()),
            key="bt_fleet_assets",
        )
        fc1, fc2, fc3, fc4 = st.columns([1.4, 1, 1, 1])
        with fc1:
            fl_use_range = st.checkbox(
                "Limit time range", value=False, key="bt_fleet_use_range",
                help="Clip each plan's candle history to these dates after fetch. "
                     "eToro returns at most ~1000 candles per interval — widen "
                     "Candles if your range starts before the fetched history.",
            )
        with fc2:
            fl_from = st.date_input(
                "From", value=date.today() - timedelta(days=7),
                key="bt_fleet_from", disabled=not fl_use_range,
            )
        with fc3:
            fl_to = st.date_input(
                "To", value=date.today(),
                key="bt_fleet_to", disabled=not fl_use_range,
            )
        with fc4:
            fl_candles = st.number_input(
                "Candles", min_value=150, max_value=1000, value=1000, step=50,
                key="bt_fleet_candles",
                help="History depth fetched from eToro before optional date clip.",
            )
        if st.button("Optimize the whole fleet", key="bt_fleet_run"):
            st.session_state["_bt_fleet_pending"] = True
            st.rerun()

        if st.session_state.pop("_bt_fleet_pending", False):
            import instrument_config
            _fl_use = bool(st.session_state.get("bt_fleet_use_range", False))
            _fl_from = st.session_state.get("bt_fleet_from", date.today() - timedelta(days=7))
            _fl_to = st.session_state.get("bt_fleet_to", date.today())
            _fl_candles = int(st.session_state.get("bt_fleet_candles", 1000))
            _fleet_meta = {
                "use_range": _fl_use,
                "from": _fl_from.isoformat() if _fl_use else None,
                "to": _fl_to.isoformat() if _fl_use else None,
                "candles": _fl_candles,
            }
            # Sweep EVERY sync strategy in the registry — not just those that
            # already have a TOML bot — so new strategies show their potential
            # before earning fleet slots.  LLM excluded by design (lookahead,
            # nondeterminism, cost).  Intervals = the fleet's proven set.
            from types import SimpleNamespace
            import strategies as _strats_mod
            _sweep_secs = (600, 900, 1800, 3600)
            _strat_keys = [s.key for s in _strats_mod.all_strategies() if not s.is_async]
            seen, plans = set(), []
            for _label in sorted(fl_assets):
                for _skey in _strat_keys:
                    for _secs in _sweep_secs:
                        k = (_skey, _label, _secs)
                        if k in seen:
                            continue
                        seen.add(k)
                        plans.append(SimpleNamespace(
                            strategy=_skey, label=_label,
                            interval_secs=_secs, candle_count=300,
                        ))
            client = get_shared_client(
                os.environ.get("ETORO_API_KEY", ""), os.environ.get("ETORO_USER_KEY", ""),
            )
            sec_to_label = {v: k for k, v in _INTERVALS.items()}
            dfs_cache: dict = {}
            fleet_rows = []
            fprog = st.progress(0.0, text="Starting fleet sweep…")
            for p_i, sp in enumerate(plans):
                ivl = sec_to_label.get(sp.interval_secs, f"{sp.interval_secs // 60}m")
                fprog.progress(p_i / max(1, len(plans)),
                               text=f"{sp.strategy} · {sp.label.split()[0]} · {ivl} "
                                    f"({p_i + 1}/{len(plans)})")
                dkey = (sp.label, sp.interval_secs)
                if dkey not in dfs_cache:
                    try:
                        raw = client.get_hist_candles(
                            instruments[sp.label], sp.interval_secs, _fl_candles,
                        )
                        # Deepen with the on-disk candle archive — backtests
                        # silently get longer as the archive grows past
                        # eToro's ~1000-candle fetch ceiling.
                        try:
                            import candle_archive
                            raw = candle_archive.load_merged(
                                instruments[sp.label], sp.interval_secs, raw,
                            )
                        except Exception:
                            pass
                        if _fl_use and raw is not None and not raw.empty:
                            raw, clip_err = _clip_candles_to_range(raw, _fl_from, _fl_to)
                            dfs_cache[dkey] = (raw, clip_err)
                        else:
                            dfs_cache[dkey] = (raw, None)
                    except Exception:
                        dfs_cache[dkey] = (None, "fetch failed")
                fdf, clip_err = dfs_cache[dkey]
                base = {"Strategy": names.get(sp.strategy, sp.strategy),
                        "Asset": sp.label.split()[0], "Interval": ivl}
                if clip_err:
                    fleet_rows.append({**base, "Status": clip_err})
                    continue
                if fdf is None or fdf.empty:
                    fleet_rows.append({**base, "Status": "no history"})
                    continue
                if len(fdf) < sp.candle_count + 50:
                    fleet_rows.append({
                        **base,
                        "Status": (
                            f"too few candles ({len(fdf)}<{sp.candle_count + 50})"
                        ),
                    })
                    continue
                sweep = backtester.optimize_exits(
                    fdf, sp.strategy, sp.label, instruments[sp.label], sp.interval_secs,
                    amount=float(amount), spread_pct=float(spread_pct),
                    min_is_trades=8, window_bars=sp.candle_count,
                )
                if not sweep:
                    fleet_rows.append({**base, "Status": "not replayable"})
                    continue
                fvalid = [r for r in sweep["rows"] if not r["excluded"]]
                if not fvalid:
                    fleet_rows.append({**base, "Status": "too few signals"})
                    continue
                fbest = max(fvalid, key=lambda r: (
                    99.0 if r["oos"]["pf"] == float("inf") else r["oos"]["pf"],
                    r["oos"]["pnl"]))
                fres = backtester.simulate_exits(
                    fdf, sweep["signals"], sp.strategy, sp.label, sp.interval_secs,
                    stop_mult=fbest["stop_mult"], trail_mult=fbest["trail_mult"],
                    tp_pct=fbest["tp_pct"], min_conf=int(fbest.get("min_conf", 0)),
                    amount=float(amount), spread_pct=float(spread_pct),
                    window_bars=sp.candle_count,
                )
                fs = fres.summary()
                fleet_rows.append({
                    **base, "Status": "ok",
                    "Stop ×ATR": fbest["stop_mult"], "Trail ×ATR": fbest["trail_mult"],
                    "TP %": fbest["tp_pct"], "Min conf": int(fbest.get("min_conf", 0)),
                    # Exit check-in interval.  Default = the trade interval (= today's
                    # behaviour: signal-reversal exit re-checked once per candle).
                    # The interactive sweep doesn't optimize it (finer-TF history is
                    # data-starved; see _exitcheck_study) — it's swept in the deep
                    # CLI run once the candle archive is deep enough.
                    "Check-in": ivl,
                    "Trades": fs["n"], "Win %": round(fs["win_rate"] * 100, 1),
                    "P&L $": fs["pnl"], "Max DD $": fs["max_dd"],
                    "OOS PF": 99.0 if fbest["oos"]["pf"] == float("inf") else fbest["oos"]["pf"],
                    "OOS n": fbest["oos"]["n"],
                })
            fprog.progress(1.0, text="Done")
            st.session_state["bt_fleet"] = fleet_rows
            st.session_state["bt_fleet_meta"] = _fleet_meta
            _save_fleet(fleet_rows, meta=_fleet_meta)
            st.rerun()

        fleet_rows = st.session_state.get("bt_fleet")
        fleet_meta = st.session_state.get("bt_fleet_meta")
        _saved_note = ""
        if not fleet_rows:
            _saved = _load_fleet_saved()
            if _saved:
                fleet_rows = _saved["rows"]
                fleet_meta = _saved.get("meta")
                _saved_note = f" (saved run from {_saved.get('ts', '?')} UTC)"
        if fleet_rows:
            st.caption(
                f"Replay window: **{_fleet_range_label(fleet_meta)}** · "
                f"${float(amount):,.0f}/trade · spread {float(spread_pct):.2f}%"
            )
            try:
                import fleet_scheduler
                _wf = fleet_scheduler.last_report()
                if _wf:
                    st.caption(
                        f"🔁 Walk-forward {_wf.get('ts', '?')}: "
                        f"{len(_wf.get('applied', []))} param set(s) applied · "
                        f"{len(_wf.get('held_unstable', []))} held (unstable across windows) · "
                        f"{len(_wf.get('skipped', []))} without a qualified row"
                    )
            except Exception:
                pass
            if _saved_note:
                st.caption(f"Showing the last saved fleet run{_saved_note} — "
                           "press the button above for a fresh one.")
            ok_rows = [r for r in fleet_rows if r.get("Status") == "ok"]
            skipped = [r for r in fleet_rows if r.get("Status") != "ok"]
            if ok_rows:
                all_strats = sorted({r["Strategy"] for r in ok_rows})
                all_assets = sorted({r["Asset"] for r in ok_rows})
                all_ivls = sorted(
                    {r["Interval"] for r in ok_rows},
                    key=lambda x: _INTERVALS.get(x, 999999),
                )
                st.markdown("**Filter results**")
                ff1, ff2, ff3 = st.columns(3)
                with ff1:
                    fl_strats = st.multiselect(
                        "Strategy", all_strats, default=[], key="bt_fleet_f_strat",
                        placeholder="All strategies",
                    )
                with ff2:
                    fl_assets_tbl = st.multiselect(
                        "Asset", all_assets, default=[], key="bt_fleet_f_asset",
                        placeholder="All assets",
                    )
                with ff3:
                    fl_ivls = st.multiselect(
                        "Interval", all_ivls, default=[], key="bt_fleet_f_ivl",
                        placeholder="All intervals",
                    )
                ff4, ff5, ff6, ff7 = st.columns(4)
                with ff4:
                    fl_min_pnl = st.number_input(
                        "Min P&L $", value=0.0, step=50.0, key="bt_fleet_f_pnl",
                    )
                with ff5:
                    fl_min_pf = st.number_input(
                        "Min OOS PF", value=0.0, min_value=0.0, step=0.1,
                        key="bt_fleet_f_pf",
                        help="99 in the table means infinite PF.",
                    )
                with ff6:
                    fl_min_trades = st.number_input(
                        "Min trades", value=0, min_value=0, step=1,
                        key="bt_fleet_f_trades",
                    )
                with ff7:
                    fl_min_oos_n = st.number_input(
                        "Min OOS n", value=0, min_value=0, step=1,
                        key="bt_fleet_f_oosn",
                        help="Out-of-sample trade count for the chosen combo.",
                    )

                filtered = _filter_fleet_rows(
                    ok_rows,
                    strategies=fl_strats or None,
                    assets=fl_assets_tbl or None,
                    intervals=fl_ivls or None,
                    min_pnl=float(fl_min_pnl),
                    min_oos_pf=float(fl_min_pf),
                    min_trades=int(fl_min_trades),
                    min_oos_n=int(fl_min_oos_n),
                )
                if filtered:
                    ftable = (pd.DataFrame(filtered)
                              .drop(columns=["Status"])
                              .sort_values("P&L $", ascending=False)
                              .reset_index(drop=True))
                    # Backfill Check-in for rows saved before the column existed:
                    # an absent check-in means it equalled the trade interval.
                    if "Check-in" in ftable.columns and "Interval" in ftable.columns:
                        ftable["Check-in"] = ftable["Check-in"].fillna(ftable["Interval"])
                    elif "Interval" in ftable.columns:
                        ftable["Check-in"] = ftable["Interval"]
                    ftable.insert(0, "Rank", range(1, len(ftable) + 1))
                    st.caption(
                        f"Showing **{len(filtered)}** of **{len(ok_rows)}** "
                        f"successful plan(s) · ranked by P&L $"
                    )
                    _render_df_table(ftable)
                else:
                    st.warning(
                        f"No plans match the current filters "
                        f"({len(ok_rows)} successful plan(s) in the full run)."
                    )
                st.caption(
                    "⚠️ Each row is that plan's **luckiest of 196 combos** (winner's "
                    "curse) — trust rows with healthy **OOS n**, and treat 99 = ∞ PF "
                    "with suspicion.  P&L / win rate are the best combo replayed over "
                    "the full ~1000-candle window at your $/trade and spread inputs."
                )
                # ── Apply learnings — same stability-gated path the weekly
                #    walk-forward uses: new exits only for ON bots whose row
                #    passes the OOS gate AND whose params didn't jump across
                #    the grid (regime-chasing protection).
                if st.button(
                    "✅ Apply learnings to bots (stability-gated)",
                    key="bt_fleet_apply",
                    help="Writes each ON bot's OOS-best stop/trail/TP from this "
                         "saved run into its per-bot Settings overrides — but only "
                         "when the row passes the OOS gate (PF ≥ 1, n ≥ 5) and the "
                         "new params are within one grid step of the current ones. "
                         "Unstable jumps are HELD for your judgment. Never "
                         "enables or disables bots.",
                ):
                    try:
                        import fleet_scheduler
                        _rep = fleet_scheduler.apply_with_stability_gate()
                        _no_row = sum(1 for s in _rep["skipped"] if "no qualified" in s[1])
                        _weak = len(_rep["skipped"]) - _no_row
                        st.success(
                            f"Applied {len(_rep['applied'])} bot(s) · held "
                            f"{len(_rep['held_unstable'])} (params jumped — kept old) · "
                            f"skipped {len(_rep['skipped'])} "
                            f"({_no_row} not in this run — e.g. LLM or instruments "
                            f"the run didn't sweep; {_weak} with weak OOS).  "
                            "Skipped bots KEEP their existing exits."
                        )
                        if _rep["applied"]:
                            st.caption("Applied: " + ", ".join(
                                f"`{a[0]}` {a[1]}/{a[2]}/{a[3]}" for a in _rep["applied"]))
                        if _rep["held_unstable"]:
                            st.caption("Held (unstable): " + ", ".join(
                                f"`{h[0]}`" for h in _rep["held_unstable"]))
                    except Exception as _exc:
                        st.error(f"Apply failed: {_exc}")
            if skipped:
                sk_show = skipped
                if ok_rows:
                    sk_show = _filter_fleet_rows(
                        skipped,
                        strategies=st.session_state.get("bt_fleet_f_strat") or None,
                        assets=st.session_state.get("bt_fleet_f_asset") or None,
                        intervals=st.session_state.get("bt_fleet_f_ivl") or None,
                        min_pnl=0.0, min_oos_pf=0.0,
                        min_trades=0, min_oos_n=0,
                    )
                if sk_show:
                    st.caption("Skipped: " + " · ".join(
                        f"{r['Strategy']}/{r['Asset']}/{r['Interval']} ({r['Status']})"
                        for r in sk_show))


def render() -> None:
    st.subheader("Backtest this strategy")
    st.caption(
        "Replays the strategy selected in the **Strategy Reference above** over "
        "real eToro history, using the live signal code, the live regime filter, "
        "and your current Settings exits (2×ATR stop, chandelier trail, TP)."
    )

    instruments = trading_engine.configured_instruments()
    if not instruments:
        st.info("No configured instruments yet — start the engines first.")
        return

    keys = _rule_strategy_keys()
    names = strategies_mod.display_names()
    guide_key = st.session_state.get("guide_strategy_key")
    default_idx = keys.index(guide_key) if guide_key in keys else 0
    if guide_key and guide_key not in keys:
        st.caption("ℹ️ The LLM strategy can't be replayed honestly (model knows the "
                   "future, memory leaks backwards) — pick a rule strategy below.")

    c0, c1, c2, c3, c4 = st.columns([1.8, 1.8, 1.3, 1, 1])
    strat_key = c0.selectbox(
        "Strategy", keys, index=default_idx, key="bt_strategy",
        format_func=lambda k: names.get(k, k),
    )
    label = c1.selectbox("Asset", sorted(instruments.keys()), key="bt_asset")
    interval_label = c2.selectbox("Interval", list(_INTERVALS.keys()), index=2, key="bt_interval")
    candles = c3.number_input("Candles", min_value=150, max_value=1000, value=1000, step=50, key="bt_candles",
                              help="More candles = more signals = more trustworthy stats. "
                                   "1000 is eToro's history ceiling per interval.")
    amount = c4.number_input("$/trade", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0, key="bt_amount")

    s1, s2 = st.columns([1, 2.6])
    spread_pct = s1.number_input(
        "Spread (%)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, format="%.2f",
        key="bt_spread",
        help="Round-trip cost model: half on entry, half on exit. Crypto ≈ 0.01–0.10, "
             "liquid stocks ≈ 0.02.",
    )
    with s2:
        st.write("")  # vertical alignment with the number input
        sim_risk = st.checkbox(
            "Apply exit & risk management (ATR stop · chandelier trail · take-profit · regime filter)",
            value=True, key="bt_apply_risk",
            help="ON = trade the way your bots actually trade.  OFF = the NAKED "
                 "strategy: enter on signal, exit only when the signal reverses — "
                 "no stops, no trail, no TP, no regime gate.  Run both and compare: "
                 "the difference IS what your risk layer contributes.",
        )

    f1, f2, f3 = st.columns([1.4, 1, 1])
    with f1:
        st.write("")
        use_range = st.checkbox(
            "Limit time range", value=False, key="bt_use_range",
            help="Replay only candles inside this window.  Note: eToro returns at "
                 "most the last ~1000 candles per interval, so the reachable past "
                 "depends on the Candles setting (e.g. 1000 × 15m ≈ 10 days back); "
                 "the first 60 candles of the range are warm-up for indicators.",
        )
    from_d = f2.date_input("From", value=date.today() - timedelta(days=7),
                           key="bt_from", disabled=not use_range)
    to_d = f3.date_input("To", value=date.today(), key="bt_to", disabled=not use_range)

    if st.button("Run backtest", type="primary", key="bt_run"):
        iid = instruments[label]
        secs = _INTERVALS[interval_label]
        client = get_shared_client(
            os.environ.get("ETORO_API_KEY", ""), os.environ.get("ETORO_USER_KEY", ""),
        )
        with st.spinner(f"Fetching {int(candles)} × {interval_label} candles for {label}…"):
            df = client.get_hist_candles(iid, secs, int(candles))
        if df is None or df.empty:
            st.error("No history returned — try a different interval.")
            return
        if use_range:
            df, range_err = _clip_candles_to_range(df, from_d, to_d)
            if range_err:
                st.error(
                    f"No candles inside {from_d} → {to_d}.  {range_err} — "
                    "raise Candles to reach further back."
                )
                return
        window_bars = _live_window_bars(label, interval_label)
        if len(df) < window_bars + 10:
            st.error(
                f"Only {len(df)} candles in the selected window — the live bot "
                f"computes signals on a rolling {window_bars}-candle window, so at "
                f"least {window_bars + 10} are needed.  Widen the range or raise "
                "Candles."
            )
            return
        with st.spinner(f"Replaying {names.get(strat_key, strat_key)} over {len(df)} candles…"):
            res = backtester.run_backtest(
                df, strat_key, label, iid, secs,
                amount=float(amount), spread_pct=float(spread_pct),
                apply_exits=bool(sim_risk), apply_regime_filter=bool(sim_risk),
                window_bars=window_bars,
            )
        if res is None:
            st.error("This strategy can't be replayed (async or insufficient data).")
            return
        st.session_state["bt_run_data"] = {
            "res": res, "df": df.reset_index(drop=True),
            "label": label, "interval": interval_label, "strategy": strat_key,
            "span": f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}",
            "risk_on": bool(sim_risk),
            "window_bars": window_bars,
        }
        st.rerun()

    data = st.session_state.get("bt_run_data")
    if not data:
        st.info("Pick the parameters and hit **Run backtest**.")
    else:
        res, df = data["res"], data["df"]
        s = res.summary()
        ins, oos = res.oos_split()
        _risk_tag = ("🛡️ exits & risk ON" if data.get("risk_on", True)
                     else "⚠️ NAKED strategy — exits & risk OFF (reversal-only exits)")
        st.markdown(
            f"**{names.get(data['strategy'], data['strategy'])} · {data['label']} · "
            f"{data['interval']}** — {res.n_bars} candles ({data['span']}) · {_risk_tag} · "
            f"signal window **{data.get('window_bars', 0)} candles, same as the live bot**"
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Trades", s["n"])
        m2.metric("P&L", f"${s['pnl']:+,.2f}")
        m3.metric("Win rate", f"{s['win_rate']*100:.0f}%")
        m4.metric("Profit factor", "∞" if s["pf"] == float("inf") else f"{s['pf']:.2f}")
        m5.metric("Max drawdown", f"${s['max_dd']:,.2f}")
        m6.metric("OOS PF (last 30%)", "∞" if oos["pf"] == float("inf") else f"{oos['pf']:.2f}",
                  help=f"{oos['n']} out-of-sample trades")
        if res.regime_skipped:
            st.caption(f"🧭 Regime filter suppressed {res.regime_skipped} signal(s) — "
                       "same gate the live bots apply.")

        # ── Price chart with every entry & exit ──────────────────────────────────
        st.plotly_chart(_chart_with_trades(df, res.trades),
                        use_container_width=True, key="bt_price_chart")
        st.caption(_MARKER_CAPTION)

        # ── Equity curve + breakdowns ─────────────────────────────────────────────
        if res.trades:
            eq = go.Figure(go.Scatter(x=[t.exit_time for t in res.trades], y=res.equity_curve,
                                      mode="lines+markers", line=dict(width=2)))
            eq.add_hline(y=0, line_dash="dot", line_width=1)
            eq.update_layout(height=240, margin=dict(l=10, r=10, t=26, b=10),
                             title="Cumulative P&L ($)",
                             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(eq, use_container_width=True, key="bt_equity")

            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**By exit reason**")
                br = res.by_reason()
                _render_df_table(pd.DataFrame(
                    [{"Reason": k, "n": v["n"], "P&L $": v["pnl"]} for k, v in
                     sorted(br.items(), key=lambda kv: kv[1]["pnl"])],
                ))
            with r2:
                st.markdown("**In-sample vs out-of-sample**")
                _render_df_table(pd.DataFrame([
                    {"Segment": "first 70%", "n": ins["n"], "P&L $": ins["pnl"],
                     "PF": ins["pf"] if ins["pf"] != float("inf") else 99.0},
                    {"Segment": "last 30%", "n": oos["n"], "P&L $": oos["pnl"],
                     "PF": oos["pf"] if oos["pf"] != float("inf") else 99.0},
                ]))

            with st.expander(f"All {len(res.trades)} trades"):
                _render_df_table(pd.DataFrame([{
                    "Dir": t.direction, "Entry": t.entry_time, "Exit": t.exit_time,
                    "Entry px": t.entry_price, "Exit px": t.exit_price,
                    "P&L $": t.pnl_dollars, "P&L %": t.pnl_pct,
                    "Reason": t.reason, "Conf": t.confidence,
                } for t in res.trades]))
        else:
            st.caption("No trades in this window (signals never fired, or the regime "
                       "filter suppressed them all).")

        with st.expander("How faithful is this to the live bots?"):
            st.markdown(
                "**Identical to live:** the strategy signal code itself (same module the "
                "bots call), signals computed on closed candles only **over the same "
                "rolling window length the live bot uses (its candle_count)**, the regime "
                "entry filter (honouring your Settings toggle), and the exit parameters — 2×ATR "
                "entry stop, chandelier trail multiplier and take-profit all read from "
                "your live Settings at run time, with the same priority order "
                "(stop → trail → TP → reversal).\n\n"
                "**Approximated:** fills (live: paced order at the signal-time quote; "
                "here: next bar's open ± half-spread), intrabar exits (live: tick-by-tick; "
                "here: bar extremes vs the previous bar's levels, worst-case ordering), "
                "spread (live: actual; here: your fixed % input).\n\n"
                "**Not simulated:** position sizing & cash reserve, portfolio risk caps, "
                "the journal-evidence entry veto, execution-quality gate, pacing/stale-"
                "signal guard, recovery/breakeven floor, and the LLM strategy. These "
                "mostly REMOVE marginal trades live, so the live bot typically takes a "
                "subset of the trades you see here."
            )

    # ── Fleet optimization — always available below the single backtest ───────
    _render_fleet_optimization(instruments, names, float(amount), float(spread_pct))
