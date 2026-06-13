"""Headless fleet optimization: best exit params per plan, ranked by P&L."""
import os

import backtester
from etoro_client import get_shared_client

IIDS = {
    "XRP  (XRP)": 100003,
    "Bitcoin  (BTC)": 100000,
    "Tesla Motors, Inc.  (TSLA)": 1111,
    "NVIDIA Corporation  (NVDA)": 1137,
    "Gold (Non Expiry)  (GOLD)": 18,
    "Oil (Non Expiry)  (OIL)": 17,
    "Exxon-Mobil  (XOM)": 1036,
    "JPMorgan Chase & Co  (JPM)": 1023,
    "Amazon.com Inc  (AMZN)": 1005,
}


def main():
    c = get_shared_client(os.environ["ETORO_API_KEY"], os.environ["ETORO_USER_KEY"])
    # Sweep EVERY sync registry strategy × asset × interval — not just plans
    # that already have a TOML bot — so new strategies (daviddtech, rsi2,
    # ttm_squeeze, turtle_soup, …) are ranked alongside the incumbents.
    from types import SimpleNamespace
    import strategies as strats_mod
    seen, plans = set(), []
    for label in IIDS:
        for s in strats_mod.all_strategies():
            if s.is_async:
                continue                      # llm — excluded by design
            for secs in (600, 900, 1800, 3600):
                k = (s.key, label, secs)
                if k in seen:
                    continue
                seen.add(k)
                plans.append(SimpleNamespace(
                    strategy=s.key, label=label,
                    interval_secs=secs, candle_count=300,
                ))

    # ── Resumable: each finished plan is checkpointed, so a container restart
    #    mid-sweep costs minutes, not the whole run.  The checkpoint is deleted
    #    on successful completion.
    import json as _json
    PARTIAL = "/app/data/fleet_sweep_partial.json"
    try:
        with open(PARTIAL, encoding="utf-8") as _f:
            partial = _json.load(_f)
    except Exception:
        partial = {}
    if partial:
        print(f"resuming: {len(partial)} plan(s) already checkpointed", flush=True)

    def _ckpt():
        try:
            with open(PARTIAL + ".tmp", "w", encoding="utf-8") as _f:
                _json.dump(partial, _f)
            os.replace(PARTIAL + ".tmp", PARTIAL)
        except Exception:
            pass

    def _finish(pk, kind, payload):
        """Record a completed plan (kind 'row' or 'skip') and checkpoint it."""
        (rows if kind == "row" else skipped).append(payload)
        partial[pk] = (kind, list(payload))
        _ckpt()

    dfs, rows, skipped = {}, [], []
    for sp in plans:
        _pk = f"{sp.strategy}|{sp.label}|{sp.interval_secs}"
        if _pk in partial:
            kind, payload = partial[_pk]
            (rows if kind == "row" else skipped).append(tuple(payload))
            continue
        dkey = (sp.label, sp.interval_secs)
        if dkey not in dfs:
            try:
                dfs[dkey] = c.get_hist_candles(IIDS[sp.label], sp.interval_secs, 1000)
            except Exception:
                dfs[dkey] = None
        df = dfs[dkey]
        name = (sp.strategy, sp.label.split()[0], f"{sp.interval_secs // 60}m")
        if df is None or len(df) < sp.candle_count + 50:
            _finish(_pk, "skip", (*name, "no history"))
            continue
        sweep = backtester.optimize_exits(
            df, sp.strategy, sp.label, IIDS[sp.label], sp.interval_secs,
            min_is_trades=8, window_bars=sp.candle_count,
        )
        if not sweep:
            _finish(_pk, "skip", (*name, "not replayable"))
            continue
        valid = [r for r in sweep["rows"] if not r["excluded"]]
        if not valid:
            _finish(_pk, "skip", (*name, "too few signals"))
            continue
        best = max(valid, key=lambda r: (
            99.0 if r["oos"]["pf"] == float("inf") else r["oos"]["pf"], r["oos"]["pnl"]))
        res = backtester.simulate_exits(
            df, sweep["signals"], sp.strategy, sp.label, sp.interval_secs,
            stop_mult=best["stop_mult"], trail_mult=best["trail_mult"],
            tp_pct=best["tp_pct"], min_conf=int(best.get("min_conf", 0)),
            window_bars=sp.candle_count,
        )
        s = res.summary()
        oospf = 99.0 if best["oos"]["pf"] == float("inf") else best["oos"]["pf"]
        _finish(_pk, "row", (*name, best["stop_mult"], best["trail_mult"], best["tp_pct"],
                             s["n"], s["win_rate"] * 100, s["pnl"], s["max_dd"],
                             oospf, best["oos"]["n"], int(best.get("min_conf", 0))))
        print("done:", name, flush=True)

    rows.sort(key=lambda r: r[8], reverse=True)
    print()
    print("RANK  STRATEGY            ASSET    IVL    STOPx TRAILx TP%  CONF   N   WIN%   PNL$    MAXDD$  OOSPF OOSn")
    for i, r in enumerate(rows, 1):
        print(f"{i:4}  {r[0]:18} {r[1]:8} {r[2]:6} {r[3]:5.1f} {r[4]:5.1f} {r[5]:4.1f} {r[12]:5} {r[6]:4} {r[7]:5.1f} {r[8]:+8.2f} {r[9]:7.2f} {r[10]:5.2f} {r[11]:4}")
    print()
    for s in skipped:
        print("skipped:", s)

    # ── Persist in the dashboard's format so the Backtest page shows this run ──
    import json
    from datetime import datetime, timezone
    import strategies as strategies_mod
    names = strategies_mod.display_names()
    ivl_label = {60: "1 Minute", 300: "5 Minutes", 600: "10 Minutes",
                 900: "15 Minutes", 1800: "30 Minutes", 3600: "1 Hour",
                 14400: "4 Hours", 86400: "1 Day"}
    out = []
    for r in rows:
        secs = int(r[2].rstrip("m")) * 60
        out.append({
            "Strategy": names.get(r[0], r[0]), "Asset": r[1],
            "Interval": ivl_label.get(secs, r[2]), "Status": "ok",
            "Stop ×ATR": r[3], "Trail ×ATR": r[4], "TP %": r[5],
            "Min conf": r[12],
            # Exit check-in interval — default = trade interval (no-op today).
            # Becomes a swept value once the candle archive holds enough finer-TF
            # history to optimize it without surfacing noise (see _exitcheck_study).
            "Check-in": ivl_label.get(secs, r[2]),
            "Trades": r[6], "Win %": round(r[7], 1), "P&L $": r[8],
            "Max DD $": r[9], "OOS PF": r[10], "OOS n": r[11],
        })
    for s in skipped:
        out.append({"Strategy": names.get(s[0], s[0]), "Asset": s[1],
                    "Interval": s[2], "Status": s[3]})
    with open("/app/data/fleet_opt.json", "w", encoding="utf-8") as f:
        json.dump({"ts": datetime.now(tz=timezone.utc).isoformat(timespec="minutes"),
                   "rows": out}, f)
    # Run completed cleanly — drop the resume checkpoint so the next launch
    # starts fresh rather than replaying this run.
    try:
        os.remove(PARTIAL)
    except OSError:
        pass
    print("saved -> /app/data/fleet_opt.json")


main()
