"""One-off study: per-plan exit optimization vs one-size-fits-all.

For every rule-strategy plan (strategy x asset x native interval, live window
sizes), sweep the 196-combo exit grid, then:
  1. each plan's OWN best combo (by capped OOS PF, min 8 in-sample trades);
  2. the GLOBAL consensus combo (best average capped OOS PF across plans);
  3. the penalty each plan pays if forced onto the consensus combo.
"""
import json

import backtester
import instrument_config
import exit_profiles
from etoro_client import get_shared_client
import os

IIDS = {"XRP  (XRP)": 100003, "Bitcoin  (BTC)": 100000}
CAP = 5.0


def pf_cap(v):
    if v == float("inf"):
        return CAP
    return min(float(v), CAP)


def main():
    c = get_shared_client(os.environ["ETORO_API_KEY"], os.environ["ETORO_USER_KEY"])
    specs = [
        s for s in instrument_config.load_specs()
        if s.strategy != "llm" and s.label in IIDS
    ]
    # de-dup identical plans (same strategy+label+interval)
    seen, plans = set(), []
    for s in specs:
        k = (s.strategy, s.label, s.interval_secs)
        if k not in seen:
            seen.add(k)
            plans.append(s)

    dfs = {}
    plan_rows = {}      # plan_name -> {combo_key: capped oos pf or None}
    plan_best = {}      # plan_name -> dict
    plan_kind = {}

    for s in plans:
        key = (s.label, s.interval_secs)
        if key not in dfs:
            dfs[key] = c.get_hist_candles(IIDS[s.label], s.interval_secs, 1000)
        df = dfs[key]
        if df is None or len(df) < s.candle_count + 50:
            continue
        name = f"{s.strategy}|{s.label.split()[0]}|{s.interval_secs//60}m"
        sweep = backtester.optimize_exits(
            df, s.strategy, s.label, IIDS[s.label], s.interval_secs,
            window_bars=s.candle_count, min_is_trades=8,
        )
        if not sweep:
            continue
        rows = sweep["rows"]
        valid = [r for r in rows if not r["excluded"]]
        plan_kind[name] = exit_profiles.profile(s.strategy).kind
        if not valid:
            plan_best[name] = None
            continue
        scores = {}
        for r in rows:
            ck = (r["stop_mult"], r["trail_mult"], r["tp_pct"])
            scores[ck] = None if r["excluded"] else pf_cap(r["oos"]["pf"])
        plan_rows[name] = scores
        best = max(valid, key=lambda r: (pf_cap(r["oos"]["pf"]), r["oos"]["pnl"]))
        plan_best[name] = {
            "stop": best["stop_mult"], "trail": best["trail_mult"], "tp": best["tp_pct"],
            "oos_pf": pf_cap(best["oos"]["pf"]), "oos_n": best["oos"]["n"],
            "is_n": best["is"]["n"],
        }
        print(f"PLAN {name:34} kind={plan_kind[name]:11} best: "
              f"stop {best['stop_mult']:.1f}x trail {best['trail_mult']:.1f}x "
              f"tp {best['tp_pct']:.1f}% -> OOS pf {pf_cap(best['oos']['pf']):.2f} "
              f"(n={best['oos']['n']})", flush=True)

    judged = {k: v for k, v in plan_best.items() if v}
    print()
    print(f"=== {len(judged)} of {len(plans)} plans had enough trades to judge ===")

    # ── Consensus combo: best average capped OOS PF across plans ─────────────
    all_combos = set()
    for scores in plan_rows.values():
        all_combos.update(scores.keys())
    combo_stats = []
    for ck in sorted(all_combos):
        vals = [scores[ck] for scores in plan_rows.values()
                if scores.get(ck) is not None]
        if len(vals) >= max(3, int(0.6 * len(plan_rows))):
            combo_stats.append((ck, sum(vals) / len(vals), len(vals)))
    combo_stats.sort(key=lambda x: x[1], reverse=True)
    print()
    print("=== TOP 5 one-size-fits-all combos (avg capped OOS PF across plans) ===")
    for ck, avg, cov in combo_stats[:5]:
        print(f"  stop {ck[0]:.1f}x trail {ck[1]:.1f}x tp {ck[2]:.1f}%  "
              f"avg pf {avg:.2f}  (judged on {cov} plans)")

    if combo_stats:
        g = combo_stats[0][0]
        print()
        print(f"=== Penalty per plan when forced onto consensus "
              f"(stop {g[0]:.1f}x trail {g[1]:.1f}x tp {g[2]:.1f}%) ===")
        penalties = []
        for name, b in sorted(judged.items()):
            gpf = plan_rows.get(name, {}).get(g)
            if gpf is None:
                continue
            pen = b["oos_pf"] - gpf
            penalties.append(pen)
            print(f"  {name:34} own {b['oos_pf']:.2f} vs consensus {gpf:.2f}  "
                  f"penalty {pen:+.2f}")
        if penalties:
            print(f"  AVG penalty: {sum(penalties)/len(penalties):+.2f} pf  "
                  f"| plans hurt >0.5 pf: {sum(1 for p in penalties if p > 0.5)}"
                  f"/{len(penalties)}")

    # ── Family clustering: do behaviour classes want different params? ───────
    print()
    print("=== Average OWN-BEST params by behaviour class ===")
    by_kind = {}
    for name, b in judged.items():
        by_kind.setdefault(plan_kind[name], []).append(b)
    for kind, bs in sorted(by_kind.items()):
        n = len(bs)
        print(f"  {kind:12} n={n}: stop {sum(b['stop'] for b in bs)/n:.2f}x  "
              f"trail {sum(b['trail'] for b in bs)/n:.2f}x  "
              f"tp {sum(b['tp'] for b in bs)/n:.2f}%")

    print()
    print(json.dumps({"judged": len(judged), "plans": len(plans)}))


main()
