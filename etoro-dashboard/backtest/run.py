"""
Backtest CLI.

Examples
--------
# Smoke test on synthetic data, one strategy, with walk-forward:
python -m backtest.run --strategy supertrend --synthetic --walk-forward

# Backtest every deterministic strategy on a CSV of OHLC candles:
python -m backtest.run --csv data/btc_15m.csv --all --walk-forward

# Single strategy on a CSV, custom costs:
python -m backtest.run --csv data/xrp_1m.csv --strategy rsi --spread 0.08 --fee 0.0

CSV format: columns Open,High,Low,Close (case-insensitive), optional time/Volume.
"""
from __future__ import annotations

import argparse
import json
import sys

from .engine import Backtester, BacktestConfig, walk_forward, synthetic_ohlc, load_csv, SKIP_STRATEGIES


def _fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return "  no trades"
    flag = "" if m.get("sufficient") else "  ⚠ insufficient sample"
    return (
        f"  trades={m['n']:>4}  win%={m['win_rate']*100:5.1f}  "
        f"PF={m['profit_factor']:>5}  exp$={m['expectancy_usd']:+8.3f}  "
        f"expR={m['expectancy_r']:+6.3f}  maxDD$={m['max_drawdown_usd']:>8.2f}  "
        f"Sharpe={m['sharpe']:+5.2f}  Sortino={m['sortino']:+5.2f}{flag}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EtoroDesk strategy backtester")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to an OHLC CSV")
    src.add_argument("--synthetic", action="store_true", help="Use generated data")
    ap.add_argument("--strategy", default="supertrend", help="Strategy key")
    ap.add_argument("--all", action="store_true", help="Backtest every deterministic strategy")
    ap.add_argument("--label", default="Bitcoin  (BTC)", help="Instrument label (asset class)")
    ap.add_argument("--spread", type=float, default=0.06, help="Round-trip spread %% (e.g. 0.06)")
    ap.add_argument("--fee", type=float, default=0.0, help="Commission %% per side")
    ap.add_argument("--equity", type=float, default=10000.0, help="Starting equity")
    ap.add_argument("--conf-min", type=int, default=0, help="Ignore signals below this confidence")
    ap.add_argument("--walk-forward", action="store_true", help="Run walk-forward folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--synthetic-trend", type=float, default=0.0, help="Per-bar drift for synthetic data")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args(argv)

    if args.synthetic:
        df = synthetic_ohlc(n=3000, trend=args.synthetic_trend)
        src_desc = f"synthetic n={len(df)} trend={args.synthetic_trend}"
    else:
        df = load_csv(args.csv)
        src_desc = f"{args.csv} n={len(df)}"

    import strategies
    if args.all:
        keys = [s.key for s in strategies.all_strategies() if s.key not in SKIP_STRATEGIES]
    else:
        keys = [args.strategy]

    out = {"source": src_desc, "results": {}}
    print(f"# Backtest — {src_desc}\n# costs: spread {args.spread}%  fee {args.fee}%/side  "
          f"equity ${args.equity:,.0f}\n")
    for key in keys:
        cfg = BacktestConfig(
            strategy=key, instrument_label=args.label, spread_pct=args.spread,
            fee_pct=args.fee, start_equity=args.equity, confidence_min=args.conf_min,
        )
        if args.walk_forward:
            wf = walk_forward(cfg, df, folds=args.folds)
            out["results"][key] = wf
            if not args.json:
                if wf.get("skipped"):
                    print(f"{key:18} SKIPPED — {wf['skipped']}")
                    continue
                agg = wf["aggregate"]
                print(f"{key:18} [walk-forward {wf['profitable_folds']}/{wf['total_folds']} folds "
                      f"profitable, stable={wf['stable']}]")
                print(_fmt(agg))
        else:
            res = Backtester(cfg).run(df)
            if res.skipped:
                out["results"][key] = {"skipped": res.skipped}
                if not args.json:
                    print(f"{key:18} SKIPPED — {res.skipped}")
                continue
            m = res.metrics()
            out["results"][key] = m
            if not args.json:
                print(f"{key:18}")
                print(_fmt(m))

    if args.json:
        print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
