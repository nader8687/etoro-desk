# EtoroDesk — Quant Improvements (Implementation Report)

*Companion to `docs/quant-audit.md`. Covers what was built, why it raises the
profitability ceiling, what risk remains, and how to run/test/backtest it.*

---

## What changed

### New modules
| File | Purpose |
|---|---|
| `risk_manager.py` | **Master portfolio risk gate.** Single check between signal and order that inspects the *combined* book: caps concurrent positions, gross exposure, correlated-cluster gross/net, same-direction stacking, per-asset count, portfolio heat (Σ risk-to-stop), anti-internal-hedge, and a daily-drawdown kill-switch. No API calls; fails open. |
| `regime.py` | **Market-regime detector.** ADX + EMA-slope + ATR-percentile → `trend ∈ {up,down,range}`, `vol ∈ {low,normal,high}`. Drives both the entry filter and ATR stop sizing. Fails open to "unknown" (allows). |
| `analytics.py` | **Risk-adjusted bot ranking.** Expectancy ($ and R), profit factor, win rate, payoff, max drawdown, Sharpe, Sortino — with a `MIN_SAMPLE` guard so nothing is ranked on noise. |
| `backtest/` | **Event-driven backtester + walk-forward.** Reuses the live strategy classes; models spread, fees, ATR slippage, and one-bar execution latency (which also prevents look-ahead). CLI: `python -m backtest.run`. |

### Modified (additive, backward-compatible)
| File | Change |
|---|---|
| `trading_engine.py` | Entry path now runs, in order: exec-quality gate → journal guidance → **regime filter** → sizing → **portfolio risk gate** (shrink/block) → open. Computes live ATR% + **calibrated confidence**. |
| `exit_profiles.py` | Added `adaptive_stop_pct()` — stop = `clamp(k×ATR%, floor, 3×floor)`. Fixed % becomes a calm-market floor; widens in vol spikes. Falls back to the old fixed % when no ATR. |
| `trade_manager.py` | `open_trade()` accepts `atr_pct`, `interval`, `entry_reason`, `regime`, `confidence_calibrated`; stop now uses the adaptive distance; new fields stored on the trade. |
| `trade_journal.py` | Records add `interval, entry_reason, regime, atr_pct_entry, stop_pct_entry, asset_class, take_profit_pct, fee_pct, fee_dollars, confidence_calibrated`. Added `calibrated_confidence()` (Bayesian shrinkage toward realised win rate). |
| `instruments.toml` | New documented `[risk]` section (all caps tunable, hot-reloaded); diversification guidance for adding uncorrelated assets. |

All caps are **default-on but conservative** — they only bite when the *fleet*
concentrates risk; a small book trades as before.

---

## Why it improves profitability potential

1. **Stops correlated stacking (the #1 fleet risk).** 30 bots on 2 correlated
   crypto assets could pile into one directional bet bounded only by cash. The
   cluster net/gross and same-direction caps convert that into controlled,
   diversified exposure — directly reducing coordinated drawdowns.
2. **Kills self-hedging churn.** The anti-internal-hedge rule stops a bot opening
   directly against a larger same-asset position — eliminating spread paid to net
   flat, a pure cost leak.
3. **Regime-aware edge.** Mean-reversion bots no longer knife-catch strong trends;
   trend bots sit out dead low-vol chop. Trading the right strategy in the right
   regime is the highest-ROI improvement to per-strategy expectancy.
4. **ATR-adaptive stops** keep "risk per unit of normal noise" constant across vol
   regimes — fewer noise-stops in calm markets, sane risk in spikes. Because the
   sizer divides risk-$ by stop-%, sizing is now genuinely volatility-adjusted.
5. **Validation + measurement.** The backtester turns "we think this has edge"
   into a number, with realistic costs and walk-forward stability. Analytics rank
   bots by *risk-adjusted* return so capital can later flow to real edge, not
   lucky streaks.
6. **Drawdown kill-switch** caps the worst-day tail — the single most important
   determinant of long-run survival.

---

## What risks remain

- **No proven edge yet.** The machinery is sound, but until the backtester is run
  on real historical candles (and the live journal accrues ≥ a few hundred trades
  per strategy), profitability is unproven. Do **not** judge on small samples.
- **Only 2 correlated assets.** The risk manager *contains* this, but true
  diversification still requires adding uncorrelated instruments (gold, equities,
  FX). Scaffolding/guidance is in `instruments.toml`; the assets aren't added.
- **LLM, stat-arb, rate-arb are not backtestable here** (async / need a second
  feed). They must be validated in forward paper trading.
- **Stat-arb/rate-arb are not true arbitrage** (no cointegration test, no hedge
  leg) — they're cross-asset mean-reversion. Treat their risk accordingly.
- **Idempotency is still local.** Dedup relies on per-bot locks + position-id
  diffing (robust in-process, survives reruns) but there is no eToro-side
  idempotency key, so a hard crash mid-open is a small residual risk. Left as a
  documented next step rather than changed blind.
- **Confidence calibration is conservative by design** — it does nothing until
  ≥5 samples and only partially adjusts; it won't rescue a bad signal, just
  de-weights chronically over-confident ones.

---

## Commands — run, test, backtest

```bash
# ── Run the system (unchanged) ───────────────────────────────────────────────
docker compose up --build -d
#   dashboard  → http://localhost:8501
#   visual-bot → http://localhost:8083/docs

# ── Compile-check everything (run from etoro-dashboard/) ─────────────────────
python -m py_compile *.py backtest/*.py

# ── Backtest a single strategy on real OHLC candles ──────────────────────────
#   CSV needs columns: Open,High,Low,Close (+ optional time,Volume)
python -m backtest.run --csv data/btc_15m.csv --strategy supertrend --walk-forward

# ── Backtest every deterministic strategy with realistic costs ───────────────
python -m backtest.run --csv data/btc_15m.csv --all --walk-forward \
    --spread 0.06 --fee 0.0 --folds 5

# ── Smoke test with no data file (synthetic candles) ─────────────────────────
python -m backtest.run --synthetic --strategy rsi --walk-forward

# ── JSON output for piping into a report ─────────────────────────────────────
python -m backtest.run --csv data/xrp_1m.csv --all --json > bt_results.json

# ── Tune risk caps live (hot-reloaded, no restart) ───────────────────────────
#   edit the [risk] section of etoro-dashboard/instruments.toml
```

To pull real candles for backtesting, export them from eToro history (or the
existing `etoro_client.get_hist_candles`) into a CSV and point `--csv` at it.

### Verification status
The four new modules (`risk_manager`, `regime`, `analytics`, `backtest`) were
unit-smoke-tested: the risk gate's stacking/exposure/heat/drawdown rules, the
backtester's fill/exit/PnL math, walk-forward fold reporting, ATR-stop clamping,
and analytics metrics all behave correctly. Run the `py_compile` line above once
to confirm the edited files parse in your environment.

---

## What data must still be collected before judging profitability

1. **Real historical candles** (≥ 6–12 months, per asset/timeframe) to backtest
   the deterministic strategies with walk-forward — the fastest path to an edge
   verdict that doesn't risk capital.
2. **A meaningful live sample: ≥ ~30 closed trades per (strategy × asset)** before
   any per-bot ranking is trusted (analytics flags anything below `MIN_SAMPLE=20`
   as insufficient). For Sharpe/Sortino stability, prefer ≥ 100.
3. **Regime coverage** — trades across up-trend, down-trend, and range regimes
   (now logged per trade), so you can see *where* each bot makes or loses money,
   not just an aggregate.
4. **Cost reality check** — compare logged `slippage_pct`/`entry_spread` to the
   backtest assumptions; if live spreads exceed the model, re-run backtests with
   the real number before believing any positive expectancy.
5. **Drawdown path** — at least one full peak-to-trough cycle to confirm the
   kill-switch and heat caps behave as intended under stress.

**Bottom line:** the system now has the risk controls, regime-awareness, logging,
and validation tooling a profitable multi-bot system needs. Profitability itself
remains to be *demonstrated* — run the backtests on real candles and let the
enriched journal accrue before drawing conclusions.
