# EtoroDesk — Quant / Risk / Execution Audit

*Reviewer role: quant researcher + algo engineer + risk manager. Scope: code as of June 2026.
Verdict basis = system design and logic, not current P&L (data still being collected).*

---

## 0. What the system actually is

- **Fleet:** 15 strategies × 2 assets (XRP, BTC) = **30 bots**, each bot = one `(asset × strategy)` pair holding its **own** independent position. Configured declaratively in `instruments.toml`.
- **Strategies:** LLM vision bot + 14 classical indicators (Supertrend, MA-cross, MACD, ADX, Ichimoku, Donchian, ORB, RSI, StochRSI, Bollinger squeeze, Z-score mean-reversion, candlestick, stat-arb, rate-arb).
- **Per-trade risk layer (`position_sizer.py`, `exit_profiles.py`, `cash_manager.py`):** risk-based sizing, performance-adaptive multiplier, per-strategy exit profiles, intelligent cash-freeing. **This is the strongest part of the system** — more mature than most retail bots.
- **Execution-quality gate (`execution_quality.py`):** models spread + ATR vol buffer slippage, edge decay by strategy half-life, net-edge EV test, vetoes HIGH-friction trades. Good.
- **Journal (`trade_journal.py`):** forward log of closed trades with PF / win-rate / by-strategy aggregation, and an evidence-based entry veto.

This is a competent, thoughtfully-built system. The gaps below are **structural**, not sloppy.

---

## 1. Current strategy logic (classification & condition fit)

| Bot | Type | Best regime | Notes / weakness |
|---|---|---|---|
| Supertrend, MA-cross, MACD, ADX, Ichimoku | Trend-following | Trending, mid/high vol | Whipsaw in chop; **fixed confidence constants** (e.g. Supertrend always 78) not calibrated to hit-rate |
| Donchian, ORB | Breakout | Expansion / range-break | False breakouts in low vol; ORB half-life 90 s is realistic |
| RSI, StochRSI, Bollinger, Z-score MR, Candlestick | Mean-reversion / oscillator | **Ranging only** | **No trend filter** → these knife-catch in strong trends. The single biggest per-strategy flaw |
| Stat-arb, Rate-arb | Labelled "arbitrage", actually **cross-asset mean-reversion** | Range-bound, cointegrated pair | XRP/BTC are correlated but **not cointegration-tested**; no hedge leg → this is directional MR, not arb. Mislabeled risk |
| LLM | Discretionary / hybrid | Any (model's call) | Confidence floor 70, temp 0.1; adds API cost + up to 60 s latency; hardest to validate |

**Cross-cutting:** confidence scores are hand-set heuristics, not empirical. ADX (a natural regime filter) is used as its *own* bot but **not** as a meta-filter to silence the mean-reversion bots in trends.

---

## 2. Main profitability risks

1. **Fake diversification.** 30 bots, but only **2 highly-correlated crypto assets**. Most "diversification" is illusory — in a BTC selloff, XRP follows, and trend bots on both assets stack the same directional loss.
2. **No portfolio-level edge.** Each bot optimises itself; nothing optimises the *book*. Mean-reversion bots (BUY the dip) and trend bots (SELL the breakdown) routinely take **opposite positions on the same asset at the same time** → internal hedging that pays the spread twice and nets ~0 before costs.
3. **Unvalidated edge.** **No backtest, no walk-forward, no tests anywhere** (confirmed by grep). Confidence numbers, half-lives and exit %s are plausible guesses. There is no evidence any strategy has positive expectancy after costs.
4. **Cost drag.** ~0.6–1.0% effective round-trip crypto spread on eToro + LLM API cost. Arb/MR targets (0.6–1.2%) are *barely* above the spread — fragile.

## 3. Main execution risks

1. **Coordinated stacking.** No cap on concurrent positions, net directional exposure, or correlated trades. A regime flip can fire 6–10 bots the same way within one candle — bounded only by the cash reserve. That is a portfolio-heat time-bomb.
2. **Stops are not volatility-adaptive.** `compute_stop_loss_price` = `max(2×spread, fixed_pct)`. The fixed % comes from a static per-strategy/asset-class table — **not live ATR**. In a vol spike, stops are too tight (noise-stopped); in dead markets, too wide. Sizing divides by this stop, so the "vol-adjusted sizing" is only half-real.
3. **LLM latency.** Signals fire on candle close but the LLM round-trip is up to 60 s; on 1-min context that's a meaningful fill delay. `edge_decay` *models* it but the order still goes in late.
4. **No portfolio drawdown kill-switch.** Per-trade stops exist; there is no "halt all bots at −X% day / −Y% peak-to-trough" circuit breaker.
5. **Idempotency is local.** Duplicate-order protection relies on `has_open(bot_uuid)` + in-process locks; survives reruns but not a hard restart mid-open (guards exist but it's not an eToro-side idempotency key).

## 4. Risk-management gaps (summary)

- ❌ No master/portfolio risk manager (exposure, heat, correlation, net-bias caps).
- ❌ No drawdown circuit breaker.
- ❌ Stops/targets not ATR/regime-adaptive.
- ❌ No regime detector gating which *family* of bots may trade.
- ⚠️ Performance multiplier is pro-cyclical (sizes up after wins) on a 5-trade minimum sample — mild overfit risk.
- ✅ Per-trade sizing, cash reserve, exec-quality veto, cooldowns — solid.

## 5. Logging / analytics gaps

Journal already records: bot_id, strategy, instrument, direction, entry/exit price, spread, slippage_pct, trade_amount, exec_risk, net_edge_pct, close reason, peak_pnl, stop_loss_price, holding_min, pnl.

Missing for the user's spec: **interval/timeframe, take-profit level, fees, market-condition/regime label, explicit entry-reason text, asset_class**. Analytics has PF + win-rate but **no expectancy, Sharpe/Sortino, or max-drawdown** ranking.

---

## 6. Verdict — is there a realistic path to profitability?

**Conditionally yes — but not demonstrated, and not on the current 2-asset book without changes.**

The plumbing (sizing, exec-quality gating, adaptive exits, journaling) is genuinely good and rare for a retail system. What's missing is exactly the part that determines whether a *fleet* of bots makes money rather than just trades a lot:

- **Blocker 1 — Validation:** no backtest/walk-forward means zero proof of edge. Must be built before any capital judgement.
- **Blocker 2 — Portfolio risk:** 30 correlated bots with no master controller will produce coordinated drawdowns and self-cancelling churn. A risk manager that caps net exposure, correlated stacking, and total heat is mandatory.
- **Blocker 3 — Regime & adaptive stops:** fixed stops + uncalibrated confidence + no regime gate make any per-strategy edge fragile across market states.

Fix those three and the per-trade machinery already in place gives this a credible shot. Left as-is, it will most likely **churn to roughly break-even-minus-costs** regardless of how good individual signals look.

---

## 7. Prioritised improvement plan

**P0 — Validate & protect (do first, lowest risk, highest leverage)**
1. **Master Risk Manager** (`risk_manager.py`, new): single gate called in `_maybe_open_trade` before sizing. Caps: max concurrent positions, max gross & net directional exposure, max correlated-cluster exposure, total portfolio heat (Σ risk_$), and a day/drawdown kill-switch. Pure additive gate — preserves all existing behaviour when within limits.
2. **Backtest + walk-forward harness** (`backtest/`, new): replays OHLCV through the *same* strategy classes with spread/fee/slippage/latency modelling, no lookahead, walk-forward splits. Zero risk to live code.
3. **Richer journal + analytics**: add interval, take_profit, fees, regime label, entry_reason, asset_class to the record; add expectancy / profit-factor / Sharpe / Sortino / max-DD ranking.

**P1 — Sharpen the edge**
4. **ATR-adaptive stops & targets** (extend `exit_profiles`/`compute_stop_loss_price`): stop distance = `k × ATR%` (regime-aware) instead of a fixed %.
5. **Regime filter**: a shared regime signal (e.g. ADX/EMA-slope/vol-percentile) that silences mean-reversion bots in strong trends and breakout bots in dead vol.
6. **Confidence recalibration**: replace hard-coded confidence constants with journal-derived hit-rate once enough samples exist (guarded, no overfit on small N).

**P2 — Structural**
7. **Real diversification**: add uncorrelated assets / timeframes; treat stat-arb properly (cointegration test + hedge leg) or relabel it MR.
8. **Latency/idempotency hardening**: eToro-side idempotency key; consider tightening LLM timeout / pre-computing on partial candles.

---

*Recommended starting point: P0 items 1–3. They are additive, reversible, and turn the system from "untestable" into "measurable" without touching live order semantics.*
