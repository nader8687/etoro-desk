"""
Backtesting engine — replays eToro history through the LIVE trading stack.

Fidelity principles
-------------------
* SAME code paths as production: strategies.get(key).generate() for signals,
  exit_profiles for stop/TP/trail parameters (which read the Settings tab —
  a backtest therefore reflects your CURRENT exit configuration).
* NO LOOKAHEAD:
    - signals are computed on candles[: i+1] (closed bars only, like live
      candle-close dispatch) and FILLED AT THE NEXT BAR'S OPEN;
    - the chandelier trail/peak is updated with a bar's extremes only AFTER
      that bar has been tested against the PREVIOUS bar's level.
* COSTS: half-spread paid on entry and on exit (spread_pct input), plus the
  optional TRADE_FEE_PCT round-trip fee used elsewhere in analytics.
* CONSERVATIVE intrabar ordering: when several exit levels are touched within
  one bar, the WORST for us fires first (stop → trail → take-profit).

Honest limitations (also shown on the Backtest page)
----------------------------------------------------
* The LLM strategy can't be replayed (per-candle vision calls + journal
  memory would leak future knowledge) — rule strategies only.
* eToro's history endpoint caps the window (~1000 candles), so a 1m backtest
  covers hours, 15m ≈ ten days, 4h ≈ half a year.
* Recovery/breakeven-floor exits are not simulated (they depend on the live
  journal's average-hold statistics).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

WARMUP_BARS = 60   # min closed candles before the first signal is taken


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class BTTrade:
    direction: str          # LONG | SHORT
    entry_idx: int
    exit_idx: int
    entry_time: object
    exit_time: object
    entry_price: float      # cost-adjusted fill
    exit_price: float       # cost-adjusted fill
    pnl_dollars: float
    pnl_pct: float
    reason: str             # stop_loss | trailing_stop | take_profit | reversal | end_of_data
    confidence: int = 0


@dataclass
class BTResult:
    strategy: str
    instrument_label: str
    interval_secs: int
    n_bars: int
    amount: float
    spread_pct: float
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)   # cumulative $ after each closed trade
    regime_skipped: int = 0   # signals suppressed by the live regime filter
    conf_skipped: int = 0     # entries skipped by the min-confidence gate

    # ── metrics ──────────────────────────────────────────────────────────────
    def _pnls(self, trades=None):
        return [t.pnl_dollars for t in (trades if trades is not None else self.trades)]

    def summary(self, trades=None) -> dict:
        pnls = self._pnls(trades)
        n = len(pnls)
        if n == 0:
            return {"n": 0, "pnl": 0.0, "win_rate": 0.0, "pf": 0.0,
                    "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "max_dd": 0.0, "avg_hold_bars": 0.0}
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gw, gl = sum(wins), -sum(losses)
        eq, peak, dd = 0.0, 0.0, 0.0
        for p in pnls:
            eq += p
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        tl = trades if trades is not None else self.trades
        return {
            "n": n,
            "pnl": round(sum(pnls), 2),
            "win_rate": round(len(wins) / n, 3),
            "pf": round(gw / gl, 2) if gl > 0 else (math.inf if gw > 0 else 0.0),
            "expectancy": round(sum(pnls) / n, 2),
            "avg_win": round(gw / len(wins), 2) if wins else 0.0,
            "avg_loss": round(-gl / len(losses), 2) if losses else 0.0,
            "max_dd": round(dd, 2),
            "avg_hold_bars": round(sum(t.exit_idx - t.entry_idx for t in tl) / n, 1),
        }

    def oos_split(self, frac: float = 0.7) -> tuple[dict, dict]:
        """In-sample (first frac of bars) vs out-of-sample (rest) by ENTRY bar."""
        cut = int(self.n_bars * frac)
        ins = [t for t in self.trades if t.entry_idx < cut]
        oos = [t for t in self.trades if t.entry_idx >= cut]
        return self.summary(ins), self.summary(oos)

    def by_reason(self) -> dict:
        out: dict[str, dict] = {}
        for t in self.trades:
            d = out.setdefault(t.reason, {"n": 0, "pnl": 0.0})
            d["n"] += 1
            d["pnl"] = round(d["pnl"] + t.pnl_dollars, 2)
        return out


# ── ATR (same math as regime.py — Wilder RMA on True Range) ──────────────────

def _atr_pct_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return (atr / c * 100.0).fillna(0.0)


# ── Engine ────────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy_key: str,
    instrument_label: str,
    instrument_id: int,
    interval_secs: int,
    *,
    amount: float = 1000.0,
    spread_pct: float = 0.05,
    apply_regime_filter: bool = True,
    apply_exits: bool = True,
    window_bars: int = 0,
    progress_cb=None,
) -> Optional[BTResult]:
    """Replay one strategy over one candle DataFrame.  None if not runnable.

    apply_exits=False runs the NAKED strategy: entries on signal, exits ONLY on
    a strategy reversal (or end of data) — no stop, no trail, no take-profit.
    Comparing the two runs shows exactly what the exit/risk layer contributes.

    window_bars: the live bot's candle_count — every signal is computed on a
    ROLLING window of exactly this many candles (like the live hub), not on a
    growing one.  0 = legacy growing-window behaviour."""
    import strategies
    import exit_profiles
    import regime as regime_mod

    # Mirror the live engine: the regime entry-gate runs only when it's enabled
    # in Settings → Behavior (same flag the bots honour).
    if apply_regime_filter:
        try:
            import user_settings
            apply_regime_filter = bool(user_settings.behavior_settings().regime_filter_enabled)
        except Exception:
            pass

    try:
        strat = strategies.get(strategy_key)
    except Exception:
        return None
    if strat is None or getattr(strat, "is_async", False):
        return None   # async (LLM) strategies are not replayable

    df = df.reset_index(drop=True)
    n = len(df)
    if n < WARMUP_BARS + 10:
        return None

    opens  = df["Open"].astype(float).tolist()
    highs  = df["High"].astype(float).tolist()
    lows   = df["Low"].astype(float).tolist()
    closes = df["Close"].astype(float).tolist()
    times  = df["time"].tolist()
    atrp   = _atr_pct_series(df).tolist()

    half_spread = spread_pct / 100.0 / 2.0
    fee_rt = 0.0
    try:
        import trade_journal
        fee_rt = float(getattr(trade_journal, "_FEE_PCT", 0.0)) / 100.0
    except Exception:
        pass

    trail_mult = exit_profiles.atr_trail_mult(strategy_key)
    trailing_cfg, tp_cfg = exit_profiles.resolve(strategy_key, instrument_label=instrument_label)

    result = BTResult(
        strategy=strategy_key, instrument_label=instrument_label,
        interval_secs=interval_secs, n_bars=n, amount=amount, spread_pct=spread_pct,
    )

    # Mirror the live bot's view: signals start once a FULL live-sized window
    # exists and always see exactly window_bars candles.
    win = int(window_bars) if window_bars and window_bars > 0 else 0
    start_bar = max(WARMUP_BARS, win) if win else WARMUP_BARS
    if n < start_bar + 10:
        return None

    in_pos = False
    direction = ""
    entry_fill = 0.0
    entry_idx = 0
    entry_conf = 0
    stop_level = 0.0
    tp_level = 0.0
    trail_level = 0.0
    peak = 0.0
    pending_signal = None   # (direction, confidence) — filled at next bar open
    equity = 0.0

    def _close(i: int, raw_px: float, reason: str) -> None:
        nonlocal in_pos, equity
        if direction == "LONG":
            exit_fill = raw_px * (1.0 - half_spread)
            move = (exit_fill - entry_fill) / entry_fill
        else:
            exit_fill = raw_px * (1.0 + half_spread)
            move = (entry_fill - exit_fill) / entry_fill
        pnl = amount * move - amount * fee_rt
        equity += pnl
        result.trades.append(BTTrade(
            direction=direction, entry_idx=entry_idx, exit_idx=i,
            entry_time=times[entry_idx], exit_time=times[i],
            entry_price=round(entry_fill, 6), exit_price=round(exit_fill, 6),
            pnl_dollars=round(pnl, 4), pnl_pct=round(move * 100.0, 4),
            reason=reason, confidence=entry_conf,
        ))
        result.equity_curve.append(round(equity, 4))
        in_pos = False

    for i in range(start_bar, n):
        if progress_cb and i % 100 == 0:
            progress_cb(i / n)

        # ── 1. Fill a pending entry at THIS bar's open (no lookahead) ─────────
        if pending_signal is not None and not in_pos:
            d, conf = pending_signal
            pending_signal = None
            raw = opens[i]
            if raw > 0 and atrp[i - 1] > 0:
                direction = d
                entry_conf = conf
                entry_fill = raw * (1.0 + half_spread) if d == "LONG" else raw * (1.0 - half_spread)
                entry_idx = i
                stop_pct = exit_profiles.adaptive_stop_pct(
                    strategy_key, instrument_label, atr_pct=atrp[i - 1],
                )
                if d == "LONG":
                    stop_level = entry_fill * (1.0 - stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 + tp_cfg / 100.0) if tp_cfg > 0 else 0.0
                    peak = entry_fill
                else:
                    stop_level = entry_fill * (1.0 + stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 - tp_cfg / 100.0) if tp_cfg > 0 else 0.0
                    peak = entry_fill
                trail_level = stop_level   # chandelier starts AT the hard stop
                in_pos = True

        # ── 2. Manage an open position against THIS bar ──────────────────────
        if in_pos and apply_exits:
            lo, hi = lows[i], highs[i]
            closed_this_bar = False
            # Conservative ordering: stop → trail → TP, tested on PRIOR levels.
            if direction == "LONG":
                if lo <= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_this_bar = True
                elif trailing_cfg > 0 and lo <= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_this_bar = True
                elif tp_level > 0 and hi >= tp_level:
                    _close(i, tp_level, "take_profit"); closed_this_bar = True
            else:
                if hi >= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_this_bar = True
                elif trailing_cfg > 0 and hi >= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_this_bar = True
                elif tp_level > 0 and lo <= tp_level:
                    _close(i, tp_level, "take_profit"); closed_this_bar = True

            # AFTER testing, update peak + ratchet the chandelier with this bar
            if in_pos and not closed_this_bar and atrp[i] > 0:
                atr_px = atrp[i] / 100.0
                if direction == "LONG":
                    peak = max(peak, hi)
                    trail_level = max(trail_level, peak * (1.0 - trail_mult * atr_px))
                else:
                    peak = min(peak, lo)
                    trail_level = min(trail_level, peak * (1.0 + trail_mult * atr_px))

        # ── 3. Signal on THIS closed bar — rolling live-sized window ─────────
        window = df.iloc[max(0, i + 1 - win) if win else 0 : i + 1]
        c = closes[i]
        ask = c * (1.0 + half_spread)
        bid = c * (1.0 - half_spread)
        try:
            sig = strat.generate(window, ask, bid, instrument_id)
        except Exception:
            sig = None
        s = (getattr(sig, "signal", "") or "").upper() if sig else ""

        if in_pos:
            # Reversal exit at the close of the signal bar (mirrors live).
            if (direction == "LONG" and s == "SELL") or (direction == "SHORT" and s == "BUY"):
                _close(i, c, "reversal")
        elif s in ("BUY", "SELL") and pending_signal is None:
            allowed = True
            if apply_regime_filter:
                # Same gate the live engine applies before an entry: silence
                # strategy families that are in the wrong trend/vol regime.
                try:
                    rs = regime_mod.classify(window)
                    allowed, _why = regime_mod.allows(strategy_key, rs)
                except Exception:
                    allowed = True
            if allowed:
                pending_signal = ("LONG" if s == "BUY" else "SHORT",
                                  int(getattr(sig, "confidence", 0) or 0))
            else:
                result.regime_skipped += 1

    if in_pos:
        _close(n - 1, closes[n - 1], "end_of_data")
    if progress_cb:
        progress_cb(1.0)
    return result


# ── Exit-parameter optimisation (robustness sweep) ────────────────────────────
# Signals are INDEPENDENT of exit parameters, so the expensive part (running the
# strategy on every bar) is computed once; each parameter combo then replays the
# cheap position/exit logic over the cached series.  Guardrails against
# curve-fitting live in the CALLER: rank by OUT-OF-SAMPLE stats, require minimum
# trade counts, and show the whole surface (plateau ≫ peak).

def compute_signal_series(
    df: pd.DataFrame,
    strategy_key: str,
    instrument_id: int,
    spread_pct: float,
    *,
    apply_regime_filter: bool = True,
    window_bars: int = 0,
    progress_cb=None,
) -> Optional[list]:
    """Per-bar (signal, confidence, regime_ok) — exit-agnostic.

    window_bars mirrors the live bot's candle_count: each bar's signal is
    computed on a rolling window of exactly that many candles."""
    import strategies
    import regime as regime_mod
    try:
        strat = strategies.get(strategy_key)
    except Exception:
        return None
    if strat is None or getattr(strat, "is_async", False):
        return None
    df = df.reset_index(drop=True)
    n = len(df)
    win = int(window_bars) if window_bars and window_bars > 0 else 0
    start_bar = max(WARMUP_BARS, win) if win else WARMUP_BARS
    if n < start_bar + 10:
        return None
    closes = df["Close"].astype(float).tolist()
    half_spread = spread_pct / 100.0 / 2.0
    out: list = [("", 0, True)] * n
    for i in range(start_bar, n):
        if progress_cb and i % 100 == 0:
            progress_cb(i / n)
        window = df.iloc[max(0, i + 1 - win) if win else 0 : i + 1]
        c = closes[i]
        try:
            sig = strat.generate(window, c * (1 + half_spread), c * (1 - half_spread), instrument_id)
        except Exception:
            sig = None
        s = (getattr(sig, "signal", "") or "").upper() if sig else ""
        conf = int(getattr(sig, "confidence", 0) or 0) if sig else 0
        ok = True
        if apply_regime_filter and s in ("BUY", "SELL"):
            try:
                rs = regime_mod.classify(window)
                ok, _ = regime_mod.allows(strategy_key, rs)
            except Exception:
                ok = True
        out[i] = (s, conf, ok)
    if progress_cb:
        progress_cb(1.0)
    return out


def simulate_exits(
    df: pd.DataFrame,
    signals: list,
    strategy_key: str,
    instrument_label: str,
    interval_secs: int,
    *,
    stop_mult: float,
    trail_mult: float,
    tp_pct: float,
    min_conf: int = 0,
    amount: float = 1000.0,
    spread_pct: float = 0.05,
    window_bars: int = 0,
) -> BTResult:
    """Replay position/exit logic over a cached signal series with EXPLICIT exit
    parameters.  Mirrors run_backtest's mechanics exactly (same fills, same
    conservative intrabar ordering, same chandelier ratchet) — drift between the
    two is checked in tests by running identical parameters through both."""
    import exit_profiles

    df = df.reset_index(drop=True)
    n = len(df)
    opens  = df["Open"].astype(float).tolist()
    highs  = df["High"].astype(float).tolist()
    lows   = df["Low"].astype(float).tolist()
    closes = df["Close"].astype(float).tolist()
    times  = df["time"].tolist()
    atrp   = _atr_pct_series(df).tolist()
    half_spread = spread_pct / 100.0 / 2.0

    floor = exit_profiles.stop_loss_min_pct(strategy_key, instrument_label)
    s_cfg = exit_profiles._atr_user()
    noise = float(s_cfg.noise_floor_pct) if s_cfg is not None else exit_profiles.ATR_STOP_NOISE_FLOOR_PCT
    widen = float(s_cfg.widen_max) if s_cfg is not None else exit_profiles.STOP_WIDEN_MAX

    result = BTResult(
        strategy=strategy_key, instrument_label=instrument_label,
        interval_secs=interval_secs, n_bars=n, amount=amount, spread_pct=spread_pct,
    )
    in_pos = False
    direction, entry_fill, entry_idx, entry_conf = "", 0.0, 0, 0
    stop_level = tp_level = trail_level = peak = 0.0
    pending = None
    equity = 0.0

    def _close(i, raw_px, reason):
        nonlocal in_pos, equity
        if direction == "LONG":
            exit_fill = raw_px * (1.0 - half_spread)
            move = (exit_fill - entry_fill) / entry_fill
        else:
            exit_fill = raw_px * (1.0 + half_spread)
            move = (entry_fill - exit_fill) / entry_fill
        pnl = amount * move
        equity += pnl
        result.trades.append(BTTrade(
            direction=direction, entry_idx=entry_idx, exit_idx=i,
            entry_time=times[entry_idx], exit_time=times[i],
            entry_price=round(entry_fill, 6), exit_price=round(exit_fill, 6),
            pnl_dollars=round(pnl, 4), pnl_pct=round(move * 100.0, 4),
            reason=reason, confidence=entry_conf,
        ))
        result.equity_curve.append(round(equity, 4))
        in_pos = False

    _win = int(window_bars) if window_bars and window_bars > 0 else 0
    _start = max(WARMUP_BARS, _win) if _win else WARMUP_BARS
    for i in range(_start, n):
        if pending is not None and not in_pos:
            d, conf = pending
            pending = None
            raw = opens[i]
            if raw > 0 and atrp[i - 1] > 0:
                direction, entry_conf, entry_idx = d, conf, i
                entry_fill = raw * (1.0 + half_spread) if d == "LONG" else raw * (1.0 - half_spread)
                stop_pct = min(max(stop_mult * atrp[i - 1], noise), floor * widen)
                if d == "LONG":
                    stop_level = entry_fill * (1.0 - stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 + tp_pct / 100.0) if tp_pct > 0 else 0.0
                else:
                    stop_level = entry_fill * (1.0 + stop_pct / 100.0)
                    tp_level = entry_fill * (1.0 - tp_pct / 100.0) if tp_pct > 0 else 0.0
                peak, trail_level, in_pos = entry_fill, stop_level, True

        if in_pos:
            lo, hi = lows[i], highs[i]
            closed_bar = False
            if direction == "LONG":
                if lo <= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_bar = True
                elif trail_mult > 0 and lo <= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_bar = True
                elif tp_level > 0 and hi >= tp_level:
                    _close(i, tp_level, "take_profit"); closed_bar = True
            else:
                if hi >= stop_level:
                    _close(i, stop_level, "stop_loss"); closed_bar = True
                elif trail_mult > 0 and hi >= trail_level:
                    _close(i, trail_level, "trailing_stop"); closed_bar = True
                elif tp_level > 0 and lo <= tp_level:
                    _close(i, tp_level, "take_profit"); closed_bar = True
            if in_pos and not closed_bar and atrp[i] > 0 and trail_mult > 0:
                atr_px = atrp[i] / 100.0
                if direction == "LONG":
                    peak = max(peak, hi)
                    trail_level = max(trail_level, peak * (1.0 - trail_mult * atr_px))
                else:
                    peak = min(peak, lo)
                    trail_level = min(trail_level, peak * (1.0 + trail_mult * atr_px))

        s, conf, ok = signals[i]
        if in_pos:
            if (direction == "LONG" and s == "SELL") or (direction == "SHORT" and s == "BUY"):
                _close(i, closes[i], "reversal")
        elif s in ("BUY", "SELL") and pending is None:
            if min_conf > 0 and conf < min_conf:
                result.conf_skipped += 1
            elif ok:
                pending = ("LONG" if s == "BUY" else "SHORT", conf)
            else:
                result.regime_skipped += 1

    if in_pos:
        _close(n - 1, closes[n - 1], "end_of_data")
    return result


def optimize_exits(
    df: pd.DataFrame,
    strategy_key: str,
    instrument_label: str,
    instrument_id: int,
    interval_secs: int,
    *,
    amount: float = 1000.0,
    spread_pct: float = 0.05,
    stop_mults=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    trail_mults=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    tp_pcts=(0.0, 0.6, 1.2, 2.0),
    min_confs=(0, 55, 70),
    min_is_trades: int = 8,
    window_bars: int = 0,
    progress_cb=None,
) -> Optional[dict]:
    """Sweep the exit grid over ONE cached signal series.

    Returns {"rows": [...], "signals_n": int} where each row carries the params
    plus IN-SAMPLE (first 70%) and OUT-OF-SAMPLE (last 30%) summaries.  Rows
    with fewer than min_is_trades in-sample trades are marked excluded — too
    little evidence to mean anything."""
    signals = compute_signal_series(
        df, strategy_key, instrument_id, spread_pct,
        window_bars=window_bars,
        progress_cb=(lambda f: progress_cb(f * 0.5)) if progress_cb else None,
    )
    if signals is None:
        return None
    combos = [
        (sm, tm, tp, mc)
        for sm in stop_mults for tm in trail_mults
        for tp in tp_pcts for mc in min_confs
    ]
    rows = []
    for c_i, (sm, tm, tp, mc) in enumerate(combos):
        if progress_cb and c_i % 10 == 0:
            progress_cb(0.5 + 0.5 * c_i / len(combos))
        res = simulate_exits(
            df, signals, strategy_key, instrument_label, interval_secs,
            stop_mult=sm, trail_mult=tm, tp_pct=tp, min_conf=mc,
            amount=amount, spread_pct=spread_pct, window_bars=window_bars,
        )
        ins, oos = res.oos_split()
        rows.append({
            "stop_mult": sm, "trail_mult": tm, "tp_pct": tp, "min_conf": mc,
            "is": ins, "oos": oos,
            "excluded": ins["n"] < min_is_trades,
        })
    if progress_cb:
        progress_cb(1.0)
    return {
        "rows": rows,
        "signals_n": sum(1 for s, _, _ in signals if s in ("BUY", "SELL")),
        # Cached series so the caller can instantly replay ANY combo in full
        # (e.g. to chart the best parameters' trades) without recomputing.
        "signals": signals,
    }
