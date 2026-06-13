"""Multi-timeframe exit-check backtester — the equivalence GATE.

The two-timeframe path (slow entries on the interval, faster signal-reversal
exits on a closed lower timeframe) is the validation core for the
exit_check_tf optimizer.  If it cannot reproduce the single-timeframe result
EXACTLY when the LTF equals the interval, none of its finer-TF numbers are
trustworthy — so that reduction is the gate everything else depends on.
"""
from __future__ import annotations

import backtester as bt

STRAT, LABEL, IID, SECS, WIN = "macd", "Bitcoin  (BTC)", 100000, 900, 100
_BASE = dict(stop_mult=2.0, trail_mult=2.0, tp_pct=0.0, window_bars=WIN)


def test_mtf_reduces_to_single_tf_at_interval(ohlc):
    """LTF == interval ⇒ MTF path is bar-for-bar identical to single-TF."""
    sigs = bt.compute_signal_series(ohlc, STRAT, IID, 0.05, window_bars=WIN)
    assert sigs is not None
    single = bt.simulate_exits(ohlc, sigs, STRAT, LABEL, SECS, **_BASE)
    rev_map = bt.build_exit_rev_by_bar(ohlc, ohlc, sigs)        # LTF == HTF
    mtf = bt.simulate_exits(ohlc, sigs, STRAT, LABEL, SECS, exit_rev_by_bar=rev_map, **_BASE)

    assert len(single.trades) == len(mtf.trades)
    for a, b in zip(single.trades, mtf.trades):
        assert (a.entry_idx, a.exit_idx, a.reason) == (b.entry_idx, b.exit_idx, b.reason)
        assert abs(a.entry_price - b.entry_price) < 1e-9
        assert abs(a.exit_price - b.exit_price) < 1e-9
        assert abs(a.pnl_dollars - b.pnl_dollars) < 1e-9
    assert abs(single.summary()["pnl"] - mtf.summary()["pnl"]) < 1e-9


def test_mtf_map_shape_at_interval(ohlc):
    """At interval, each HTF bar maps to exactly itself (its own signal/close)."""
    sigs = bt.compute_signal_series(ohlc, STRAT, IID, 0.05, window_bars=WIN)
    rev_map = bt.build_exit_rev_by_bar(ohlc, ohlc, sigs)
    assert len(rev_map) == len(ohlc)
    closes = ohlc["Close"].astype(float).tolist()
    for i, events in enumerate(rev_map):
        assert len(events) == 1
        sig, px = events[0]
        assert sig == sigs[i][0]
        assert abs(px - closes[i]) < 1e-9


def test_mtf_ltf_reversal_fires_early(ohlc):
    """Injecting an opposing LTF event mid-trade forces an earlier reversal exit
    — proves the LTF branch actually acts, not just reduces."""
    sigs = bt.compute_signal_series(ohlc, STRAT, IID, 0.05, window_bars=WIN)
    single = bt.simulate_exits(ohlc, sigs, STRAT, LABEL, SECS, **_BASE)
    # find a trade with room to inject an earlier reversal
    tr = next((t for t in single.trades if t.exit_idx - t.entry_idx >= 3), None)
    if tr is None:
        return  # data-dependent; nothing to assert this fixture
    closes = ohlc["Close"].astype(float).tolist()
    rev_map = bt.build_exit_rev_by_bar(ohlc, ohlc, sigs)
    inject_i = tr.entry_idx + 1
    opp = "SELL" if tr.direction == "LONG" else "BUY"
    rev_map[inject_i] = [(opp, closes[inject_i])]      # opposing signal mid-trade
    mtf = bt.simulate_exits(ohlc, sigs, STRAT, LABEL, SECS, exit_rev_by_bar=rev_map, **_BASE)
    same_entry = [t for t in mtf.trades if t.entry_idx == tr.entry_idx]
    assert same_entry, "the trade should still open at the same bar"
    assert same_entry[0].exit_idx <= inject_i        # exited no later than the injection
    assert same_entry[0].reason == "reversal"
