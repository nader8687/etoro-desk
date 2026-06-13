"""ATR stop sizing decides how much room every trade gets — test the bounds
and relationships that must hold regardless of the user's Settings values."""
from __future__ import annotations

import exit_profiles as ep

STRAT = "supertrend"   # a 'trend' class strategy
LABEL = "Bitcoin  (BTC)"


def test_no_atr_returns_fixed_floor():
    floor = ep.stop_loss_min_pct(STRAT, LABEL)
    assert ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=None) == floor
    assert ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=0.0) == floor


def test_stop_never_below_noise_floor():
    # A near-zero ATR must not produce a microscopic (noise-tripped) stop.
    tiny = ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=0.0001)
    assert tiny >= ep.ATR_STOP_NOISE_FLOOR_PCT - 1e-9


def test_stop_capped_in_volatility_spike():
    floor = ep.stop_loss_min_pct(STRAT, LABEL)
    huge = ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=99.0)
    # Hard cap: never wider than floor x STOP_WIDEN_MAX (max-loss rule).
    assert huge <= floor * ep.STOP_WIDEN_MAX + 1e-9


def test_stop_monotonic_in_atr():
    a = ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=0.5)
    b = ep.adaptive_stop_pct(STRAT, LABEL, atr_pct=1.5)
    assert b >= a   # more volatility never gives a tighter stop


def test_atr_mult_positive():
    assert ep.atr_mult(STRAT) > 0
    assert ep.atr_trail_mult(STRAT) > 0


def test_resolve_returns_two_floats():
    trail, tp = ep.resolve(STRAT, instrument_label=LABEL)
    assert isinstance(trail, float) and isinstance(tp, float)
    assert trail >= 0 and tp >= 0


def test_per_bot_override_beats_class(monkeypatch):
    # A per-bot atr_stop_mult override must win over the class value.
    monkeypatch.setattr(
        ep, "_bot_atr_overrides", lambda bot_key: (5.5, None) if bot_key == "k" else (None, None),
    )
    assert ep.atr_mult(STRAT, "k") == 5.5
    assert ep.atr_mult(STRAT, "other") == ep.atr_mult(STRAT)
