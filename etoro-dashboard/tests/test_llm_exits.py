"""LLM-managed exits — the asymmetric stop/TP rule set.

Stop is a safety floor (tighten-only, never loosen, reject garbage); TP is
uncapped (LLM free to raise/lower/drop). These guard the exact invariants the
design locked, on pure helpers that run without the engine or the broker.
"""
from __future__ import annotations

import trade_manager as tm


# ── _validate_llm_level ───────────────────────────────────────────────────────
def test_validate_rejects_non_numeric():
    assert tm._validate_llm_level("abc", entry=100, direction="LONG", side="stop") is None
    assert tm._validate_llm_level(None, entry=100, direction="LONG", side="tp") is None


def test_validate_rejects_wrong_side():
    # LONG stop must be BELOW entry; a stop above entry is nonsense → rejected
    assert tm._validate_llm_level(105, entry=100, direction="LONG", side="stop") is None
    # LONG tp must be ABOVE entry
    assert tm._validate_llm_level(95, entry=100, direction="LONG", side="tp") is None
    # SHORT stop must be ABOVE entry
    assert tm._validate_llm_level(95, entry=100, direction="SHORT", side="stop") is None


def test_validate_accepts_correct_side():
    assert tm._validate_llm_level(97, entry=100, direction="LONG", side="stop") == 97
    assert tm._validate_llm_level(108, entry=100, direction="LONG", side="tp") == 108
    assert tm._validate_llm_level(103, entry=100, direction="SHORT", side="stop") == 103


def test_validate_rejects_absurd_distance():
    # >50% away is almost certainly a hallucination
    assert tm._validate_llm_level(40, entry=100, direction="LONG", side="stop") is None
    assert tm._validate_llm_level(1000, entry=100, direction="LONG", side="tp") is None


# ── _ratchet_stop_price (NEVER loosens) ──────────────────────────────────────
def test_ratchet_long_tightens_only():
    # raising a long's stop is tightening → accepted
    assert tm._ratchet_stop_price("LONG", 95.0, 97.0) == 97.0
    # lowering it would loosen → rejected, keep current
    assert tm._ratchet_stop_price("LONG", 95.0, 92.0) == 95.0


def test_ratchet_short_tightens_only():
    # lowering a short's stop is tightening → accepted
    assert tm._ratchet_stop_price("SHORT", 105.0, 103.0) == 103.0
    # raising it would loosen → rejected, keep current
    assert tm._ratchet_stop_price("SHORT", 105.0, 108.0) == 105.0


def test_ratchet_none_is_noop():
    assert tm._ratchet_stop_price("LONG", 95.0, None) == 95.0


# ── apply_llm_exit_update (integration of both rules on a PaperTrade) ─────────
def _long_trade():
    from datetime import datetime, timezone
    return tm.PaperTrade(
        instrument_id=1, instrument_label="X", direction="LONG",
        entry_price=100.0, entry_spread=0.05, stop_loss_price=95.0,
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc), signal="BUY", confidence=80,
    )


def test_apply_tightens_stop_and_sets_tp_uncapped():
    t = _long_trade()
    out = tm.apply_llm_exit_update(t, llm_stop_price=98.0, llm_take_profit_price=130.0)
    assert out["stop_tightened"] is True and t.stop_loss_price == 98.0
    assert out["tp_set"] is True
    assert t.llm_take_profit_pct == 30.0   # 130 vs 100 entry — NOT capped


def test_apply_never_loosens_stop():
    t = _long_trade()
    tm.apply_llm_exit_update(t, llm_stop_price=90.0)   # would loosen
    assert t.stop_loss_price == 95.0                   # unchanged


def test_apply_ignores_garbage_and_reports():
    t = _long_trade()
    out = tm.apply_llm_exit_update(t, llm_stop_price="oops", llm_take_profit_price=9999)
    assert t.stop_loss_price == 95.0 and t.llm_take_profit_pct is None
    assert "stop" in out["rejected"] and "tp" in out["rejected"]
