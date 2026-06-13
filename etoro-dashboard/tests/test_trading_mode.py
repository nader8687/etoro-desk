"""The live-trading guard is the single most safety-critical policy here."""
from __future__ import annotations

import pytest

import trading_mode


def test_demo_is_always_allowed(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert trading_mode.trade_allowed(is_demo=True) is True
    trading_mode.assert_trade_allowed(is_demo=True)   # must not raise


def test_live_blocked_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    assert trading_mode.live_trading_allowed() is False
    assert trading_mode.trade_allowed(is_demo=False) is False
    with pytest.raises(PermissionError):
        trading_mode.assert_trade_allowed(is_demo=False)


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_live_opt_in_recognised(monkeypatch, val):
    monkeypatch.setenv("ALLOW_LIVE_TRADING", val)
    assert trading_mode.live_trading_allowed() is True
    assert trading_mode.trade_allowed(is_demo=False) is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "maybe", " "])
def test_live_opt_in_rejects_other_values(monkeypatch, val):
    monkeypatch.setenv("ALLOW_LIVE_TRADING", val)
    assert trading_mode.live_trading_allowed() is False
    with pytest.raises(PermissionError):
        trading_mode.assert_trade_allowed(is_demo=False)


def test_open_trade_refuses_live_without_optin(monkeypatch):
    """The order boundary itself must refuse a non-demo order — defense in
    depth even if the engine gate is bypassed."""
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    import trade_manager

    class _Boom:
        def open_demo_market_by_amount(self, **_):
            raise AssertionError("must never reach the broker on a live-refused order")

    out = trade_manager.open_trade(
        instrument_id=100000, instrument_label="BTC", signal="BUY",
        confidence=80, ask=100.0, bid=99.9, client=_Boom(),
        demo_amount=1000.0, bot_id="test-bot-uuid", is_demo=False,
    )
    assert out is None
