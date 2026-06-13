"""US-session calendar drives the auction gate and stock-bot market hours."""
from __future__ import annotations

from datetime import datetime, timezone

import market_calendar as mc

# 2026-06-15 is a Monday and not a US market holiday (Juneteenth is the 19th).
_MON = datetime(2026, 6, 15, tzinfo=timezone.utc)


def test_weekend_is_closed():
    sat = datetime(2026, 6, 13, 15, 0, tzinfo=timezone.utc)
    sun = datetime(2026, 6, 14, 15, 0, tzinfo=timezone.utc)
    assert mc.us_session_state(sat) == "closed"
    assert mc.us_session_state(sun) == "closed"
    assert mc.is_us_equity_open(sat) is False


def test_regular_session_open_and_closed():
    # 14:30 UTC = 10:30 ET → open;  02:00 UTC = 22:00 prev ET → closed
    assert mc.us_session_state(_MON.replace(hour=14, minute=30)) == "open"
    assert mc.us_session_state(_MON.replace(hour=2)) == "closed"


def test_juneteenth_holiday_closed():
    juneteenth = datetime(2026, 6, 19, 15, 0, tzinfo=timezone.utc)
    assert mc.us_session_state(juneteenth) == "closed"


def test_session_edge_minutes_none_when_closed():
    assert mc.us_session_edge_minutes(_MON.replace(hour=2)) is None


def test_session_edge_minutes_at_open():
    # 5 min after the 13:30 UTC open: ~5 since open, ~385 to a 20:00 UTC close
    edge = mc.us_session_edge_minutes(_MON.replace(hour=13, minute=35))
    assert edge is not None
    since_open, to_close = edge
    assert 0 <= since_open <= 10
    assert to_close > 60


def test_naive_datetime_treated_as_utc():
    naive = datetime(2026, 6, 15, 14, 30)   # no tzinfo
    assert mc.us_session_state(naive) == "open"
