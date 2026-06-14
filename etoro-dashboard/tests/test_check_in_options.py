"""Exit check-in candidate snapping for fleet optimization."""
from __future__ import annotations

import instrument_config as ic


def test_check_in_options_hourly():
    opts = ic.check_in_options(3600)
    assert opts == [3600, 1800, 900]


def test_check_in_options_thirty_min():
    opts = ic.check_in_options(1800)
    assert 1800 in opts
    assert 900 in opts
    # ¼ of 30m = 7.5m → nearest supported is 5m or 10m (tie → finer 5m)
    assert 300 in opts


def test_check_in_options_one_minute_collapses():
    assert ic.check_in_options(60) == [60]


def test_check_in_options_never_exceeds_trade_interval():
    for base in (60, 300, 900, 1800, 3600, 14400):
        for ci in ic.check_in_options(base):
            assert ci <= base
