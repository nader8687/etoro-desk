"""Pytest setup: make the app package importable and provide synthetic data.

Tests target PURE / deterministic logic only — risk policy, exit math, calendar,
sizing fail-open, and backtest determinism/invariants.  They never touch the
broker, the network, or live engine state, so they are safe to run anywhere.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

# App modules live one dir up (the container runs from /app; locally from
# etoro-dashboard/).  Make both work.
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _synthetic_ohlc(n: int = 400, seed: int = 7, drift: float = 0.04,
                    swing: float = 6.0, period: int = 25) -> pd.DataFrame:
    """Deterministic OHLC with trend + cyclic swings + noise (no network)."""
    rng = np.random.default_rng(seed)
    close = (100.0
             + np.cumsum(rng.normal(drift, 1.0, n))
             + swing * np.sin(np.arange(n) / period))
    high = close + np.abs(rng.normal(0.4, 0.25, n))
    low = close - np.abs(rng.normal(0.4, 0.25, n))
    openp = np.concatenate([[close[0]], close[:-1]])
    ts = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"time": ts, "Open": openp, "High": high,
                         "Low": low, "Close": close})


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return _synthetic_ohlc()
