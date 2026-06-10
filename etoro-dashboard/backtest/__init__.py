"""Offline backtesting + walk-forward evaluation for EtoroDesk strategies."""
from .engine import Backtester, BacktestConfig, walk_forward, synthetic_ohlc, load_csv

__all__ = ["Backtester", "BacktestConfig", "walk_forward", "synthetic_ohlc", "load_csv"]
