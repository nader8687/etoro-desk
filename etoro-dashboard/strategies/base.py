"""Base class and signal dataclass for all trading strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class StrategySignal:
    signal: str                          # "BUY" | "SELL" | "HOLD"
    confidence: int                      # 0–100
    reasoning: str
    risk_level: str = "MEDIUM"           # "LOW" | "MEDIUM" | "HIGH"
    observations: list[str] = field(default_factory=list)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def to_result_dict(self, trigger_at: str = "") -> dict:
        """Convert to the same format signal_worker stores for LLM results."""
        return {
            "signal":       self.signal,
            "confidence":   self.confidence,
            "reasoning":    self.reasoning,
            "risk_level":   self.risk_level,
            "observations": self.observations,
            "_status":      "done",
            "_at":          trigger_at,
        }


class Strategy(ABC):
    key:          str = ""
    display_name: str = ""
    description:  str = ""

    @property
    def is_async(self) -> bool:
        """True for strategies that fire background jobs (e.g. LLM).
        The caller must not expect an immediate return value."""
        return False

    @abstractmethod
    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        """
        Compute a signal from the latest OHLC data.

        Called on each candle close.
        - Synchronous strategies: return a StrategySignal (or None to skip).
        - Async strategies (LLM): fire a background job and return None;
          the result lands in signal_worker independently.
        """
