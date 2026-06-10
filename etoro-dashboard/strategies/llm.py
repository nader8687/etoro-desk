"""LLM strategy — delegates to the visual-bot via signal_worker (async)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

import signal_worker
from .base import Strategy, StrategySignal


class LLMStrategy(Strategy):
    key          = "llm"
    display_name = "LLM (AI Vision)"
    description  = (
        "Sends the chart image to the Visual-Bot LLM for analysis. "
        "Generates rich reasoning, confidence scores, and risk warnings."
    )

    @property
    def is_async(self) -> bool:
        return True

    def generate(
        self,
        df: pd.DataFrame,
        ask: float,
        bid: float,
        instrument_id: int,
        **kwargs,
    ) -> Optional[StrategySignal]:
        signal_worker.request_signal(
            df,
            instrument_id,
            kwargs.get("instrument_label", ""),
            kwargs.get("interval_label", "1 Minute"),
            trigger_at=kwargs.get("trigger_at", ""),
            ask=ask if ask else None,
            bid=bid if bid else None,
            bot_id=kwargs.get("bot_id", ""),
        )
        return None  # result arrives later via signal_worker.get_result()
