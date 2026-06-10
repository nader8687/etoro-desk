"""
Execution quality assessment: slippage, edge decay, and execution risk.

Called after every strategy signal to determine whether the theoretical
edge survives real-world execution costs.  Every signal gets these fields
added to its result dict so the engine, UI, and signal log all share a
consistent picture.

Concepts
--------
slippage_pct
    Round-trip execution cost as a percentage of mid price.
    = bid-ask spread  +  ATR-based volatility buffer
    The spread cost is paid once on entry and once on exit; the volatility
    buffer models the extra fill-price movement during order submission.

edge_decay
    A [0, 1] multiplier that shrinks a signal's edge the older it gets.
    Modelled as exponential decay: after one half-life the decay is 0.5.
    Each strategy has a calibrated half-life (ORB decays in ~90 s; MA
    crossover stays valid for ~600 s).  At candle-close dispatch the age
    is 0 and decay = 1.0.  For async strategies (LLM) the actual LLM
    response time is used as the age.

exec_risk
    LOW / MEDIUM / HIGH tier based on spread width and ATR volatility.
    HIGH conditions are always vetoed regardless of confidence.

gross_edge_pct
    Expected gross edge derived from a simple risk-reward model:
        P_win × (RR × stop_pct)  −  P_lose × stop_pct
    where P_win = (confidence / 100) × edge_decay and stop_pct = 0.5 %.

net_edge_pct
    gross_edge_pct  −  slippage_pct

viable
    True → trade should be placed.
    Rules (in order, first match wins):
        • exec_risk == HIGH               → False  (always)
        • net_edge_pct > 0                → True   (positive EV)
        • confidence ≥ 75, risk == LOW    → True   (high conviction, tight market)
        • confidence ≥ 85, risk == MEDIUM → True   (very high conviction)
        • otherwise                        → False
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── Strategy-specific edge half-lives (seconds) ───────────────────────────────
# Shorter half-life → signal decays faster after generation.
HALF_LIVES: dict[str, float] = {
    # ── AI ────────────────────────────────────────────────────────────────────
    "llm":               300.0,   # LLM: 5 min — forward-looking analysis

    # ── Trend-following ───────────────────────────────────────────────────────
    "supertrend":        600.0,   # Supertrend: 10 min — ATR trend persists
    "ma_crossover":      600.0,   # MA cross: 10 min — trend signals last longer
    "macd":              300.0,   # MACD: 5 min — momentum can fade mid-candle
    "adx":               600.0,   # ADX: 10 min — directional moves persist
    "ichimoku":          900.0,   # Ichimoku: 15 min — cloud signals are slow-moving

    # ── Breakout ─────────────────────────────────────────────────────────────
    "orb":                90.0,   # ORB: 90 s — momentum breakouts fade fast
    "donchian":          240.0,   # Donchian: 4 min — rolling breakouts moderate decay

    # ── Mean reversion / oscillators ─────────────────────────────────────────
    "rsi":               180.0,   # RSI: 3 min — divergences fade quickly
    "stoch_rsi":         120.0,   # StochRSI: 2 min — faster than RSI, decays sooner
    "bollinger_squeeze":  600.0,  # Bollinger: 10 min — squeezes resolve slowly
    "mean_reversion":    120.0,   # Mean reversion: 2 min — snap-back is immediate or not

    # ── Arbitrage ─────────────────────────────────────────────────────────────
    "stat_arb":          120.0,   # Stat arb: 2 min — spreads snap back fast
    "rate_arb":          120.0,   # Rate arb: 2 min — similar

    # ── Price action ──────────────────────────────────────────────────────────
    "candlestick":        60.0,   # Candlestick: 1 min — patterns valid 1-2 candles only
}
DEFAULT_HALF_LIFE = 300.0

# Risk-reward ratio assumed for edge estimation (1 : RR_RATIO)
RR_RATIO = 1.5

# Assumed stop-loss width as % of mid price (~2–3× typical 1-min ATR)
BASE_STOP_PCT = 0.5


@dataclass
class ExecutionQuality:
    slippage_pct:   float               # round-trip execution cost as % of price
    edge_decay:     float               # 0–1; fraction of signal edge remaining
    exec_risk:      str                 # "LOW" | "MEDIUM" | "HIGH"
    gross_edge_pct: float               # expected edge before slippage
    net_edge_pct:   float               # gross_edge_pct − slippage_pct
    viable:         bool                # True = trade should be placed
    notes:          list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slippage_pct":   self.slippage_pct,
            "edge_decay":     self.edge_decay,
            "exec_risk":      self.exec_risk,
            "gross_edge_pct": self.gross_edge_pct,
            "net_edge_pct":   self.net_edge_pct,
            "viable":         self.viable,
            "exec_notes":     self.notes,
        }


def assess(
    df: pd.DataFrame,
    ask: float,
    bid: float,
    strategy_key: str,
    confidence: int,
    signal: str,
    signal_age_seconds: float = 0.0,
    atr_periods: int = 14,
) -> ExecutionQuality:
    """
    Assess execution quality for a strategy signal.

    Parameters
    ----------
    df                  : OHLCV DataFrame, most-recent candle last
    ask / bid           : current best quotes (use 0 if unavailable)
    strategy_key        : registry key — affects edge half-life
    confidence          : strategy confidence 0–100
    signal              : "BUY" | "SELL" | "HOLD"
    signal_age_seconds  : seconds since signal was generated (0 at candle close)
    atr_periods         : lookback window for ATR volatility estimate
    """
    notes: list[str] = []

    # HOLD signals never execute — fast path
    if signal == "HOLD":
        return ExecutionQuality(
            slippage_pct=0.0,
            edge_decay=1.0,
            exec_risk="LOW",
            gross_edge_pct=0.0,
            net_edge_pct=0.0,
            viable=False,
            notes=["HOLD — no execution needed"],
        )

    # ── 1. Slippage ───────────────────────────────────────────────────────────
    ask = ask or 0.0
    bid = bid or 0.0
    mid: Optional[float] = None
    if ask > 0 and bid > 0 and ask > bid:
        mid = (ask + bid) / 2
    elif not df.empty:
        mid = float(df["Close"].iloc[-1])

    # Bid-ask spread as a percentage of mid (round-trip = full spread once each way)
    spread_pct = 0.0
    if mid and mid > 0 and ask > 0 and bid > 0 and ask > bid:
        spread_pct = (ask - bid) / mid * 100.0

    # Volatility buffer: quarter-ATR as % of mid, capped at 2× spread
    # Represents extra price movement during order submission
    vol_buffer_pct = 0.0
    if len(df) > atr_periods and mid and mid > 0:
        highs  = df["High"].values[-atr_periods:]
        lows   = df["Low"].values[-atr_periods:]
        closes = df["Close"].values[-atr_periods:]
        trs = [
            max(
                float(highs[i])  - float(lows[i]),
                abs(float(highs[i]) - float(closes[i - 1])),
                abs(float(lows[i])  - float(closes[i - 1])),
            )
            for i in range(1, len(highs))
        ]
        if trs:
            atr = sum(trs) / len(trs)
            # Quarter-ATR represents typical half-range of a candle's worst leg
            raw_buf = (atr / mid) * 25.0
            cap     = max(spread_pct * 2, 0.10)   # never cap below 0.1 %
            vol_buffer_pct = min(raw_buf, cap)

    # Round-trip slippage = spread (entry + exit) + volatility buffer
    slippage_pct = spread_pct + vol_buffer_pct
    notes.append(
        f"Spread {spread_pct:.3f}% + vol buffer {vol_buffer_pct:.3f}% "
        f"= slippage {slippage_pct:.3f}%"
    )

    # ── 2. Edge decay ─────────────────────────────────────────────────────────
    half_life  = HALF_LIVES.get(strategy_key, DEFAULT_HALF_LIFE)
    edge_decay = (
        math.exp(-signal_age_seconds * math.log(2) / half_life)
        if signal_age_seconds > 0
        else 1.0
    )
    if signal_age_seconds > 0:
        notes.append(
            f"Signal age {signal_age_seconds:.0f}s · "
            f"half-life {half_life:.0f}s · "
            f"decay ×{edge_decay:.3f}"
        )

    # ── 3. Execution risk tier ────────────────────────────────────────────────
    if spread_pct > 0.50 or vol_buffer_pct > 0.25:
        exec_risk = "HIGH"
    elif spread_pct > 0.15 or vol_buffer_pct > 0.10:
        exec_risk = "MEDIUM"
    else:
        exec_risk = "LOW"
    notes.append(
        f"Exec risk {exec_risk} "
        f"(spread {spread_pct:.3f}%, vol buf {vol_buffer_pct:.3f}%)"
    )

    # ── 4. Gross edge estimate ────────────────────────────────────────────────
    # Expected value with risk-reward RR_RATIO and stop width BASE_STOP_PCT:
    #   EV = P_win × RR × stop  −  P_lose × stop
    #      = stop × (P_win × RR − (1 − P_win))
    p_win          = (confidence / 100.0) * edge_decay
    p_lose         = 1.0 - p_win
    gross_edge_pct = BASE_STOP_PCT * (p_win * RR_RATIO - p_lose)

    # ── 5. Net edge & viability ───────────────────────────────────────────────
    net_edge_pct = gross_edge_pct - slippage_pct
    notes.append(
        f"Gross edge {gross_edge_pct:.3f}% − slippage {slippage_pct:.3f}% "
        f"= net {net_edge_pct:+.3f}%"
    )

    # Viability rules (ordered — first match wins)
    if exec_risk == "HIGH":
        viable = False
        notes.append("Not viable — HIGH execution risk (spread or volatility too wide)")
    elif net_edge_pct > 0:
        viable = True
    elif confidence >= 75 and exec_risk == "LOW":
        viable = True        # High conviction, tight market: override marginal edge
        notes.append("Viable — high confidence in low-friction market")
    elif confidence >= 85 and exec_risk == "MEDIUM":
        viable = True        # Very high conviction can absorb moderate friction
        notes.append("Viable — very high confidence despite moderate friction")
    else:
        viable = False
        notes.append(
            f"Not viable — net edge {net_edge_pct:+.3f}% insufficient "
            f"for {exec_risk} execution risk"
        )

    return ExecutionQuality(
        slippage_pct=slippage_pct,
        edge_decay=edge_decay,
        exec_risk=exec_risk,
        gross_edge_pct=gross_edge_pct,
        net_edge_pct=net_edge_pct,
        viable=viable,
        notes=notes,
    )
