"""
Visual Bot — REST signal service.
POST /analyse  →  {signal, confidence, reasoning, observations, risk_level, key_level}
"""
import logging
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chart_generator import candles_to_image
from signal_engine import (
    analyse_chart,
    analyse_exit,
    default_spread_pct,
    display_asset_name,
)
import log_buffer

logging.basicConfig(level=logging.INFO)
log_buffer.install(logging.INFO)
log = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o")
LLM_PROVIDER   = os.environ.get("LLM_PROVIDER", "openai")
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL     = os.environ.get("GROQ_MODEL", "llama-3.2-90b-vision-preview")


def _llm_credentials(model_override: Optional[str] = None) -> tuple[str, str, str]:
    if LLM_PROVIDER == "groq":
        return GROQ_API_KEY, model_override or GROQ_MODEL, "groq"
    return OPENAI_API_KEY, model_override or OPENAI_MODEL, "openai"

log.info("Visual-bot starting · provider=%s · model=%s", LLM_PROVIDER, OPENAI_MODEL)

app = FastAPI(title="Visual Bot", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Candle(BaseModel):
    time:   str
    Open:   float
    High:   float
    Low:    float
    Close:  float
    Volume: Optional[float] = None


class AnalyseRequest(BaseModel):
    instrument: str
    interval:   str
    candles:    list[Candle]
    model:      Optional[str] = None   # override OPENAI_MODEL per request
    current_price: Optional[float] = None
    spread_pct: Optional[float] = None
    position_type: str = "NONE"        # NONE | LONG | SHORT
    entry_price: Optional[float] = None
    memory: Optional[str] = None       # inline track-record summary for the LLM
    # Authoritative class from eToro instrumentTypeID (crypto/stock/commodity/
    # etf/index/forex).  Empty = fall back to keyword detection.
    asset_class: Optional[str] = None


class SignalResponse(BaseModel):
    signal:       str          # BUY | SELL | HOLD (mapped from current_signal)
    confidence:   int          # 1–100
    reasoning:    str
    observations: list[str]
    risk_level:   str          # LOW | MEDIUM | HIGH
    key_level:    Optional[float] = None
    current_signal: Optional[str] = None
    expected_direction_next: Optional[str] = None
    nearest_support: Optional[str] = None
    nearest_resistance: Optional[str] = None
    spread_impact: Optional[str] = None
    risk_warning: Optional[str] = None


class OpenPosition(BaseModel):
    direction:      str    # LONG | SHORT
    entry_price:    float
    unrealised_pnl: float  # dollars, live tick mark
    entry_spread:   float
    minutes_open:   float = 0
    current_price:  float = 0
    pnl_pct:        float = 0
    amount_invested: float = 0
    spread_cost:    float = 0
    stop_loss_price: float = 0
    peak_pnl:       float = 0
    in_profit:      bool = False


class ExitAnalyseRequest(BaseModel):
    instrument: str
    interval:   str
    candles:    list[Candle]
    position:   OpenPosition
    model:      Optional[str] = None
    memory:     Optional[str] = None   # inline exit-discipline track record
    asset_class: Optional[str] = None  # authoritative class from eToro metadata


class ExitResponse(BaseModel):
    action:         str          # HOLD | CLOSE
    confidence:     int
    reasoning:      str
    observations:   list[str]
    trend_strength: str          # STRONG | WEAKENING | REVERSING


@app.get("/health")
def health():
    return {"status": "ok", "openai_configured": bool(OPENAI_API_KEY)}


@app.get("/logs")
def logs(limit: int = 1000, level: Optional[str] = None):
    """Recent log records (newest last) for the dashboard's Logs tab."""
    return {"records": log_buffer.get_records(level=level, limit=limit)}


@app.post("/analyse", response_model=SignalResponse)
def analyse(req: AnalyseRequest):
    if len(req.candles) < 5:
        raise HTTPException(400, "Need at least 5 candles to analyse")

    api_key, model, provider = _llm_credentials(req.model)
    if not api_key:
        raise HTTPException(503, f"{provider.upper()} API key not configured")

    df = pd.DataFrame([c.model_dump() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    try:
        asset_name = display_asset_name(req.instrument)
        title     = f"{asset_name}  ·  {req.interval}  ·  {len(df)} candles"
        img_bytes = candles_to_image(
            df, title=title, add_ema20=True, add_ema50=True,
            entry_price=req.entry_price if req.position_type != "NONE" else None,
            direction=req.position_type if req.position_type != "NONE" else None,
        )
    except Exception as exc:
        log.error("Chart render failed: %s", exc)
        raise HTTPException(500, f"Chart render failed: {exc}")

    current_price = req.current_price
    if current_price is None and not df.empty:
        current_price = float(df["Close"].iloc[-1])

    asset_name = display_asset_name(req.instrument)
    spread_pct = req.spread_pct
    if spread_pct is None:
        spread_pct = default_spread_pct(asset_name)

    try:
        result = analyse_chart(
            img_bytes,
            api_key,
            asset=asset_name,
            timeframe=req.interval,
            current_price=current_price,
            spread_pct=spread_pct,
            position_type=req.position_type,
            entry_price=req.entry_price,
            memory=req.memory,
            model=model,
            provider=provider,
            asset_class_hint=req.asset_class,
        )
    except Exception as exc:
        log.error("LLM call failed: %s", exc)
        raise HTTPException(502, f"LLM error: {exc}")

    return SignalResponse(**result)


@app.post("/analyse-exit", response_model=ExitResponse)
def analyse_exit_endpoint(req: ExitAnalyseRequest):
    api_key, model, provider = _llm_credentials(req.model)
    if not api_key:
        raise HTTPException(503, f"{provider.upper()} API key not configured")

    if len(req.candles) < 5:
        raise HTTPException(400, "Need at least 5 candles to analyse")

    df = pd.DataFrame([c.model_dump() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    pos = req.position
    asset_name = display_asset_name(req.instrument)
    title = (
        f"{asset_name}  ·  {req.interval}  ·  "
        f"{pos.direction} @ {pos.entry_price:.5f}  →  {pos.current_price:.5f}  ·  "
        f"P&L ${pos.unrealised_pnl:+,.2f} ({pos.pnl_pct:+.1f}%)"
    )
    try:
        img_bytes = candles_to_image(
            df, title=title, add_ema20=True, add_ema50=True,
            entry_price=pos.entry_price,
            direction=pos.direction,
            minutes_open=pos.minutes_open,
        )
    except Exception as exc:
        log.error("Chart render failed: %s", exc)
        raise HTTPException(500, f"Chart render failed: {exc}")

    try:
        spread_pct = None
        if pos.current_price and pos.entry_spread:
            mid = pos.current_price
            if mid > 0:
                spread_pct = (pos.entry_spread / mid) * 100

        result = analyse_exit(
            img_bytes,
            api_key,
            position=pos.model_dump(),
            asset=asset_name,
            timeframe=req.interval,
            spread_pct=spread_pct,
            memory=req.memory,
            model=model,
            provider=provider,
            asset_class_hint=req.asset_class,
        )
    except Exception as exc:
        log.error("Exit LLM call failed: %s", exc)
        raise HTTPException(502, f"LLM error: {exc}")

    return ExitResponse(**result)
