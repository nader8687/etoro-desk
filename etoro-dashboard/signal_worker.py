"""
Non-blocking signal requests — runs HTTP calls in background threads so the
live chart fragment never freezes waiting for the LLM.
"""
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

import signal_log
import trade_journal

log = logging.getLogger(__name__)

VISUAL_BOT_URL = os.environ.get("VISUAL_BOT_URL", "http://visual-bot:8080")

_lock = threading.Lock()
_results: dict[str, dict] = {}
_in_flight: set[str] = set()

_exit_results: dict[str, dict] = {}
_exit_in_flight: set[str] = set()


# Results are keyed by (instrument, interval, bot) so multiple bots on the same
# instrument AND interval (e.g. several 15-minute trend bots) never clobber each
# other's signals.  bot_id defaults to "" for legacy callers (one bot per slot).
def _key(instrument_id: int, interval_label: str, bot_id: str = "") -> str:
    return f"{instrument_id}:{interval_label}:{bot_id}"


def _exit_key(instrument_id: int, interval_label: str, bot_id: str = "") -> str:
    return f"exit:{instrument_id}:{interval_label}:{bot_id}"


def _candles_payload(df: pd.DataFrame) -> list[dict]:
    candles = []
    for _, row in df.iterrows():
        c = {
            "time":  row["time"].isoformat(),
            "Open":  float(row["Open"]),
            "High":  float(row["High"]),
            "Low":   float(row["Low"]),
            "Close": float(row["Close"]),
        }
        if "Volume" in df.columns and pd.notna(row.get("Volume")):
            c["Volume"] = float(row["Volume"])
        candles.append(c)
    return candles


def display_asset_name(instrument: str) -> str:
    """eToro labels look like 'Bank of America Corp  (BAC)' — return the name part."""
    text = (instrument or "").strip()
    if not text:
        return "Unknown"
    if "  (" in text and text.endswith(")"):
        return text.rsplit("  (", 1)[0].strip()
    return text


def _spread_pct(ask: float | None, bid: float | None, instrument: str) -> float | None:
    if ask is None or bid is None or ask <= bid:
        return None
    mid = (ask + bid) / 2
    if mid <= 0:
        return None
    return (ask - bid) / mid * 100


def _call_api(
    df: pd.DataFrame,
    instrument: str,
    interval: str,
    *,
    ask: float | None = None,
    bid: float | None = None,
    position: dict | None = None,
    memory: str | None = None,
) -> dict:
    current_price = None
    if ask is not None and bid is not None and ask > bid:
        current_price = (ask + bid) / 2
    elif not df.empty:
        current_price = float(df["Close"].iloc[-1])

    spread_pct = _spread_pct(ask, bid, instrument)
    position_type = "NONE"
    entry_price = None
    if position:
        position_type = str(position.get("direction", "LONG")).upper()
        entry_price = position.get("entry_price")

    import exit_profiles
    payload = {
        "instrument": display_asset_name(instrument),
        "interval": interval,
        "candles": _candles_payload(df),
        "current_price": current_price,
        "spread_pct": spread_pct,
        "position_type": position_type,
        "entry_price": entry_price,
        "memory": memory or "",
        # API-authoritative class (registered at bot start from instrumentTypeID)
        "asset_class": exit_profiles.asset_class(instrument),
    }
    resp = requests.post(f"{VISUAL_BOT_URL}/analyse", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _call_exit_api(
    df: pd.DataFrame,
    instrument: str,
    interval: str,
    position: dict,
    *,
    memory: str | None = None,
) -> dict:
    import exit_profiles
    payload = {
        "instrument": display_asset_name(instrument),
        "interval": interval,
        "candles": _candles_payload(df),
        "position": position,
        "memory": memory or "",
        "asset_class": exit_profiles.asset_class(instrument),
    }
    resp = requests.post(f"{VISUAL_BOT_URL}/analyse-exit", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def request_signal(
    df: pd.DataFrame,
    instrument_id: int,
    instrument_label: str,
    interval_label: str,
    trigger_at: str,
    *,
    force: bool = False,
    ask: float | None = None,
    bid: float | None = None,
    bot_id: str = "",
) -> bool:
    """Fire-and-forget entry signal request."""
    key = _key(instrument_id, interval_label, bot_id)

    with _lock:
        if key in _in_flight and not force:
            return False
        _in_flight.add(key)
        _results[key] = {"_status": "pending", "_at": trigger_at}

    def _worker():
        _start = time.monotonic()
        try:
            # Inline LLM memory: a short, honest summary of this instrument's
            # own win/loss track record (empty until enough history accrues).
            try:
                mem = trade_journal.llm_memory_block(instrument_id, "llm")
            except Exception:
                mem = ""
            result = _call_api(
                df, instrument_label, interval_label, ask=ask, bid=bid, memory=mem,
            )
            result["_status"] = "done"
            result["_at"] = trigger_at
        except requests.exceptions.ConnectionError:
            result = {
                "_error": "Visual Bot not reachable — is it running?",
                "_status": "done",
                "_at": trigger_at,
            }
        except Exception as exc:
            log.error("Signal request failed: %s", exc)
            result = {"_error": str(exc), "_status": "done", "_at": trigger_at}

        # ── Execution quality assessment for LLM results ─────────────────────
        if "_error" not in result:
            try:
                import market_data_hub
                from strategies.execution_quality import assess as _eq_assess
                snap     = market_data_hub.get_snapshot(instrument_id)
                cur_ask  = float(snap.latest_ask) if snap and snap.latest_ask else (ask or 0.0)
                cur_bid  = float(snap.latest_bid) if snap and snap.latest_bid else (bid or 0.0)
                # If LLM says the trade is not profitable after spread, reduce
                # effective confidence so execution quality reflects that view.
                eff_conf = int(result.get("confidence", 0))
                if not result.get("profitable_after_spread", True):
                    eff_conf = max(0, eff_conf - 20)
                elapsed = time.monotonic() - _start
                eq = _eq_assess(
                    df, cur_ask, cur_bid, "llm",
                    confidence=eff_conf,
                    signal=result.get("signal", "HOLD"),
                    signal_age_seconds=elapsed,
                )
                result.update(eq.to_dict())
                log.info(
                    "LLM signal %s conf=%s viable=%s net_edge=%+.3f%% risk=%s elapsed=%.1fs",
                    result.get("signal"), result.get("confidence"),
                    eq.viable, eq.net_edge_pct, eq.exec_risk, elapsed,
                )
            except Exception:
                log.warning("Execution quality assessment failed — defaulting viable=True", exc_info=True)
                result.setdefault("viable", True)

        with _lock:
            _results[key] = result
            _in_flight.discard(key)

        # Persist to signal log (no lock needed — file I/O is serialised inside signal_log)
        if "_error" not in result:
            signal_log.append({
                "ts":                       datetime.now(timezone.utc).isoformat(),
                "type":                     "entry",
                "strategy":                 "llm",
                "bot_id":                   bot_id,
                "instrument_id":            instrument_id,
                "instrument_label":         instrument_label,
                "interval":                 interval_label,
                "trigger_at":               trigger_at,
                "signal":                   result.get("signal"),
                "current_signal":           result.get("current_signal"),
                "confidence":               result.get("confidence"),
                "reasoning":                result.get("reasoning"),
                "risk_warning":             result.get("risk_warning"),
                "spread_impact":            result.get("spread_impact"),
                "observations":             result.get("observations"),
                "expected_direction_next":  result.get("expected_direction_next"),
                "nearest_support":          result.get("nearest_support"),
                "nearest_resistance":       result.get("nearest_resistance"),
                "risk_level":               result.get("risk_level"),
                "profitable_before_spread": result.get("profitable_before_spread"),
                "profitable_after_spread":  result.get("profitable_after_spread"),
                "slippage_pct":             result.get("slippage_pct"),
                "net_edge_pct":             result.get("net_edge_pct"),
                "exec_risk":                result.get("exec_risk"),
                "viable":                   result.get("viable"),
            })

    try:
        threading.Thread(target=_worker, daemon=True, name=f"sig-{key}").start()
    except RuntimeError as exc:
        # Process is out of resources — surface as a soft error and unblock retries
        log.error("Failed to start signal worker thread: %s", exc)
        with _lock:
            _in_flight.discard(key)
            _results[key] = {"_error": "worker spawn failed", "_status": "done", "_at": trigger_at}
        return False
    return True


def request_exit_signal(
    df: pd.DataFrame,
    instrument_id: int,
    instrument_label: str,
    interval_label: str,
    position: dict,
    trigger_at: str,
    bot_id: str = "",
) -> bool:
    """Fire-and-forget exit advisory while a position is open."""
    key = _exit_key(instrument_id, interval_label, bot_id)

    with _lock:
        if key in _exit_in_flight:
            return False
        _exit_in_flight.add(key)
        _exit_results[key] = {"_status": "pending", "_at": trigger_at}

    def _worker():
        try:
            # Inline exit-discipline memory — bounded; empty until enough history.
            try:
                mem = trade_journal.exit_memory_block(instrument_id, "llm")
            except Exception:
                mem = ""
            result = _call_exit_api(
                df, instrument_label, interval_label, position, memory=mem,
            )
            result["_status"] = "done"
            result["_at"] = trigger_at
        except requests.exceptions.ConnectionError:
            result = {
                "_error": "Visual Bot not reachable — is it running?",
                "_status": "done",
                "_at": trigger_at,
            }
        except Exception as exc:
            log.error("Exit signal request failed: %s", exc)
            result = {"_error": str(exc), "_status": "done", "_at": trigger_at}

        with _lock:
            _exit_results[key] = result
            _exit_in_flight.discard(key)

        # Persist to signal log
        if "_error" not in result:
            signal_log.append({
                "ts":             datetime.now(timezone.utc).isoformat(),
                "type":           "exit",
                "bot_id":         bot_id,
                "instrument_id":  instrument_id,
                "instrument_label": instrument_label,
                "interval":       interval_label,
                "trigger_at":     trigger_at,
                "action":         result.get("action"),
                "current_signal": result.get("current_signal"),
                "confidence":     result.get("confidence"),
                "reasoning":      result.get("reasoning"),
                "risk_warning":   result.get("risk_warning"),
                "trend_strength": result.get("trend_strength"),
                "observations":   result.get("observations"),
            })

    try:
        threading.Thread(target=_worker, daemon=True, name=f"exit-{key}").start()
    except RuntimeError as exc:
        log.error("Failed to start exit-signal worker thread: %s", exc)
        with _lock:
            _exit_in_flight.discard(key)
            _exit_results[key] = {"_error": "worker spawn failed", "_status": "done", "_at": trigger_at}
        return False
    return True


def set_result_direct(
    instrument_id: int,
    interval_label: str,
    result: dict,
    instrument_label: str = "",
    trigger_at: str = "",
    bot_id: str = "",
) -> None:
    """Store a synchronous (non-LLM) strategy result directly.

    Writes to the same _results store that LLM results land in, so all
    existing UI code (get_result, is_pending, render_signal) works unchanged.
    """
    key = _key(instrument_id, interval_label, bot_id)
    with _lock:
        _results[key] = result
        _in_flight.discard(key)

    if "_error" not in result:
        signal_log.append({
            "ts":             datetime.now(timezone.utc).isoformat(),
            "type":           "entry",
            "strategy":       result.get("strategy", "unknown"),
            "bot_id":         bot_id,
            "instrument_id":  instrument_id,
            "instrument_label": instrument_label,
            "interval":       interval_label,
            "trigger_at":     trigger_at,
            "signal":         result.get("signal"),
            "confidence":     result.get("confidence"),
            "reasoning":      result.get("reasoning"),
            "risk_level":     result.get("risk_level"),
            "observations":   result.get("observations"),
            "slippage_pct":   result.get("slippage_pct"),
            "net_edge_pct":   result.get("net_edge_pct"),
            "exec_risk":      result.get("exec_risk"),
            "viable":         result.get("viable"),
        })


def get_result(instrument_id: int, interval_label: str, bot_id: str = "") -> Optional[dict]:
    with _lock:
        return _results.get(_key(instrument_id, interval_label, bot_id))


def get_exit_result(instrument_id: int, interval_label: str, bot_id: str = "") -> Optional[dict]:
    with _lock:
        return _exit_results.get(_exit_key(instrument_id, interval_label, bot_id))


def is_pending(instrument_id: int, interval_label: str, bot_id: str = "") -> bool:
    with _lock:
        r = _results.get(_key(instrument_id, interval_label, bot_id))
        return bool(r and r.get("_status") == "pending")


def is_exit_pending(instrument_id: int, interval_label: str, bot_id: str = "") -> bool:
    with _lock:
        r = _exit_results.get(_exit_key(instrument_id, interval_label, bot_id))
        return bool(r and r.get("_status") == "pending")
