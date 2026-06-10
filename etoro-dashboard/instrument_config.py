"""
Loads instruments.toml and exposes the InstrumentSpec list.

Usage:
    from instrument_config import load_specs, resolve_ids
    specs = load_specs()                          # enabled instruments, id=0 means unresolved
    specs = resolve_ids(specs, ALL_INSTRUMENTS)   # fills in instrument_ids from eToro lookup
"""
from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "instruments.toml"

# Cache parsed TOML in memory; invalidate when the file's mtime changes.
# Bots page fragment refreshes every 5 s and otherwise would re-parse on every
# tick — TOML parsing is cheap individually but adds up with many bots.
_cached_specs: list["InstrumentSpec"] = []
_cached_mtime: float = -1.0


@dataclass(frozen=True)
class InstrumentSpec:
    key:           str    # config section key, e.g. "xrp"
    label:         str    # eToro display label, e.g. "XRP  (XRP)"
    instrument_id: int    # 0 = unresolved; call resolve_ids() to fill in
    interval:      str    # "1 Minute"
    interval_secs: int    # 60
    candle_count:  int    # candles in chart / LLM context
    demo_amount:   float  # dollars per demo trade
    enabled:            bool
    auto_trade:         bool   # initial value; Streamlit can override at runtime
    strategy:           str   = "llm"  # strategy key; Bots page can override at runtime
    # Exit params default to the strategy's profile (exit_profiles.py) when unset.
    # Set them in instruments.toml only to OVERRIDE a specific bot.
    trailing_stop_pct:  Optional[float] = None   # % pullback from peak (None = use profile)
    take_profit_pct:    Optional[float] = None   # hard take-profit % (None = use profile)


def load_specs(*, enabled_only: bool = True) -> list[InstrumentSpec]:
    """Parse instruments.toml and return InstrumentSpec list.

    Caches the parsed result and re-parses only when the file's mtime changes.
    """
    global _cached_specs, _cached_mtime
    if not CONFIG_PATH.exists():
        log.warning("instruments.toml not found at %s", CONFIG_PATH)
        return []

    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = -1.0

    if mtime == _cached_mtime and _cached_specs:
        return [s for s in _cached_specs if (not enabled_only or s.enabled)]

    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    specs: list[InstrumentSpec] = []
    for key, sec in data.get("instruments", {}).items():
        spec = InstrumentSpec(
            key=key,
            label=sec["label"],
            instrument_id=int(sec.get("instrument_id", 0)),
            interval=sec.get("interval", "1 Minute"),
            interval_secs=int(sec.get("interval_secs", 60)),
            candle_count=int(sec.get("candle_count", 100)),
            demo_amount=float(sec.get("demo_amount", 1000.0)),
            enabled=bool(sec.get("enabled", True)),
            auto_trade=bool(sec.get("auto_trade", False)),
            strategy=str(sec.get("strategy", "llm")),
            trailing_stop_pct=(
                float(sec["trailing_stop_pct"]) if "trailing_stop_pct" in sec else None
            ),
            take_profit_pct=(
                float(sec["take_profit_pct"]) if "take_profit_pct" in sec else None
            ),
        )
        specs.append(spec)
    _cached_specs = specs
    _cached_mtime = mtime
    return [s for s in specs if (not enabled_only or s.enabled)]


def resolve_ids(
    specs: list[InstrumentSpec],
    all_instruments: dict[str, int],
) -> list[InstrumentSpec]:
    """
    Replace instrument_id=0 with real IDs looked up from the eToro instruments dict.

    Matching strategy (in order):
      1. Exact label match
      2. Case-insensitive partial match (label substring or superstring)

    Instruments whose ID cannot be resolved are dropped with a warning.
    """
    result: list[InstrumentSpec] = []
    for spec in specs:
        if spec.instrument_id != 0:
            result.append(spec)
            continue

        # 1. Exact
        iid = all_instruments.get(spec.label)

        # 2. Partial (case-insensitive)
        if not iid:
            label_lower = spec.label.lower()
            for lbl, candidate in all_instruments.items():
                if label_lower in lbl.lower() or lbl.lower() in label_lower:
                    iid = candidate
                    log.info(
                        "Resolved %r via partial match → %r (id=%d)",
                        spec.label, lbl, candidate,
                    )
                    break

        if iid:
            result.append(replace(spec, instrument_id=iid))
        else:
            log.warning(
                "Could not resolve instrument_id for %r — skipping. "
                "Set instrument_id explicitly in instruments.toml.",
                spec.label,
            )
    return result
