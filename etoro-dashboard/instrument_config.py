"""
Loads instruments.toml and exposes the InstrumentSpec list.

Usage:
    from instrument_config import load_specs, resolve_ids
    specs = load_specs()                          # enabled instruments, id=0 means unresolved
    specs = resolve_ids(specs, ALL_INSTRUMENTS)   # fills in instrument_ids from eToro lookup
"""
from __future__ import annotations

import logging
import os
import threading
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# How a bot was added to instruments.toml (shown on Bots tab).
BOT_SOURCE_FLEET = "fleet"
BOT_SOURCE_CUSTOM = "custom"
BOT_SOURCE_BUNDLED = "bundled"
BOT_SOURCE_MANUAL = "manual"
VALID_BOT_SOURCES: frozenset[str] = frozenset({
    BOT_SOURCE_FLEET,
    BOT_SOURCE_CUSTOM,
    BOT_SOURCE_BUNDLED,
    BOT_SOURCE_MANUAL,
})
BOT_SOURCE_LABELS: dict[str, str] = {
    BOT_SOURCE_FLEET: "Fleet top-N",
    BOT_SOURCE_CUSTOM: "Custom builder",
    BOT_SOURCE_BUNDLED: "Bundled default",
    BOT_SOURCE_MANUAL: "Manual / TOML",
}


def created_via_display(source: str) -> str:
    """Human label for a bot's creation source."""
    key = (source or "").strip().lower()
    if not key:
        return "Legacy (unknown)"
    return BOT_SOURCE_LABELS.get(key, key.replace("_", " ").title())


def created_via_badge(source: str) -> str:
    """Short badge for bot cards."""
    key = (source or "").strip().lower()
    icons = {
        BOT_SOURCE_FLEET: "🏁",
        BOT_SOURCE_CUSTOM: "🛠️",
        BOT_SOURCE_BUNDLED: "📦",
        BOT_SOURCE_MANUAL: "✏️",
    }
    icon = icons.get(key, "❓")
    return f"{icon} {created_via_display(source)}"

# Shipped inside the image — used only to seed a fresh data volume once.
BUNDLED_CONFIG_PATH = Path(__file__).parent / "instruments.toml"
# Live config: persisted on the Docker data volume so delete/create survives rebuild.
CONFIG_PATH = Path(
    os.environ.get("INSTRUMENTS_CONFIG_PATH", str(BUNDLED_CONFIG_PATH)),
)

# Cache parsed TOML in memory; invalidate when the file's mtime changes.
# Bots page fragment refreshes every 5 s and otherwise would re-parse on every
# tick — TOML parsing is cheap individually but adds up with many bots.
_cached_specs: list["InstrumentSpec"] = []
_cached_mtime: float = -1.0
_config_lock = threading.RLock()


def _toml_str(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _format_toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_str(value)
    return str(value)


def _instrument_section_header(key: str) -> str:
    """TOML table header for a bot key.

    Keys containing ``.`` must be quoted — otherwise ``[instruments.btc.fut_x]``
    is parsed as nested tables and leaves a broken ``instruments.btc`` parent.
    """
    if "." in key:
        return f"[instruments.{_toml_str(key)}]"
    return f"[instruments.{key}]"


def _flatten_instruments_tree(raw: dict, prefix: str = "") -> dict[str, dict]:
    """Expand legacy nested instrument tables into flat bot keys."""
    flat: dict[str, dict] = {}
    for key, sec in raw.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if not isinstance(sec, dict):
            continue
        if "label" in sec:
            flat[full_key] = sec
        else:
            flat.update(_flatten_instruments_tree(sec, full_key))
    return flat


def _normalize_instrument(key: str, sec: dict) -> dict:
    """Validate and normalize one bot table — raises ValueError if invalid."""
    if not isinstance(sec, dict):
        raise ValueError("must be a table")
    label = sec.get("label")
    if not label or not str(label).strip():
        raise ValueError("missing non-empty label")
    strategy = str(sec.get("strategy") or "").strip()
    if not strategy:
        raise ValueError("missing strategy")
    interval = str(sec.get("interval") or "").strip()
    if not interval:
        raise ValueError("missing interval")
    try:
        interval_secs = int(sec.get("interval_secs", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid interval_secs") from exc
    if interval_secs <= 0:
        raise ValueError("interval_secs must be positive")

    out: dict = {
        "label": str(label).strip(),
        "instrument_id": int(sec.get("instrument_id", 0)),
        "interval": interval,
        "interval_secs": interval_secs,
        "candle_count": int(sec.get("candle_count", 200)),
        "demo_amount": float(sec.get("demo_amount", 1000.0)),
        "enabled": bool(sec.get("enabled", True)),
        "auto_trade": bool(sec.get("auto_trade", False)),
        "strategy": strategy,
    }
    if "trailing_stop_pct" in sec and sec["trailing_stop_pct"] is not None:
        out["trailing_stop_pct"] = float(sec["trailing_stop_pct"])
    if "take_profit_pct" in sec and sec["take_profit_pct"] is not None:
        out["take_profit_pct"] = float(sec["take_profit_pct"])
    raw_via = str(sec.get("created_via") or "").strip().lower()
    if raw_via:
        if raw_via not in VALID_BOT_SOURCES:
            raise ValueError(f"invalid created_via {raw_via!r}")
        out["created_via"] = raw_via
    if "check_in_secs" in sec and sec["check_in_secs"] is not None:
        try:
            ci = int(sec["check_in_secs"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid check_in_secs") from exc
        if ci > 0:
            if ci > interval_secs:
                raise ValueError("check_in_secs cannot exceed interval_secs")
            out["check_in_secs"] = nearest_interval_secs(ci, max_secs=interval_secs)
    return out


def _spec_from_normalized(key: str, sec: dict) -> InstrumentSpec:
    return InstrumentSpec(
        key=key,
        label=sec["label"],
        instrument_id=sec["instrument_id"],
        interval=sec["interval"],
        interval_secs=sec["interval_secs"],
        candle_count=sec["candle_count"],
        demo_amount=sec["demo_amount"],
        enabled=sec["enabled"],
        auto_trade=sec["auto_trade"],
        strategy=sec["strategy"],
        trailing_stop_pct=sec.get("trailing_stop_pct"),
        take_profit_pct=sec.get("take_profit_pct"),
        created_via=str(sec.get("created_via") or ""),
        check_in_secs=int(sec.get("check_in_secs") or 0),
    )


def _serialize_config(data: dict, *, show_empty_marker: bool = False) -> str:
    """Render a complete, valid instruments.toml from structured data."""
    lines = [
        "# ── EtoroDesk bot fleet (instruments.toml) ────────────────────────────",
        "# Written by instrument_config — every [instruments.*] section is complete.",
        "",
    ]
    risk = data.get("risk") or {}
    if risk:
        lines.append("[risk]")
        for rk, val in risk.items():
            lines.append(f"{rk} = {_format_toml_value(val)}")
        lines.append("")

    instruments = data.get("instruments") or {}
    for key in sorted(instruments):
        sec = instruments[key]
        lines.append(_instrument_section_header(key))
        lines.append(f'label              = {_toml_str(sec["label"])}')
        lines.append(f"instrument_id      = {int(sec['instrument_id'])}")
        lines.append(f'interval           = {_toml_str(sec["interval"])}')
        lines.append(f"interval_secs      = {int(sec['interval_secs'])}")
        lines.append(f"candle_count       = {int(sec['candle_count'])}")
        lines.append(f"demo_amount        = {float(sec['demo_amount'])}")
        lines.append(f"enabled            = {_format_toml_value(sec['enabled'])}")
        lines.append(f"auto_trade         = {_format_toml_value(sec['auto_trade'])}")
        lines.append(f'strategy           = {_toml_str(sec["strategy"])}')
        if "trailing_stop_pct" in sec:
            lines.append(f"trailing_stop_pct  = {float(sec['trailing_stop_pct'])}")
        if "take_profit_pct" in sec:
            lines.append(f"take_profit_pct    = {float(sec['take_profit_pct'])}")
        if sec.get("created_via"):
            lines.append(f'created_via        = {_toml_str(sec["created_via"])}')
        if sec.get("check_in_secs"):
            lines.append(f"check_in_secs      = {int(sec['check_in_secs'])}")
        lines.append("")

    if show_empty_marker and not instruments:
        lines.extend([
            "# ── No bots configured ───────────────────────────────────────────────",
            "# Use **Create top N from fleet** on the Bots tab, or add bots via the UI.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write_config(data: dict, *, show_empty_marker: bool = False) -> None:
    """Validate every bot, then atomically replace instruments.toml."""
    validated: dict[str, dict] = {}
    for key, sec in (data.get("instruments") or {}).items():
        validated[key] = _normalize_instrument(key, sec)
    payload = {"risk": data.get("risk") or {}, "instruments": validated}
    text = _serialize_config(payload, show_empty_marker=show_empty_marker)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)
    invalidate_cache()


def _read_config_dict(*, repair: bool = False) -> dict:
    """Load instruments.toml; optionally rewrite the file to drop invalid sections."""
    ensure_config_file()
    with open(CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    risk = raw.get("risk") or {}
    raw_instruments = raw.get("instruments") or {}
    nested = any(
        isinstance(sec, dict) and "label" not in sec
        for sec in raw_instruments.values()
    )
    source = _flatten_instruments_tree(raw_instruments) if nested else raw_instruments

    instruments: dict[str, dict] = {}
    invalid: list[str] = []
    for key, sec in source.items():
        try:
            instruments[key] = _normalize_instrument(key, sec)
        except ValueError as exc:
            invalid.append(key)
            log.error("Invalid instruments.%s in %s: %s", key, CONFIG_PATH, exc)

    metadata_updates = _backfill_fleet_metadata(instruments) if repair else 0

    if invalid or nested or metadata_updates:
        if not repair:
            detail = []
            if nested:
                detail.append("nested tables from unquoted dotted bot keys")
            if invalid:
                detail.append(f"invalid section(s): {', '.join(invalid)}")
            raise ValueError(f"instruments.toml needs repair ({'; '.join(detail)})")
        if nested:
            log.warning(
                "Rewriting instruments.toml — flattened nested bot tables "
                "(dotted keys must use quoted section headers)",
            )
        if invalid:
            log.warning(
                "Rewriting instruments.toml — removed %d invalid section(s): %s",
                len(invalid), ", ".join(invalid),
            )
        if metadata_updates:
            log.info(
                "Rewriting instruments.toml — backfilled fleet metadata for %d bot(s)",
                metadata_updates,
            )
        _atomic_write_config(
            {"risk": risk, "instruments": instruments},
            show_empty_marker=not instruments,
        )
    return {"risk": risk, "instruments": instruments}


def _backfill_fleet_metadata(instruments: dict[str, dict]) -> int:
    """Tag untagged bots that match fleet_opt.json; set fleet check-in and auto-trade ON."""
    try:
        from fleet_picker import load_fleet_rows
    except ImportError:
        return 0

    rows, _ = load_fleet_rows()
    if not rows:
        return 0

    import strategies

    names = strategies.display_names()
    updated = 0
    for key, sec in instruments.items():
        if str(sec.get("created_via") or "").strip():
            continue
        spec = _spec_from_normalized(key, sec)
        row = fleet_row_for_spec(rows, spec, names)
        if row is None:
            continue
        sec["created_via"] = BOT_SOURCE_FLEET
        sec["auto_trade"] = True
        raw_ci = row.get("check_in_secs")
        try:
            ci = int(raw_ci) if raw_ci is not None else 0
        except (TypeError, ValueError):
            ci = 0
        interval_secs = int(sec.get("interval_secs") or 0)
        if ci > 0 and interval_secs > 0:
            ci = nearest_interval_secs(ci, max_secs=interval_secs)
            if ci <= interval_secs:
                sec["check_in_secs"] = ci
        updated += 1
    return updated


def ensure_config_file() -> Path:
    """Ensure instruments.toml exists at CONFIG_PATH (seed from bundle once)."""
    if CONFIG_PATH.resolve() == BUNDLED_CONFIG_PATH.resolve():
        return CONFIG_PATH
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.exists():
        return CONFIG_PATH
    with _config_lock:
        if CONFIG_PATH.exists():
            return CONFIG_PATH
        if BUNDLED_CONFIG_PATH.exists():
            with open(BUNDLED_CONFIG_PATH, "rb") as f:
                bundled = tomllib.load(f)
            seeded: dict[str, dict] = {}
            for key, sec in (bundled.get("instruments") or {}).items():
                if not isinstance(sec, dict):
                    continue
                row = dict(sec)
                if not str(row.get("created_via") or "").strip():
                    row["created_via"] = BOT_SOURCE_BUNDLED
                seeded[key] = row
            _atomic_write_config({
                "risk": bundled.get("risk") or {},
                "instruments": seeded,
            })
            log.info(
                "Seeded instruments.toml on data volume from bundled default (%s)",
                CONFIG_PATH,
            )
        else:
            _atomic_write_config({"risk": {}, "instruments": {}}, show_empty_marker=True)
    return CONFIG_PATH


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
    created_via:        str = ""   # fleet | custom | bundled | manual
    check_in_secs:      int = 0    # exit re-check interval; 0 = same as trade interval


def load_specs(*, enabled_only: bool = True) -> list[InstrumentSpec]:
    """Parse instruments.toml and return InstrumentSpec list.

    Caches the parsed result and re-parses only when the file's mtime changes.
    """
    global _cached_specs, _cached_mtime
    ensure_config_file()
    if not CONFIG_PATH.exists():
        log.warning("instruments.toml not found at %s", CONFIG_PATH)
        return []

    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        mtime = -1.0

    if mtime == _cached_mtime and _cached_specs:
        return [s for s in _cached_specs if (not enabled_only or s.enabled)]

    try:
        with _config_lock:
            data = _read_config_dict(repair=True)
    except Exception as exc:
        log.error("Failed to load instruments.toml at %s: %s", CONFIG_PATH, exc)
        return []

    specs = [
        _spec_from_normalized(key, sec)
        for key, sec in sorted(data["instruments"].items())
    ]
    _cached_specs = specs
    _cached_mtime = mtime
    return [s for s in specs if (not enabled_only or s.enabled)]


def invalidate_cache() -> None:
    """Force the next load_specs() to re-read instruments.toml."""
    global _cached_specs, _cached_mtime
    _cached_specs = []
    _cached_mtime = -1.0


_INTERVAL_KEY_SUFFIX = {
    60: "1m", 300: "5m", 600: "10m", 900: "15m",
    1800: "30m", 3600: "1h", 14400: "4h", 86400: "1d",
}


def interval_secs_from_label(interval_label: str) -> int:
    """Map a human interval label to seconds (defaults to 15 Minutes)."""
    inv = {v: k for k, v in _INTERVAL_LABELS.items()}
    lbl = (interval_label or "").strip()
    if lbl in inv:
        return inv[lbl]
    if lbl.endswith("m") and lbl[:-1].isdigit():
        return int(lbl[:-1]) * 60
    return 900


KNOWN_ASSET_LABELS: dict[str, str] = {
    "Bitcoin": "Bitcoin  (BTC)",
    "XRP": "XRP  (XRP)",
    "Tesla": "Tesla Motors, Inc.  (TSLA)",
    "NVIDIA": "NVIDIA Corporation  (NVDA)",
    "Gold": "Gold  (GOLD)",
    "Solana": "Solana  (SOL)",
    "Ethereum": "Ethereum  (ETH)",
    "Dogecoin": "Dogecoin  (DOGE)",
    "Cardano": "Cardano  (ADA)",
    "Apple": "Apple  (AAPL)",
    "Amazon": "Amazon.com Inc  (AMZN)",
    "Oil": "Oil - Crude  (USOIL)",
}

# Default diversification set for fleet optimization sweeps (must exist on eToro).
FLEET_SWEEP_LABELS: tuple[str, ...] = tuple(KNOWN_ASSET_LABELS.values())
KNOWN_FLEET_LABELS: frozenset[str] = frozenset(KNOWN_ASSET_LABELS.values())


def asset_short_for_label(label: str) -> Optional[str]:
    """Map a full eToro label to our hardcoded fleet asset short name."""
    lbl = (label or "").strip()
    if not lbl:
        return None
    for short, known in KNOWN_ASSET_LABELS.items():
        if known == lbl:
            return short
    return None


def normalize_fleet_asset_short(asset: str, label: str = "") -> Optional[str]:
    """Normalize fleet row Asset/Label to a KNOWN_ASSET_LABELS key, or None."""
    explicit = (label or "").strip()
    if explicit:
        short = asset_short_for_label(explicit)
        if short:
            return short
    short = (asset or "").strip()
    if short in KNOWN_ASSET_LABELS:
        return short
    if short:
        for key, known in KNOWN_ASSET_LABELS.items():
            if short in known or known.split("  (", 1)[0].startswith(short):
                return key
    return None

_etoro_labels_cache: dict[str, int] = {}


def load_etoro_labels(api_key: str, user_key: str) -> dict[str, int]:
    """Fetch label → instrument_id from eToro (cached for this process)."""
    global _etoro_labels_cache
    if _etoro_labels_cache:
        return dict(_etoro_labels_cache)
    try:
        from etoro_client import get_shared_client
        raw = get_shared_client(api_key, user_key).get_instruments()
        opts: dict[str, int] = {}
        for inst in raw.get("instrumentDisplayDatas", []):
            name = inst.get("instrumentDisplayName", "")
            sym = inst.get("symbolFull", "")
            iid = inst.get("instrumentID")
            if iid and name:
                opts[f"{name}  ({sym})" if sym else name] = int(iid)
        _etoro_labels_cache.update(opts)
    except Exception as exc:
        log.warning("Could not load eToro instrument labels: %s", exc)
    return dict(_etoro_labels_cache)


def fleet_sweep_instruments(api_key: str, user_key: str) -> dict[str, int]:
    """Labels for fleet optimization — hardcoded diversification set only."""
    all_labels = load_etoro_labels(api_key, user_key)
    out: dict[str, int] = {}
    missing: list[str] = []
    for lbl in FLEET_SWEEP_LABELS:
        if lbl in all_labels:
            out[lbl] = all_labels[lbl]
        else:
            missing.append(lbl)
    if missing:
        log.warning(
            "Fleet sweep: %d known asset(s) not on eToro: %s",
            len(missing), ", ".join(missing),
        )
    return out


def label_for_asset_name(
    asset_short: str,
    specs: list[InstrumentSpec],
    *,
    full_label: str = "",
    all_labels: Optional[dict[str, int]] = None,
) -> Optional[str]:
    """Resolve a fleet asset to one of our hardcoded KNOWN_ASSET_LABELS only."""
    short = normalize_fleet_asset_short(asset_short, full_label)
    if not short:
        return None

    known = KNOWN_ASSET_LABELS[short]
    explicit = (full_label or "").strip()
    if explicit and explicit in KNOWN_FLEET_LABELS:
        known = explicit

    if all_labels is not None and known not in all_labels:
        log.warning("Known fleet asset %s (%s) not found on eToro", short, known)

    for spec in specs:
        if spec.label == known:
            return known
    return known


def fleet_row_matches_spec(row: dict, spec: InstrumentSpec, names: dict[str, str]) -> bool:
    """True when a saved fleet row matches a bot spec (KNOWN assets only)."""
    if row.get("Status") != "ok":
        return False
    strat_disp = names.get(spec.strategy, spec.strategy)
    if row.get("Strategy") != strat_disp or row.get("Interval") != spec.interval:
        return False
    spec_short = asset_short_for_label(spec.label)
    row_short = normalize_fleet_asset_short(
        (row.get("Asset") or "").strip(),
        (row.get("Label") or "").strip(),
    )
    return spec_short is not None and spec_short == row_short


def fleet_row_for_spec(
    table: list[dict],
    spec: InstrumentSpec,
    names: dict[str, str],
) -> Optional[dict]:
    """Find the fleet optimization row for a configured bot, if any."""
    for row in table:
        if fleet_row_matches_spec(row, spec, names):
            return row
    return None


def suggest_bot_key(
    label: str,
    strategy: str,
    interval_secs: int,
    existing_keys: set[str],
) -> str:
    """Derive a unique instruments.toml section key."""
    if "  (" in label and label.endswith(")"):
        ticker = label.rsplit("  (", 1)[1].rstrip(")").strip().lower()
    else:
        ticker = label.split()[0].lower()[:8]
    iv = _INTERVAL_KEY_SUFFIX.get(interval_secs, f"{interval_secs // 60}m")
    base = f"{ticker}_{strategy}_{iv}"
    key = base
    n = 2
    while key in existing_keys:
        key = f"{base}_{n}"
        n += 1
    return key


def append_bot(
    *,
    key: str,
    label: str,
    strategy: str,
    interval: str,
    interval_secs: int,
    candle_count: int = 200,
    demo_amount: float = 1000.0,
    enabled: bool = True,
    auto_trade: bool = False,
    created_via: str = BOT_SOURCE_MANUAL,
    check_in_secs: int = 0,
) -> InstrumentSpec:
    """Append a new bot section to instruments.toml and return its spec."""
    ensure_config_file()
    via = (created_via or "").strip().lower()
    if via not in VALID_BOT_SOURCES:
        raise ValueError(f"Invalid created_via: {created_via!r}")
    ci = 0
    if check_in_secs and int(check_in_secs) > 0:
        ci = nearest_interval_secs(int(check_in_secs), max_secs=int(interval_secs))
        if ci > int(interval_secs):
            raise ValueError("check_in_secs cannot exceed interval_secs")
    section = {
        "label": label,
        "instrument_id": 0,
        "interval": interval,
        "interval_secs": int(interval_secs),
        "candle_count": int(candle_count),
        "demo_amount": float(demo_amount),
        "enabled": enabled,
        "auto_trade": auto_trade,
        "strategy": strategy,
        "created_via": via,
    }
    if ci > 0:
        section["check_in_secs"] = ci
    with _config_lock:
        data = _read_config_dict(repair=True)
        if key in data["instruments"]:
            raise ValueError(f"Bot key already exists: {key}")
        data["instruments"][key] = _normalize_instrument(key, section)
        _atomic_write_config(data)

    spec = _spec_from_normalized(key, data["instruments"][key])
    log.info(
        "Appended bot %s (%s · %s · %s) [%s]",
        key, label, strategy, interval, via,
    )
    return spec


def clear_all_bots() -> int:
    """Remove every bot section; keep ``[risk]`` and rewrite the file cleanly."""
    ensure_config_file()
    with _config_lock:
        try:
            data = _read_config_dict(repair=True)
        except Exception:
            data = {"risk": {}, "instruments": {}}
        removed = len(data["instruments"])
        data["instruments"] = {}
        _atomic_write_config(data, show_empty_marker=True)
    if removed:
        log.info("Cleared %d bot section(s) from instruments.toml", removed)
    return removed


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


# ── Interval helpers (trade interval + exit check-in interval) ───────────────
# eToro's candle ladder is NOT powers of two, so a derived check-in (½, ¼ of the
# interval) won't always land on a real interval (¼ of 30m = 7.5m doesn't exist).
# Rule: ALWAYS round a target to the NEAREST supported interval.
SUPPORTED_INTERVAL_SECS = (60, 300, 600, 900, 1800, 3600, 14400, 86400)
_INTERVAL_LABELS = {
    60: "1 Minute", 300: "5 Minutes", 600: "10 Minutes", 900: "15 Minutes",
    1800: "30 Minutes", 3600: "1 Hour", 14400: "4 Hours", 86400: "1 Day",
}


def interval_label_for_secs(secs: int) -> str:
    return _INTERVAL_LABELS.get(int(secs), f"{int(secs) // 60}m")


def nearest_interval_secs(target_secs: float, *, max_secs: Optional[int] = None) -> int:
    """Snap an arbitrary target to the NEAREST supported eToro interval.
    Ties resolve to the finer (smaller) interval.  `max_secs` caps the result
    (the exit check-in is never coarser than the bot's own trade interval)."""
    cands = [s for s in SUPPORTED_INTERVAL_SECS if max_secs is None or s <= max_secs]
    if not cands:
        cands = [min(SUPPORTED_INTERVAL_SECS)]
    return min(cands, key=lambda s: (abs(s - target_secs), s))


def check_in_options(interval_secs: int) -> list[int]:
    """Exit check-in candidates for fleet optimization: trade interval, ½, and ¼.

    Each target is snapped to the nearest supported eToro candle (never coarser
    than the trade interval).  Duplicates collapse — e.g. a 1-minute bot only
    has one option (60s).  Returned coarse→fine (largest secs first)."""
    base = int(interval_secs)
    if base <= 0:
        return [60]
    seen: set[int] = set()
    ordered: list[int] = []
    for target in (float(base), base / 2.0, base / 4.0):
        snapped = nearest_interval_secs(target, max_secs=base)
        if snapped not in seen:
            seen.add(snapped)
            ordered.append(snapped)
    return sorted(ordered, reverse=True)


def effective_check_in_secs(
    bot_key: str,
    interval_secs: int,
    *,
    toml_check_in: int = 0,
) -> int:
    """The bot's exit check-in interval in seconds.

    Priority: Settings ``bot_overrides`` (fleet-optimized) → instruments.toml
    ``check_in_secs`` → trade interval.
    """
    base = int(interval_secs)
    try:
        import user_settings
        raw = user_settings.load().get("bot_overrides", {}).get(bot_key, {})
        ci = raw.get("check_in_secs") if isinstance(raw, dict) else None
        if ci and int(ci) > 0:
            return nearest_interval_secs(float(ci), max_secs=base)
    except Exception:
        pass
    if toml_check_in and int(toml_check_in) > 0:
        return nearest_interval_secs(float(toml_check_in), max_secs=base)
    return base
