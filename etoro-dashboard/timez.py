"""Display-timezone helper.

Everything stored, journaled and computed stays in **UTC** — only what the user
*sees* is converted.  The Trading page exposes a timezone picker; the chosen
zone is held in a module global that the main Streamlit script refreshes once
per rerun (rendering is single-threaded), so presentation helpers in `ui` and
`views.tables` can localise times without importing streamlit or reading
session_state.
"""
from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - py<3.9 fallback (shouldn't happen on 3.11)
    ZoneInfo = None  # type: ignore

DEFAULT_TZ = "UTC"

# Curated, practical shortlist for the selector.  Any IANA name also works if
# set programmatically, but these cover the common trading hubs.
COMMON_ZONES = [
    "UTC",
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "America/Sao_Paulo", "Europe/London", "Europe/Paris", "Europe/Berlin",
    "Europe/Zurich", "Europe/Moscow", "Africa/Johannesburg",
    "Asia/Dubai", "Asia/Tehran", "Asia/Karachi", "Asia/Kolkata", "Asia/Dhaka",
    "Asia/Singapore", "Asia/Hong_Kong", "Asia/Shanghai", "Asia/Tokyo",
    "Australia/Sydney", "Pacific/Auckland",
]

_active_name: str = DEFAULT_TZ
_active_tz: tzinfo = timezone.utc


def set_active(name: Optional[str]) -> None:
    """Set the active display zone by IANA name (falls back to UTC if invalid)."""
    global _active_name, _active_tz
    name = (name or DEFAULT_TZ).strip() or DEFAULT_TZ
    if name == "UTC" or ZoneInfo is None:
        _active_tz, _active_name = timezone.utc, "UTC"
        return
    try:
        _active_tz = ZoneInfo(name)
        _active_name = name
    except Exception:
        _active_tz, _active_name = timezone.utc, "UTC"


def active_name() -> str:
    return _active_name


def active_tz() -> tzinfo:
    return _active_tz


def abbrev() -> str:
    """Short current abbreviation for the active zone, e.g. 'UTC', 'GST', 'EDT'."""
    try:
        return datetime.now(_active_tz).strftime("%Z") or _active_name
    except Exception:
        return _active_name


def to_local(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime (naive ones are assumed UTC) to the active zone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_active_tz)


def fmt(dt: Optional[datetime], fmt_str: str = "%Y-%m-%d %H:%M") -> str:
    loc = to_local(dt)
    return loc.strftime(fmt_str) if loc else "—"


def fmt_iso(value, fmt_str: str = "%Y-%m-%d %H:%M") -> str:
    """Format an ISO string (or datetime) in the active zone."""
    if not value:
        return "—"
    if isinstance(value, datetime):
        return fmt(value, fmt_str)
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return str(value)
    return fmt(dt, fmt_str)


def now_str(fmt_str: str = "%H:%M:%S") -> str:
    """Current wall-clock in the active zone."""
    return datetime.now(_active_tz).strftime(fmt_str)


def to_local_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """Active-zone wall-clock with tzinfo stripped — for plotting libraries that
    reinterpret tz-aware values against the browser timezone."""
    loc = to_local(dt)
    return loc.replace(tzinfo=None) if loc else None


def localize_series(series):
    """Convert a pandas datetime Series (UTC) to naive wall-clock in the active
    zone, so a plot's time axis reads in local time literally."""
    import pandas as pd

    dt = pd.to_datetime(series, utc=True, errors="coerce")
    try:
        return dt.dt.tz_convert(_active_tz).dt.tz_localize(None)
    except Exception:
        return dt.dt.tz_localize(None)
