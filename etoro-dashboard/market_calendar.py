"""
US equity market session calendar — proactive market-hours awareness.

Answers "is the US stock market open right now?" from RULES, not from data
freshness: weekends, the full NYSE/NASDAQ holiday set (computed for any year —
no hardcoded tables to go stale), observed-date shifts (Sat→Fri, Sun→Mon), and
1:00 pm ET half-day closes.

Used by EToroClient.is_market_open as the FIRST check for US-listed stocks:
the calendar knows a holiday in advance (proactive), while the rate-freshness
heuristic remains as confirmation for anything the rules can't know
(trading halts, one-off closures such as a national day of mourning).

Regular session: 09:30–16:00 America/New_York (DST handled by zoneinfo).
Half days close at 13:00 ET: day after Thanksgiving, Christmas Eve (weekday),
and July 3rd (when both Jul 3 and Jul 4 are weekdays).
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

OPEN_T       = time(9, 30)
CLOSE_T      = time(16, 0)
HALF_CLOSE_T = time(13, 0)


def _easter(year: int) -> date:
    """Anonymous Gregorian computus — Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th (1-based) given weekday (Mon=0) of a month."""
    d = date(year, month, 1)
    offset = (weekday - d.weekday()) % 7
    return d + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """NYSE observed date: Saturday→Friday before, Sunday→Monday after."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def us_market_holidays(year: int) -> set[date]:
    """Full-close NYSE/NASDAQ holidays for *year* (observed dates)."""
    easter = _easter(year)
    return {
        _observed(date(year, 1, 1)),            # New Year's Day
        _nth_weekday(year, 1, 0, 3),            # MLK Day — 3rd Mon Jan
        _nth_weekday(year, 2, 0, 3),            # Washington's Birthday — 3rd Mon Feb
        easter - timedelta(days=2),             # Good Friday
        _last_weekday(year, 5, 0),              # Memorial Day — last Mon May
        _observed(date(year, 6, 19)),           # Juneteenth
        _observed(date(year, 7, 4)),            # Independence Day
        _nth_weekday(year, 9, 0, 1),            # Labor Day — 1st Mon Sep
        _nth_weekday(year, 11, 3, 4),           # Thanksgiving — 4th Thu Nov
        _observed(date(year, 12, 25)),          # Christmas
    }


def us_half_days(year: int) -> set[date]:
    """1:00 pm ET early-close days."""
    half: set[date] = set()
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    half.add(thanksgiving + timedelta(days=1))            # Black Friday
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5 and _observed(date(year, 12, 25)) != xmas_eve:
        half.add(xmas_eve)                                # Christmas Eve (weekday)
    jul3, jul4 = date(year, 7, 3), date(year, 7, 4)
    if jul3.weekday() < 5 and jul4.weekday() < 5:
        half.add(jul3)                                    # July 3rd (both weekdays)
    return half


def us_session_state(now: datetime | None = None) -> str:
    """'open' | 'closed' for the US equity regular session at *now* (UTC ok)."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ny = now.astimezone(NY)
    d = ny.date()
    if ny.weekday() >= 5:                       # weekend
        return "closed"
    if d in us_market_holidays(d.year):         # full-close holiday
        return "closed"
    close_t = HALF_CLOSE_T if d in us_half_days(d.year) else CLOSE_T
    return "open" if OPEN_T <= ny.time() < close_t else "closed"


def is_us_equity_open(now: datetime | None = None) -> bool:
    return us_session_state(now) == "open"
