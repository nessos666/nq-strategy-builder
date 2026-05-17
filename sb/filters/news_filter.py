"""
news_filter.py
==============
Filtert Timestamps die innerhalb eines High-Impact News-Fensters liegen.

Verwendung im Backtest:
    from sb.filters.news_filter import is_news_window
    if is_news_window(bar_timestamp_utc):
        continue  # Bar ueberspringen

Standalone-Test:
    .venv/bin/python -c "
    from datetime import datetime, timezone
    from sb.filters.news_filter import is_news_window
    dt = datetime(2025, 1, 10, 13, 30, tzinfo=timezone.utc)
    print(is_news_window(dt))
    "
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "economic_calendar.csv"


def _et_offset(date: datetime) -> int:
    """Gibt UTC-Offset in Stunden: -5 (Winter/EST) oder -4 (Sommer/EDT)."""
    year = date.year
    # DST start: 2. Sonntag Maerz
    mar1 = datetime(year, 3, 1)
    dst_start = mar1 + timedelta(days=(6 - mar1.weekday()) % 7 + 7)
    # DST end: 1. Sonntag November
    nov1 = datetime(year, 11, 1)
    dst_end = nov1 + timedelta(days=(6 - nov1.weekday()) % 7)
    d = date.replace(tzinfo=None)
    if dst_start <= d < dst_end:
        return -4  # EDT
    return -5  # EST


@lru_cache(maxsize=1)
def _load_events() -> tuple[datetime, ...]:
    """Laedt alle HIGH-Impact Events als UTC-datetimes (gecacht).

    Gibt leeres Tuple zurueck wenn CSV fehlt oder unlesbar ist.
    """
    if not _CSV_PATH.exists():
        import warnings

        warnings.warn(
            f"economic_calendar.csv nicht gefunden: {_CSV_PATH}. "
            "News-Filter deaktiviert (alle Timestamps durchgelassen).",
            RuntimeWarning,
            stacklevel=3,
        )
        return ()
    events: list[datetime] = []
    try:
        with open(_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("impact") != "HIGH":
                    continue
                local_dt = datetime.strptime(
                    f"{row['date']} {row['time_et']}", "%Y-%m-%d %H:%M"
                )
                offset_h = _et_offset(local_dt)
                utc_dt = local_dt - timedelta(hours=offset_h)
                events.append(utc_dt.replace(tzinfo=timezone.utc))
    except Exception as exc:
        import warnings

        warnings.warn(
            f"Fehler beim Laden der economic_calendar.csv: {exc}. "
            "News-Filter deaktiviert.",
            RuntimeWarning,
            stacklevel=3,
        )
        return ()
    return tuple(events)


def is_news_window(dt_utc: datetime, window_minutes: int = 5) -> bool:
    """
    Gibt True zurueck wenn dt_utc innerhalb eines High-Impact News-Fensters liegt.

    Args:
        dt_utc: Timestamp in UTC (timezone-aware oder naive, wird als UTC behandelt)
        window_minutes: Fenster vor und nach dem Event in Minuten

    Returns:
        True wenn geblockt, False wenn frei
    """
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    delta = timedelta(minutes=window_minutes)
    return any(abs(dt_utc - ev) <= delta for ev in _load_events())


def get_blocked_dates(
    start: datetime, end: datetime, window_minutes: int = 5
) -> list[datetime]:
    """Gibt alle Event-Zeitpunkte (UTC) im Zeitraum zurueck."""
    delta = timedelta(minutes=window_minutes)
    return [ev for ev in _load_events() if start - delta <= ev <= end + delta]
