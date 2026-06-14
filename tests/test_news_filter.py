from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest


def _has_calendar():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "economic_calendar.csv")
    return os.path.exists(path)


_needs_calendar = pytest.mark.skipif(not _has_calendar(), reason="economic_calendar.csv not available")


@_needs_calendar
def test_nfp_window_blocked():
    """NFP 08:30 ET = 13:30 UTC (Winter). 5-Min-Fenster: 13:25-13:35."""
    from sb.filters.news_filter import is_news_window

    dt = datetime(2025, 1, 10, 13, 30, 0, tzinfo=timezone.utc)
    assert is_news_window(dt) is True


@_needs_calendar
def test_outside_window_allowed():
    """30 Minuten nach NFP – soll durchgelassen werden."""
    from sb.filters.news_filter import is_news_window

    dt = datetime(2025, 1, 10, 14, 5, 0, tzinfo=timezone.utc)
    assert is_news_window(dt) is False


@_needs_calendar
def test_fomc_window_blocked():
    """FOMC 14:00 ET = 19:00 UTC (Winter)."""
    from sb.filters.news_filter import is_news_window

    dt = datetime(2025, 1, 29, 19, 1, 0, tzinfo=timezone.utc)
    assert is_news_window(dt) is True


@_needs_calendar
def test_custom_window():
    """Groesseres Fenster (10 Min) blockt mehr."""
    from sb.filters.news_filter import is_news_window

    dt = datetime(2025, 1, 10, 13, 38, 0, tzinfo=timezone.utc)
    assert is_news_window(dt, window_minutes=10) is True
    assert is_news_window(dt, window_minutes=5) is False


@_needs_calendar
def test_no_event_day_allowed():
    """Normaler Dienstag ohne Event."""
    from sb.filters.news_filter import is_news_window

    dt = datetime(2025, 1, 7, 14, 30, 0, tzinfo=timezone.utc)
    assert is_news_window(dt) is False
