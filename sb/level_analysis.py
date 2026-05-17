from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sb.inspect import _to_ny_series

_WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
_SESSION_ORDER = ["Overnight", "London", "NY Open", "Lunch", "PM"]
_MULTI_TOUCH_ORDER = ["1st", "2nd", "3rd+"]
_CONFLUENCE_ORDER = ["Nearby", "Alone"]
_DIRECTION_ORDER = ["From Below", "From Above"]


@dataclass
class _TouchEvent:
    level_col: str
    episode_start: int
    episode_end: int
    bar_index: int
    timestamp: pd.Timestamp
    level_value: float
    bounce: bool
    start_above: bool


def _prev_level_cols(df: pd.DataFrame, level_cols: list[str]) -> list[str]:
    return [col for col in level_cols if col in df.columns and col.startswith("prev_")]


def _validate_price_df(df: pd.DataFrame) -> None:
    for col in ("High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame fehlt Spalte '{col}'")


def _iter_touch_events(
    df: pd.DataFrame,
    level_cols: list[str],
    threshold: float,
) -> list[_TouchEvent]:
    _validate_price_df(df)
    _ = threshold

    cols = _prev_level_cols(df, level_cols)
    if not cols:
        return []

    hi = pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float)
    ny = _to_ny_series(df.index)

    events: list[_TouchEvent] = []
    n = len(df)
    for level_col in cols:
        level = pd.to_numeric(df[level_col], errors="coerce").to_numpy(dtype=float)
        i = 0
        while i < n:
            lv = level[i]
            if np.isnan(lv):
                i += 1
                continue

            j = i
            while j < n and level[j] == lv:
                j += 1

            for k in range(i, j):
                if np.isnan(hi[k]) or np.isnan(lo[k]) or np.isnan(close[k]):
                    continue
                if lo[k] <= lv <= hi[k]:
                    if k > i:
                        start_above = close[k - 1] > lv
                    else:
                        start_above = lo[k] >= lv
                    bounce = close[k] > lv if start_above else close[k] < lv
                    events.append(
                        _TouchEvent(
                            level_col=level_col,
                            episode_start=i,
                            episode_end=j,
                            bar_index=k,
                            timestamp=ny.iloc[k],
                            level_value=float(lv),
                            bounce=bool(bounce),
                            start_above=bool(start_above),
                        )
                    )
                    break
            i = j

    return events


def _summarize(
    rows: list[dict],
    order: list[str],
    include_avg_distance: bool = False,
) -> pd.DataFrame:
    columns = ["level_col", "group", "touches", "bounce", "through", "bounce_pct", "through_pct"]
    if include_avg_distance:
        columns.append("avg_distance")
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(["level_col", "group"], dropna=False)
        .agg(
            touches=("bounce", "size"),
            bounce=("bounce", "sum"),
            avg_distance=("distance", "mean") if include_avg_distance else ("bounce", "size"),
        )
        .reset_index()
    )
    if not include_avg_distance:
        grouped = grouped.drop(columns=["avg_distance"])
    grouped["bounce"] = grouped["bounce"].astype(int)
    grouped["touches"] = grouped["touches"].astype(int)
    grouped["through"] = grouped["touches"] - grouped["bounce"]
    grouped["bounce_pct"] = ((grouped["bounce"] / grouped["touches"]) * 100).round(1)
    grouped["through_pct"] = ((grouped["through"] / grouped["touches"]) * 100).round(1)
    if include_avg_distance:
        grouped["avg_distance"] = pd.to_numeric(grouped["avg_distance"], errors="coerce").round(2)

    order_map = {name: idx for idx, name in enumerate(order)}
    grouped["_sort"] = grouped["group"].map(order_map).fillna(len(order)).astype(int)
    grouped = grouped.sort_values(["level_col", "_sort", "group"]).drop(columns="_sort")
    return grouped[columns]


def _session_name(ts: pd.Timestamp) -> str | None:
    minutes = ts.hour * 60 + ts.minute
    if minutes >= 18 * 60 or minutes < 3 * 60:
        return "Overnight"
    if minutes < 9 * 60 + 30:
        return "London"
    if minutes < 11 * 60:
        return "NY Open"
    if minutes < 13 * 60:
        return "Lunch"
    if minutes < 16 * 60:
        return "PM"
    return None


def _trading_weekday(ts: pd.Timestamp) -> int:
    weekday = ts.weekday()
    if ts.hour >= 18:
        return (weekday + 1) % 7
    return weekday


def analyze_level_multi_touch(
    df: pd.DataFrame,
    level_cols: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    touch_counts: dict[str, dict[float, int]] = {}
    rows = []
    for event in _iter_touch_events(df, level_cols, threshold):
        level_counts = touch_counts.setdefault(event.level_col, {})
        touch_number = level_counts.get(event.level_value, 0) + 1
        level_counts[event.level_value] = touch_number
        group = "1st"
        if touch_number == 2:
            group = "2nd"
        elif touch_number >= 3:
            group = "3rd+"
        rows.append({"level_col": event.level_col, "group": group, "bounce": event.bounce})
    return _summarize(rows, _MULTI_TOUCH_ORDER)


def analyze_level_weekday(
    df: pd.DataFrame,
    level_cols: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    rows = []
    for event in _iter_touch_events(df, level_cols, threshold):
        weekday = _WEEKDAY_NAMES.get(_trading_weekday(event.timestamp))
        if weekday is None:
            continue
        rows.append({"level_col": event.level_col, "group": weekday, "bounce": event.bounce})
    return _summarize(rows, list(_WEEKDAY_NAMES.values()))


def analyze_level_session(
    df: pd.DataFrame,
    level_cols: list[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    rows = []
    for event in _iter_touch_events(df, level_cols, threshold):
        session = _session_name(event.timestamp)
        if session is None:
            continue
        rows.append({"level_col": event.level_col, "group": session, "bounce": event.bounce})
    return _summarize(rows, _SESSION_ORDER)


def analyze_level_confluence(
    df: pd.DataFrame,
    level_cols: list[str],
    proximity: float = 10.0,
    threshold: float = 3.0,
) -> pd.DataFrame:
    prev_cols = _prev_level_cols(df, level_cols)
    rows = []
    for event in _iter_touch_events(df, prev_cols, threshold):
        distances = []
        for other_col in prev_cols:
            if other_col == event.level_col:
                continue
            other_value = pd.to_numeric(df.iloc[event.bar_index][other_col], errors="coerce")
            if pd.isna(other_value):
                continue
            distance = abs(float(other_value) - event.level_value)
            if distance <= proximity:
                distances.append(distance)
        if distances:
            rows.append(
                {
                    "level_col": event.level_col,
                    "group": "Nearby",
                    "bounce": event.bounce,
                    "distance": min(distances),
                }
            )
        else:
            rows.append(
                {
                    "level_col": event.level_col,
                    "group": "Alone",
                    "bounce": event.bounce,
                    "distance": None,
                }
            )
    return _summarize(rows, _CONFLUENCE_ORDER, include_avg_distance=True)


def analyze_level_direction(
    df: pd.DataFrame,
    level_cols: list[str],
    lookback: int = 10,
    threshold: float = 3.0,
) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback muss größer als 0 sein")
    _ = lookback

    rows = []
    for event in _iter_touch_events(df, level_cols, threshold):
        group = "From Above" if event.start_above else "From Below"
        rows.append({"level_col": event.level_col, "group": group, "bounce": event.bounce})
    return _summarize(rows, _DIRECTION_ORDER)
