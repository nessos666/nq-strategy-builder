from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from sb.inspect import _OHLCV_COLS

_REQUIRED_PERFORMANCE_COLUMNS = {
    "idea",
    "avg_oos_pf",
    "holdout_pf",
    "tier",
    "is_robust",
}


def _find_performance_table(connection: sqlite3.Connection) -> str:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    for (table_name,) in rows:
        columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if _REQUIRED_PERFORMANCE_COLUMNS <= columns:
            return str(table_name)
    raise ValueError(
        "Keine Tabelle mit den Spalten idea, avg_oos_pf, holdout_pf, tier, is_robust gefunden."
    )


def _safe_pct(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.mean()) * 100.0, 1)


def _summarize_performance_group(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n": 0,
            "avg_oos_pf": 0.0,
            "avg_holdout_pf": 0.0,
            "pct_tier_a": 0.0,
            "pct_tier_b": 0.0,
            "pct_robust": 0.0,
        }

    tier_upper = frame["tier"].astype(str).str.upper().fillna("")
    robust_numeric = pd.to_numeric(frame["is_robust"], errors="coerce").fillna(0)
    holdout_mean = pd.to_numeric(frame["holdout_pf"], errors="coerce").mean()
    if pd.isna(holdout_mean):
        holdout_mean = 0.0
    return {
        "n": int(len(frame)),
        "avg_oos_pf": round(float(frame["avg_oos_pf"].mean()), 3),
        "avg_holdout_pf": round(float(holdout_mean), 3),
        "pct_tier_a": _safe_pct((tier_upper == "A").astype(float)),
        "pct_tier_b": _safe_pct((tier_upper == "B").astype(float)),
        "pct_robust": _safe_pct((robust_numeric > 0).astype(float)),
    }


def analyze_zeit_performance(algo_name_fragment: str, db_path: str) -> dict:
    db_file = Path(db_path).expanduser()
    if not db_file.exists():
        raise FileNotFoundError(f"Datenbank nicht gefunden: {db_file}")

    fragment = algo_name_fragment.strip()
    if not fragment:
        raise ValueError("algo_name_fragment darf nicht leer sein.")
    search_token = fragment.split()[0].upper()

    connection = sqlite3.connect(str(db_file))
    try:
        table_name = _find_performance_table(connection)
        frame = pd.read_sql_query(
            (
                f"SELECT idea, avg_oos_pf, holdout_pf, tier, is_robust, "
                f"CASE WHEN UPPER(COALESCE(idea, '')) LIKE ? THEN 1 ELSE 0 END AS is_match "
                f"FROM {table_name}"
            ),
            connection,
            params=[f"%{search_token}%"],
        )
    finally:
        connection.close()

    for col in ("avg_oos_pf", "holdout_pf"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["avg_oos_pf"])

    match_mask = frame["is_match"].fillna(0).astype(bool)
    return {
        "mit": _summarize_performance_group(frame[match_mask].copy()),
        "ohne": _summarize_performance_group(frame[~match_mask].copy()),
    }


def _detect_signal_columns(
    df: pd.DataFrame,
    original_cols: set[str] | None = None,
) -> list[str]:
    signal_cols: list[str] = []
    excluded_cols = _OHLCV_COLS | (original_cols or set())
    for col in df.columns:
        if col in excluded_cols:
            continue
        series = df[col]
        if series.dtype == bool or str(series.dtype) == "boolean":
            signal_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            values = set(series.dropna().unique())
            if values and values <= {0, 1, 0.0, 1.0, True, False}:
                signal_cols.append(col)
    return signal_cols


def _resolve_ohlc_columns(df: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for base_name in ("open", "high", "low", "close"):
        for candidate in (base_name, base_name.title()):
            if candidate in df.columns:
                resolved[base_name] = candidate
                break
        else:
            raise ValueError(f"DataFrame fehlt Spalte '{base_name}'")
    return resolved


def _format_hour_label(hour: int, window_bars: int) -> str:
    half_window = window_bars // 2
    start_total = (hour * 60 - half_window) % (24 * 60)
    end_total = (hour * 60 + half_window) % (24 * 60)
    start_hour, start_minute = divmod(start_total, 60)
    end_hour, end_minute = divmod(end_total, 60)
    return f"{start_hour:02d}:{start_minute:02d}–{end_hour:02d}:{end_minute:02d}"


def analyze_zeit_fenster(
    df: pd.DataFrame,
    signal_col: str,
    window_bars: int = 20,
) -> pd.DataFrame:
    if window_bars <= 0:
        raise ValueError("window_bars muss größer als 0 sein")
    if signal_col not in df.columns:
        raise ValueError(f"Signalspalte nicht gefunden: {signal_col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame muss einen DatetimeIndex haben")

    ohlc_cols = _resolve_ohlc_columns(df)
    signal = df[signal_col].fillna(False).astype(bool)
    if signal.empty:
        return pd.DataFrame(
            columns=[
                "hour_label",
                "count",
                "avg_range",
                "up_pct",
                "down_pct",
                "avg_net_move",
            ]
        )

    starts = signal & ~signal.shift(1, fill_value=False)
    ends = ~signal & signal.shift(1, fill_value=False)
    start_positions = list(df.index.get_indexer(df.index[starts]))
    end_positions = list(df.index.get_indexer(df.index[ends]))
    if signal.iloc[-1]:
        end_positions.append(len(df))

    groups: dict[int, dict[str, object]] = {}
    for start_pos, end_pos in zip(start_positions, end_positions):
        if end_pos <= start_pos:
            continue
        window_df = df.iloc[start_pos:end_pos]
        if window_df.empty:
            continue

        start_ts = window_df.index[0]
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        start_hour = start_ts.tz_convert("America/New_York").hour

        bucket = groups.setdefault(
            start_hour,
            {"count": 0, "ranges": [], "up": 0, "down": 0, "bars": 0, "net_moves": []},
        )

        open_series = pd.to_numeric(window_df[ohlc_cols["open"]], errors="coerce")
        high_series = pd.to_numeric(window_df[ohlc_cols["high"]], errors="coerce")
        low_series = pd.to_numeric(window_df[ohlc_cols["low"]], errors="coerce")
        close_series = pd.to_numeric(window_df[ohlc_cols["close"]], errors="coerce")
        range_series = (high_series - low_series).dropna()

        bucket["count"] = int(bucket["count"]) + 1
        bucket["ranges"].extend(range_series.tolist())  # type: ignore[union-attr]
        bucket["up"] = int(bucket["up"]) + int((close_series > open_series).sum())
        bucket["down"] = int(bucket["down"]) + int((close_series < open_series).sum())
        bucket["bars"] = int(bucket["bars"]) + int((open_series.notna() & close_series.notna()).sum())
        net_move = close_series.iloc[-1] - open_series.iloc[0]
        if pd.notna(net_move):
            bucket["net_moves"].append(float(net_move))  # type: ignore[union-attr]

    rows: list[dict[str, float | int | str]] = []
    for hour in sorted(groups):
        bucket = groups[hour]
        ranges = pd.Series(bucket["ranges"], dtype=float)
        net_moves = pd.Series(bucket["net_moves"], dtype=float)
        bar_count = int(bucket["bars"])
        rows.append(
            {
                "hour_label": _format_hour_label(hour, window_bars),
                "count": int(bucket["count"]),
                "avg_range": round(float(ranges.mean()) if not ranges.empty else 0.0, 3),
                "up_pct": round((int(bucket["up"]) / bar_count * 100.0) if bar_count else 0.0, 1),
                "down_pct": round((int(bucket["down"]) / bar_count * 100.0) if bar_count else 0.0, 1),
                "avg_net_move": round(
                    float(net_moves.mean()) if not net_moves.empty else 0.0, 3
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "hour_label",
            "count",
            "avg_range",
            "up_pct",
            "down_pct",
            "avg_net_move",
        ],
    )


def analyze_zeit_phasen(
    df: pd.DataFrame,
    algo_name: str,
    window: int = 30,
    original_cols: set[str] | None = None,
) -> pd.DataFrame:
    _ = algo_name
    if window <= 0:
        raise ValueError("window muss größer als 0 sein")
    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"DataFrame fehlt Spalte '{col}'")

    signal_cols = _detect_signal_columns(df, original_cols=original_cols)
    if not signal_cols:
        raise ValueError("Keine Bool-/0-1-Signalspalten erkannt.")

    signal_state = df[signal_cols].fillna(0).astype(bool).any(axis=1).to_numpy()
    n_rows = len(df)
    phases = ["neutral"] * n_rows

    for idx, is_active in enumerate(signal_state):
        if is_active:
            phases[idx] = "während"

    for idx in range(1, n_rows):
        if signal_state[idx] and not signal_state[idx - 1]:
            start = max(0, idx - window)
            for pos in range(start, idx):
                if phases[pos] == "neutral":
                    phases[pos] = "vor"

    for idx in range(1, n_rows):
        if not signal_state[idx] and signal_state[idx - 1]:
            end = min(n_rows, idx + window)
            for pos in range(idx, end):
                if phases[pos] == "neutral":
                    phases[pos] = "nach"

    frame = df.copy()
    frame["phase"] = phases
    frame["range"] = pd.to_numeric(frame["High"], errors="coerce") - pd.to_numeric(
        frame["Low"], errors="coerce"
    )
    frame["up"] = (
        pd.to_numeric(frame["Close"], errors="coerce")
        > pd.to_numeric(frame["Open"], errors="coerce")
    ).astype(float)
    frame["down"] = (
        pd.to_numeric(frame["Close"], errors="coerce")
        < pd.to_numeric(frame["Open"], errors="coerce")
    ).astype(float)

    phase_order = ["während", "vor", "nach", "neutral"]
    grouped = (
        frame.groupby("phase", dropna=False)
        .agg(
            count=("phase", "size"),
            avg_range=("range", "mean"),
            up_pct=("up", "mean"),
            down_pct=("down", "mean"),
        )
        .reindex(phase_order, fill_value=0)
        .reset_index()
    )
    grouped["avg_range"] = (
        pd.to_numeric(grouped["avg_range"], errors="coerce").fillna(0).round(3)
    )
    grouped["up_pct"] = (
        pd.to_numeric(grouped["up_pct"], errors="coerce").fillna(0) * 100
    ).round(1)
    grouped["down_pct"] = (
        pd.to_numeric(grouped["down_pct"], errors="coerce").fillna(0) * 100
    ).round(1)
    grouped["count"] = grouped["count"].astype(int)
    return grouped[["phase", "count", "avg_range", "up_pct", "down_pct"]]
