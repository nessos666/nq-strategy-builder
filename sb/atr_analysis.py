"""
ATR-Analyse – Descriptive Stats + Intraday-Profil für Float-Algos.

Verwendung:
    from sb.atr_analysis import analyze_atr_stats

    result = analyze_atr_stats(result_df, algo_name="1. ATR Standard")
    # result["stats"]   -> DataFrame mit mean/std/p25/p50/p75/p90/p95 pro Spalte
    # result["hourly"]  -> DataFrame mit NY-Stunde x Spalte (avg Wert)
    # result["columns"] -> Liste der analysierten Spalten
"""

from __future__ import annotations

import pandas as pd


_OHLCV_COLS: frozenset[str] = frozenset(
    {"open", "high", "low", "close", "volume", "Open", "High", "Low", "Close", "Volume"}
)


def _to_ny(index: pd.Index) -> pd.Series:
    s = pd.Series(pd.DatetimeIndex(index))
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    else:
        s = s.dt.tz_convert("US/Eastern")
    return s


def _detect_float_cols(df: pd.DataFrame) -> list[str]:
    """Gibt alle Float-Ausgabe-Spalten zurück (nicht OHLCV, nicht bool)."""
    cols = []
    for col in df.columns:
        if col in _OHLCV_COLS:
            continue
        if df[col].dtype == bool or (
            hasattr(df[col], "dtype") and str(df[col].dtype) == "bool"
        ):
            continue
        if pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(
            df[col]
        ):
            # Ignoriere Integer-Spalten die nur 0/1 sind (Bool-Surrogate)
            if pd.api.types.is_integer_dtype(df[col]):
                unique = df[col].dropna().unique()
                if set(unique).issubset({0, 1, True, False}):
                    continue
            cols.append(col)
    return cols


def analyze_atr_stats(
    result_df: pd.DataFrame,
    algo_name: str = "",
    cols: list[str] | None = None,
) -> dict:
    """Berechnet Descriptive Stats + Stunden-Profil für ATR-artige Float-Spalten.

    Args:
        result_df: Output des Algos (mit DatetimeIndex UTC).
        algo_name: Nur für Metadaten.
        cols: Optional: explizite Spalten. Wenn None → auto-detect.

    Returns:
        dict mit:
            "columns": list[str]
            "stats":   pd.DataFrame – Zeilen = Spalten, Spalten = Statistik
            "hourly":  pd.DataFrame – Zeilen = "HH:MM", Spalten = ATR-Spalten
            "n_bars":  int
            "date_from": str
            "date_to":   str
    """
    if cols is None:
        cols = _detect_float_cols(result_df)

    if not cols:
        raise ValueError(f"Keine Float-Ausgabe-Spalten gefunden für {algo_name!r}.")

    # ── Descriptive Stats ────────────────────────────────────────────────────────
    stats_rows = []
    for col in cols:
        s = result_df[col].dropna()
        if len(s) == 0:
            continue
        stats_rows.append(
            {
                "Spalte": col,
                "N": len(s),
                "Mean": s.mean(),
                "Std": s.std(),
                "p25": s.quantile(0.25),
                "p50": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "p90": s.quantile(0.90),
                "p95": s.quantile(0.95),
            }
        )
    stats_df = (
        pd.DataFrame(stats_rows).set_index("Spalte") if stats_rows else pd.DataFrame()
    )

    # ── Stunden-Profil (NY, 30-min Fenster) ──────────────────────────────────────
    ny = _to_ny(result_df.index)
    hour_label = (ny.dt.hour * 100 + (ny.dt.minute // 30) * 30).apply(
        lambda x: f"{x // 100:02d}:{x % 100:02d}"
    )
    hour_label.index = result_df.index

    hourly_rows: dict[str, dict[str, float]] = {}
    for col in cols:
        s = result_df[col].dropna()
        if len(s) == 0:
            continue
        grp = s.groupby(hour_label.loc[s.index]).mean()
        for lbl, val in grp.items():
            lbl_str = str(lbl)
            if lbl_str not in hourly_rows:
                hourly_rows[lbl_str] = {}
            hourly_rows[lbl_str][col] = float(val)

    hourly_df = (
        pd.DataFrame(hourly_rows).T.sort_index() if hourly_rows else pd.DataFrame()
    )

    return {
        "columns": cols,
        "stats": stats_df,
        "hourly": hourly_df,
        "n_bars": len(result_df),
        "date_from": str(result_df.index[0])[:10],
        "date_to": str(result_df.index[-1])[:10],
    }
