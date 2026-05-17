from __future__ import annotations

from pathlib import Path

import pandas as pd

from sb.models import TradeRecord


def _classify_session(hour_utc: int) -> str:
    """Klassifiziert UTC-Stunde als Trading-Session."""
    if 7 <= hour_utc < 13:
        return "london"
    if 13 <= hour_utc < 21:
        return "ny"
    return "asia"


def _compute_atr(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """ATR-14 aus OHLCV DataFrame (Spalten case-insensitive)."""
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def _atr_at(atr: pd.Series, ts: pd.Timestamp) -> float:
    """ATR-Wert zum nächstliegenden Zeitpunkt vor ts."""
    idx = atr.index.searchsorted(ts, side="right")
    if idx == 0:
        return float(atr.iloc[0])
    return float(atr.iloc[min(idx - 1, len(atr) - 1)])


def _classify_regime(atr_val: float, p25: float, p75: float) -> str:
    """Klassifiziert ATR-Wert als Marktregime."""
    if atr_val > p75:
        return "volatile"
    if atr_val < p25:
        return "range"
    return "trend"


def enrich_and_save(
    trades: list[TradeRecord],
    run_id: int,
    ohlcv_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """Reichert Trade-Records an und speichert als Parquet.

    Args:
        trades: Rohe TradeRecord-Liste aus SBReplayer.
        run_id: ID des build_runs Eintrags.
        ohlcv_df: OHLCV DataFrame mit DatetimeIndex (UTC).
        output_dir: Ziel-Verzeichnis (z.B. output_v3/).

    Returns:
        Pfad zur gespeicherten Parquet-Datei.
    """
    trades_dir = Path(output_dir) / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    out_path = trades_dir / f"run_{run_id}_trades.parquet"

    if not trades:
        pd.DataFrame(
            columns=[
                "run_id",
                "entry_time",
                "exit_time",
                "direction",
                "entry_price",
                "exit_price",
                "pnl_points",
                "pnl_usd",
                "session",
                "day_of_week",
                "hour_of_day",
                "regime",
            ]
        ).to_parquet(out_path, index=False)
        return out_path

    atr = _compute_atr(ohlcv_df)
    p25 = float(atr.quantile(0.25))
    p75 = float(atr.quantile(0.75))

    rows = []
    for t in trades:
        entry_ts = pd.Timestamp(t.entry_ns, unit="ns", tz="UTC")
        exit_ts = pd.Timestamp(t.exit_ns, unit="ns", tz="UTC")
        hour = entry_ts.hour
        atr_val = _atr_at(atr, entry_ts)

        rows.append(
            {
                "run_id": run_id,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "direction": "long" if t.direction == 1 else "short",
                "entry_price": float(t.entry_price),
                "exit_price": float(t.exit_price),
                "pnl_points": float(t.pnl_pts),
                "pnl_usd": float(t.pnl_pts) * 2.0,  # MNQ: $2/Punkt
                "session": _classify_session(hour),
                "day_of_week": int(entry_ts.dayofweek),
                "hour_of_day": int(hour),
                "regime": _classify_regime(atr_val, p25, p75),
            }
        )

    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return out_path
