from __future__ import annotations

from pathlib import Path

import pandas as pd

from sb.models import BacktestResult


class BacktestBridge:
    """Inline-Simulation: lädt Parquet einmal, simuliert Trades für jeden Parameter-Satz."""

    _REQUIRED_COLUMNS = {"high", "low", "close"}

    def __init__(self, data_path: Path) -> None:
        self.data_path = Path(data_path)
        self._df: pd.DataFrame | None = None
        self._load_data()

    def _load_data(self) -> None:
        if not self.data_path.exists():
            self._df = None
            return
        try:
            df = pd.read_parquet(self.data_path)
            # Spalten normalisieren: Open/High/Low/Close → open/high/low/close
            df.columns = [c.lower() for c in df.columns]
            if not self._REQUIRED_COLUMNS.issubset(df.columns):
                self._df = None
                return
            self._df = df
        except Exception:
            self._df = None

    def _point_scale(self) -> float:
        if self._df is None or self._df.empty:
            return 1.0
        close_diff = self._df["close"].diff().dropna()
        if close_diff.empty:
            return 1.0
        # Normalisiert Punkt-Abstände auf die Volatilität der geladenen Daten.
        return max(float(close_diff.std()) * 0.1, 0.1)

    def run(self, params: dict) -> BacktestResult:
        if self._df is None or self._df.empty:
            return BacktestResult(
                params=params,
                gross_profit=0.0,
                gross_loss=0.0,
                num_trades=0,
                num_wins=0,
            )

        sl_points = max(float(params.get("sl_points", 10.0)), 0.1)
        tp_mult = max(float(params.get("tp_mult", 2.5)), 0.1)
        offset = max(int(params.get("entry_bar_offset", 1)), 0)
        interval: int = max(int(params.get("signal_interval_bars", 30)), 5)
        scale = self._point_scale()
        sl_distance = sl_points * scale
        tp_distance = max(tp_mult * 5.0 * scale, sl_distance * 0.25)
        df = self._df

        wins = losses = 0
        gross_profit = gross_loss = 0.0
        equity = peak = max_dd = 0.0

        for sig_idx in range(0, len(df) - offset - 50, interval):
            entry_idx = sig_idx + offset
            if entry_idx >= len(df):
                break
            entry_price = float(df.iloc[entry_idx]["close"])
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            hit = "none"
            for bar in df.iloc[entry_idx + 1 : entry_idx + 200].itertuples():
                if bar.low <= sl_price:
                    hit = "sl"
                    break
                if bar.high >= tp_price:
                    hit = "tp"
                    break
            if hit == "tp":
                wins += 1
                gross_profit += tp_distance
                equity += tp_distance
            elif hit == "sl":
                losses += 1
                gross_loss += sl_distance
                equity -= sl_distance
            peak = max(peak, equity)
            dd = max(peak - equity, 0.0)
            max_dd = max(max_dd, dd)

        return BacktestResult(
            params=params,
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            num_trades=wins + losses,
            num_wins=wins,
            max_drawdown=round(max_dd, 2),
        )
