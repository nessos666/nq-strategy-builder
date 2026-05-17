from __future__ import annotations

import numpy as np
import pandas as pd

from sb.models import TradeRecord


def _make_ohlcv(n: int = 2000) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 19000 + np.cumsum(rng.standard_normal(n))
    wick = np.abs(rng.standard_normal(n)) * 2 + 0.5
    return pd.DataFrame(
        {
            "open": close + rng.standard_normal(n),
            "high": close + wick,
            "low": close - wick,
            "close": close,
            "volume": rng.integers(100, 500, n),
        },
        index=dates,
    )


def _make_trade(
    entry_ts: pd.Timestamp, direction: int = 1, pnl: float = 10.0
) -> TradeRecord:
    return TradeRecord(
        entry_ns=int(entry_ts.value),
        exit_ns=int(entry_ts.value) + 60_000_000_000,
        direction=direction,
        entry_price=19000.0,
        exit_price=19020.0 if pnl > 0 else 18990.0,
        pnl_pts=pnl,
    )


def test_enrich_and_save_creates_parquet(tmp_path):
    from sb.trade_export import enrich_and_save

    ohlcv = _make_ohlcv()
    trades = [
        _make_trade(pd.Timestamp("2024-01-02 14:30:00", tz="UTC"), pnl=10.0),
        _make_trade(pd.Timestamp("2024-01-02 15:00:00", tz="UTC"), pnl=-5.0),
    ]
    path = enrich_and_save(trades, run_id=42, ohlcv_df=ohlcv, output_dir=tmp_path)
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert "session" in df.columns
    assert "regime" in df.columns
    assert "pnl_usd" in df.columns
    assert df["run_id"].iloc[0] == 42


def test_enrich_and_save_empty_trades(tmp_path):
    from sb.trade_export import enrich_and_save

    ohlcv = _make_ohlcv()
    path = enrich_and_save([], run_id=99, ohlcv_df=ohlcv, output_dir=tmp_path)
    assert path is not None


def test_classify_session():
    from sb.trade_export import _classify_session

    assert _classify_session(8) == "london"
    assert _classify_session(14) == "ny"
    assert _classify_session(2) == "asia"


def test_ny_trade_has_ny_session(tmp_path):
    from sb.trade_export import enrich_and_save

    ohlcv = _make_ohlcv()
    trades = [_make_trade(pd.Timestamp("2024-01-02 14:30:00", tz="UTC"))]
    path = enrich_and_save(trades, run_id=1, ohlcv_df=ohlcv, output_dir=tmp_path)
    df = pd.read_parquet(path)
    assert df["session"].iloc[0] == "ny"


def test_regime_column_values_valid(tmp_path):
    from sb.trade_export import enrich_and_save

    ohlcv = _make_ohlcv()
    trades = [
        _make_trade(pd.Timestamp("2024-01-02 14:30:00", tz="UTC")),
        _make_trade(pd.Timestamp("2024-01-02 09:00:00", tz="UTC")),
    ]
    path = enrich_and_save(trades, run_id=2, ohlcv_df=ohlcv, output_dir=tmp_path)
    df = pd.read_parquet(path)
    assert df["regime"].isin(["trend", "range", "volatile"]).all()
