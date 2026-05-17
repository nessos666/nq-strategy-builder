from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _make_ohlcv(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(seed)
    close = 19000 + np.cumsum(rng.standard_normal(n) * 0.5)
    wick = np.abs(rng.standard_normal(n)) * 2 + 1.0
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


def _make_trades_parquet(tmp_path: Path, run_id: int, n: int = 40) -> Path:
    dates = pd.date_range("2024-06-01 14:00", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(run_id)
    df = pd.DataFrame(
        {
            "run_id": run_id,
            "entry_time": dates,
            "exit_time": dates + pd.Timedelta(minutes=15),
            "direction": rng.choice(["long", "short"], n),
            "entry_price": 19000.0,
            "exit_price": 19000.0 + rng.standard_normal(n) * 10,
            "pnl_points": rng.standard_normal(n) * 5,
            "pnl_usd": rng.standard_normal(n) * 10,
            "session": "ny",
            "day_of_week": dates.dayofweek,
            "hour_of_day": dates.hour,
            "regime": rng.choice(["trend", "range", "volatile"], n),
        }
    )
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir(exist_ok=True)
    path = trades_dir / f"run_{run_id}_trades.parquet"
    df.to_parquet(path, index=False)
    return path


def test_regime_shift_returns_dict():
    from sb.diagnose import analyse_regime_shift

    ohlcv = _make_ohlcv(n=10_000)
    result = analyse_regime_shift(ohlcv)
    assert isinstance(result, dict)
    assert "quarters" in result
    assert "shift_detected" in result


def test_regime_shift_has_atr_per_quarter():
    from sb.diagnose import analyse_regime_shift

    ohlcv = _make_ohlcv(n=10_000)
    result = analyse_regime_shift(ohlcv)
    quarters = result["quarters"]
    assert len(quarters) > 0
    first = quarters[0]
    assert "period" in first
    assert "atr_mean" in first
    assert "atr_std" in first


def test_overfitting_diagnose_returns_list(tmp_path):
    import sqlite3

    from sb.diagnose import analyse_overfitting

    db_path = tmp_path / "builder.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE build_runs (
        id INTEGER PRIMARY KEY, idea TEXT, avg_oos_pf REAL,
        tier TEXT, holdout_pf REAL, holdout_trades INTEGER,
        holdout_validated INTEGER, is_robust INTEGER, pbo_score REAL,
        mc_pct_profitable REAL, trials INTEGER, session TEXT, created_at TEXT
    )"""
    )
    conn.execute(
        """INSERT INTO build_runs VALUES
        (1,'INSIDE_DAY NY',2.7,'A',0.6,29,1,0,NULL,NULL,50,'ny','2026-01-01')"""
    )
    conn.execute(
        """INSERT INTO build_runs VALUES
        (2,'BOS + FVG NY',1.5,'B',1.2,45,1,0,NULL,NULL,50,'ny','2026-01-01')"""
    )
    conn.commit()
    conn.close()
    result = analyse_overfitting(db_path, tier="A")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "run_id" in result[0]
    assert "classification" in result[0]
    assert "degradation" in result[0]


def test_baustein_quality_returns_dict(tmp_path):
    import sqlite3

    from sb.diagnose import analyse_bausteine

    n = 500
    dates = pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(0)
    cache_df = pd.DataFrame(
        {
            "inside_day": rng.choice([0, 1], n, p=[0.9, 0.1]).astype(float),
            "fvg_bullish": rng.choice([0, 1], n, p=[0.8, 0.2]).astype(float),
            "hurst_value": np.zeros(n),
        },
        index=dates,
    )
    cache_path = tmp_path / "signal_cache.parquet"
    cache_df.to_parquet(cache_path)

    db_path = tmp_path / "builder.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE build_runs (
        id INTEGER PRIMARY KEY, idea TEXT, avg_oos_pf REAL, tier TEXT,
        holdout_pf REAL, holdout_trades INTEGER, holdout_validated INTEGER,
        is_robust INTEGER, pbo_score REAL, mc_pct_profitable REAL,
        trials INTEGER, session TEXT, created_at TEXT
    )"""
    )
    conn.execute(
        """INSERT INTO build_runs VALUES
        (1,'INSIDE_DAY NY',2.7,'A',0.6,29,1,0,NULL,NULL,50,'ny','2026-01-01')"""
    )
    conn.commit()
    conn.close()

    result = analyse_bausteine(cache_path=cache_path, db_path=db_path)
    assert isinstance(result, dict)
    assert "bausteine" in result
    assert len(result["bausteine"]) > 0
    assert "dead_bausteine" in result


def test_trade_distribution_returns_dict(tmp_path):
    from sb.diagnose import analyse_trade_distribution

    _make_trades_parquet(tmp_path, run_id=1)
    _make_trades_parquet(tmp_path, run_id=2)
    result = analyse_trade_distribution(output_dir=tmp_path)
    assert isinstance(result, dict)
    assert "by_session" in result
    assert "by_hour_weekday" in result
    assert "by_regime" in result
