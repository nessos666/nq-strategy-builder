from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from sb.cache.signal_cache import SignalCache, SignalCacheConfig


@pytest.fixture
def tiny_df():
    """Minimaler DataFrame mit 200 Bars (NQ OHLCV)."""
    n = 200
    rng = np.random.default_rng(42)
    close = 19000.0 + np.cumsum(rng.normal(0, 5, n))
    df = pd.DataFrame(
        {
            "Open": close + rng.uniform(-2, 2, n),
            "High": close + rng.uniform(0, 10, n),
            "Low": close - rng.uniform(0, 10, n),
            "Close": close,
            "Volume": rng.integers(100, 1000, n).astype(float),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
    )
    return df


def test_signal_cache_builds(tmp_path, tiny_df):
    """Cache baut fehlerfrei durch – Manifest-Datei wird angelegt."""
    parquet_in = tmp_path / "bars.parquet"
    tiny_df.to_parquet(parquet_in)
    cache_out = tmp_path / "signals.parquet"

    cfg = SignalCacheConfig(
        bars_path=parquet_in,
        cache_path=cache_out,
        algo_dirs=[],  # leere dirs → kein Algo → trotzdem läuft durch
    )
    cache = SignalCache(cfg)
    cache.build()

    # Manifest-Datei wurde geschrieben
    assert cache_out.exists()
    # load() gibt leeren DataFrame zurück (keine Algos, keine Shards)
    df = cache.load()
    assert isinstance(df, pd.DataFrame)


def test_signal_cache_idempotent(tmp_path, tiny_df):
    """Zweiter build()-Aufruf mit force=True überschreibt Cache ohne Fehler."""
    parquet_in = tmp_path / "bars.parquet"
    tiny_df.to_parquet(parquet_in)
    cache_out = tmp_path / "signals.parquet"
    cfg = SignalCacheConfig(bars_path=parquet_in, cache_path=cache_out, algo_dirs=[])
    cache = SignalCache(cfg)
    cache.build()
    mtime1 = cache_out.stat().st_mtime
    import time

    time.sleep(0.05)
    cache.build(force=True)
    mtime2 = cache_out.stat().st_mtime
    assert mtime2 >= mtime1


def test_signal_cache_skips_if_fresh(tmp_path, tiny_df):
    """Ohne force=True wird nicht neu gebaut wenn Cache existiert."""
    parquet_in = tmp_path / "bars.parquet"
    tiny_df.to_parquet(parquet_in)
    cache_out = tmp_path / "signals.parquet"
    cfg = SignalCacheConfig(bars_path=parquet_in, cache_path=cache_out, algo_dirs=[])
    cache = SignalCache(cfg)
    cache.build()
    mtime1 = cache_out.stat().st_mtime
    cache.build()  # kein force → kein Rebuild
    mtime2 = cache_out.stat().st_mtime
    assert mtime1 == mtime2


def test_signal_cache_with_real_algo(tmp_path, tiny_df):
    """Cache läuft mit echtem BOS-Algo und produziert boolean-Spalten."""
    parquet_in = tmp_path / "bars.parquet"
    tiny_df.to_parquet(parquet_in)
    cache_out = tmp_path / "signals.parquet"

    import os

    pda_dir = (
        Path(os.environ.get("TRADINGPROJEKT_PATH", str(Path.home() / "TRADINGPROJEKT")))
        / "nq_backtest/algo_bibliothek/v2/pda"
    )
    if not pda_dir.exists():
        pytest.skip("PDA v2 Bibliothek nicht gefunden")

    cfg = SignalCacheConfig(
        bars_path=parquet_in,
        cache_path=cache_out,
        algo_dirs=[pda_dir],
    )
    cache = SignalCache(cfg)
    # BOS-Konzept → nur BOS-Shard bauen
    cache.build(concepts=["BOS"])
    df = cache.load(concepts=["BOS"])
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    assert len(bool_cols) > 0, "Mindestens eine boolean Signal-Spalte erwartet"
