from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sb.cache.signal_cache import SignalCache, SignalCacheConfig
from sb.engine.nautilus_bridge import (
    COMMISSION_USD,
    POINT_VALUE,
    SLIPPAGE_POINTS,
    NautilusBridge,
)
from sb.models import BacktestResult


@pytest.fixture
def sample_parquet(tmp_path):
    """500 Bars NQ-Daten."""
    n = 500
    rng = np.random.default_rng(1)
    close = 19000.0 + np.cumsum(rng.normal(0, 5, n))
    df = pd.DataFrame(
        {
            "Open": close + rng.uniform(-2, 2, n),
            "High": close + rng.uniform(0, 10, n),
            "Low": close - rng.uniform(0, 10, n),
            "Close": close,
            "Volume": rng.integers(100, 1000, n).astype(float),
        },
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC"),
    )
    p = tmp_path / "nq_test.parquet"
    df.to_parquet(p)
    return p


@pytest.fixture
def signal_cache(tmp_path, sample_parquet):
    """Leerer Signal-Cache (keine echten Algos)."""
    cache_p = tmp_path / "signals.parquet"
    cfg = SignalCacheConfig(
        bars_path=sample_parquet,
        cache_path=cache_p,
        algo_dirs=[],
    )
    SignalCache(cfg).build()
    return cache_p


def test_bridge_returns_result(sample_parquet, signal_cache):
    bridge = NautilusBridge(data_path=sample_parquet, cache_path=signal_cache)
    result = bridge.run(
        {
            "sl_points": 10.0,
            "tp_mult": 2.5,
            "entry_bar_offset": 0,
            "concepts": [],
            "direction": 1,
        }
    )
    assert isinstance(result, BacktestResult)
    assert result.num_trades >= 0


def test_bridge_missing_data_returns_zero(tmp_path):
    bridge = NautilusBridge(
        data_path=tmp_path / "missing.parquet",
        cache_path=tmp_path / "missing_cache.parquet",
    )
    result = bridge.run(
        {
            "sl_points": 10.0,
            "tp_mult": 2.5,
            "entry_bar_offset": 0,
            "concepts": [],
            "direction": 1,
        }
    )
    assert result.num_trades == 0


def test_session_filter_london_excludes_asia_timestamps():
    """London-Session darf keine Timestamps aus Asia-Zeit enthalten."""
    from sb.engine.nautilus_bridge import _filter_by_session
    import pandas as pd

    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-15 08:00", tz="UTC"),  # London ✓
            pd.Timestamp("2024-01-15 02:00", tz="UTC"),  # Asia ✗
            pd.Timestamp("2024-01-15 14:00", tz="UTC"),  # NY ✗
        ]
    )
    result = _filter_by_session(idx, "london")
    assert len(result) == 1
    assert result[0].hour == 8  # type: ignore[union-attr]


def test_session_filter_all_returns_all():
    """Session 'all' gibt alle Timestamps zurück."""
    from sb.engine.nautilus_bridge import _filter_by_session
    import pandas as pd

    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-15 08:00", tz="UTC"),
            pd.Timestamp("2024-01-15 02:00", tz="UTC"),
            pd.Timestamp("2024-01-15 14:00", tz="UTC"),
        ]
    )
    result = _filter_by_session(idx, "all")
    assert len(result) == 3


def test_from_bars_creates_bridge_without_file_loading():
    from sb.engine.nautilus_bridge import NautilusBridge
    from pathlib import Path
    from unittest.mock import patch

    # Verify that from_bars() bypasses _load() completely
    with patch.object(NautilusBridge, "_load") as mock_load:
        bridge = NautilusBridge.from_bars(
            bars=[],
            cache_df=None,
            instrument=None,
            bar_type=None,
            venue=None,
        )
        mock_load.assert_not_called()
    assert bridge._bars == []
    assert bridge._cache_df is None
    assert bridge._instrument is None
    assert bridge.data_path == Path(".")


def test_bridge_uses_role_aware_query_when_entry_present(monkeypatch):
    """Wenn params['entry'] gesetzt ist, wird query_signal_bars_roles() verwendet."""
    import pandas as pd
    from sb.engine.nautilus_bridge import NautilusBridge
    import sb.engine.nautilus_bridge as bridge_module

    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    cache_df = pd.DataFrame(
        {
            "bos_bullish": [0, 1, 0, 0, 0],
            "fvg_bull_active": [0, 1, 0, 0, 0],
        },
        index=idx,
    )

    roles_called = {}

    def fake_roles(df, entry, zone, context, timing):
        roles_called["entry"] = entry
        roles_called["zone"] = zone
        return pd.Index([])

    monkeypatch.setattr(bridge_module, "query_signal_bars_roles", fake_roles)
    monkeypatch.setattr(NautilusBridge, "_build_signals", lambda *a, **k: [])

    bridge = object.__new__(NautilusBridge)
    bridge._bars = [object()]  # nicht leer damit query aufgerufen wird
    bridge._cache_df = cache_df
    bridge._instrument = object()

    params = {
        "sl_points": 10.0,
        "tp_mult": 2.0,
        "entry_bar_offset": 1,
        "session": "ny",
        "concepts": ["BOS", "FVG"],
        "entry": ["BOS"],
        "zone": ["FVG"],
        "context": [],
        "timing": [],
        "direction": 1,
    }
    bridge.run(params)
    assert roles_called.get("entry") == ["BOS"]
    assert roles_called.get("zone") == ["FVG"]


def test_bridge_falls_back_to_legacy_when_no_entry(monkeypatch):
    """Wenn params['entry'] leer ist, wird query_signal_bars() (legacy) verwendet."""
    import pandas as pd
    from sb.engine.nautilus_bridge import NautilusBridge
    import sb.engine.nautilus_bridge as bridge_module

    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    cache_df = pd.DataFrame({"bos_bullish": [0, 1, 0, 0, 0]}, index=idx)

    legacy_called = {}

    def fake_legacy(df, concepts, mode="any"):
        legacy_called["concepts"] = concepts
        return pd.Index([])

    monkeypatch.setattr(bridge_module, "query_signal_bars", fake_legacy)
    monkeypatch.setattr(NautilusBridge, "_build_signals", lambda *a, **k: [])

    bridge = object.__new__(NautilusBridge)
    bridge._bars = [object()]  # nicht leer
    bridge._cache_df = cache_df
    bridge._instrument = object()

    params = {
        "sl_points": 10.0,
        "tp_mult": 2.0,
        "entry_bar_offset": 1,
        "session": "ny",
        "concepts": ["BOS"],
        "entry": [],
        "zone": [],
        "context": [],
        "timing": [],
        "direction": 1,
    }
    bridge.run(params)
    assert legacy_called.get("concepts") == ["BOS"]


def test_nautilus_bridge_max_date_cuts_bars(tmp_path):
    """max_date filtert Bars die >= dieses Datum sind raus."""
    import pandas as pd

    dates = pd.to_datetime(
        ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"], utc=True
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "volume": [100, 100, 100, 100],
        },
        index=dates,
    )
    parquet_path = tmp_path / "bars.parquet"
    df.to_parquet(parquet_path)
    cache_path = tmp_path / "cache.parquet"
    bridge = NautilusBridge(
        data_path=parquet_path,
        cache_path=cache_path,
        max_date="2025-07-01",
    )
    assert len(bridge._bars) == 2


def test_nautilus_bridge_min_date_keeps_bars_from(tmp_path):
    """min_date filtert Bars die < dieses Datum sind raus."""
    import pandas as pd

    dates = pd.to_datetime(
        ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"], utc=True
    )
    df = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.5] * 4,
            "volume": [100] * 4,
        },
        index=dates,
    )
    parquet_path = tmp_path / "bars.parquet"
    df.to_parquet(parquet_path)
    cache_path = tmp_path / "cache.parquet"
    bridge = NautilusBridge(
        data_path=parquet_path,
        cache_path=cache_path,
        min_date="2025-07-01",
    )
    assert len(bridge._bars) == 2


def test_nautilus_bridge_no_date_filter_keeps_all(tmp_path):
    """Ohne date-Filter bleiben alle Bars."""
    import pandas as pd

    dates = pd.to_datetime(
        ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"], utc=True
    )
    df = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.5] * 4,
            "volume": [100] * 4,
        },
        index=dates,
    )
    parquet_path = tmp_path / "bars.parquet"
    df.to_parquet(parquet_path)
    cache_path = tmp_path / "cache.parquet"
    bridge = NautilusBridge(data_path=parquet_path, cache_path=cache_path)
    assert len(bridge._bars) == 4


def test_build_command_respects_holdout_start(tmp_path, monkeypatch):
    """build-Command übergibt holdout_start als max_date an NautilusBridge."""
    import pandas as pd
    from typer.testing import CliRunner
    from sb.cli import app

    # Minimales Parquet mit Bars aus 2025 und 2026
    dates = pd.to_datetime(
        ["2025-06-01", "2025-09-01", "2026-02-01", "2026-03-01"], utc=True
    )
    df = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.5] * 4,
            "volume": [100] * 4,
        },
        index=dates,
    )
    data_path = tmp_path / "bars.parquet"
    df.to_parquet(data_path)

    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text(
        f"backtest_data:\n  path: {data_path}\n  holdout_start: '2026-01-01'\n"
    )

    captured_max_dates = []

    def fake_init(self, data_path, cache_path, **kwargs):
        captured_max_dates.append(kwargs.get("max_date"))
        # Vorzeitig abbrechen – kein echter Backtest
        raise SystemExit(0)

    monkeypatch.setattr(NautilusBridge, "__init__", fake_init)

    output_dir = tmp_path / "output"
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "build",
            "BOS + FVG NY",
            "--sources",
            str(sources_yaml),
            "--trials",
            "1",
            "--output",
            str(output_dir),
        ],
    )

    assert "2026-01-01" in captured_max_dates


def test_run_engine_passes_friction_into_replayer_config(monkeypatch):
    bridge = NautilusBridge.from_bars(
        bars=[],
        cache_df=None,
        instrument=type("Instrument", (), {"id": "MNQM6.SIM"})(),
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        venue="SIM",
    )

    captured = {}

    class DummyReplayer:
        def __init__(self, config) -> None:
            captured["config"] = config
            self.wins = 0
            self.losses = 0
            self.gross_profit_pts = 0.0
            self.gross_loss_pts = 0.0
            self.pnl_series = []
            self._trade_records = []

    class DummyEngine:
        def __init__(self, config) -> None:
            self.config = config

        def add_venue(self, **kwargs) -> None:
            pass

        def add_instrument(self, instrument) -> None:
            pass

        def add_data(self, bars) -> None:
            pass

        def add_strategy(self, strategy) -> None:
            self.strategy = strategy

        def run(self) -> None:
            pass

    monkeypatch.setattr("sb.engine.nautilus_bridge.BacktestEngine", DummyEngine)
    monkeypatch.setattr("sb.engine.nautilus_bridge.SBReplayer", DummyReplayer)

    result = bridge._run_engine(
        signals=[],
        sl_points=10.0,
        tp_mult=2.5,
        params={"sl_points": 10.0, "tp_mult": 2.5},
    )

    cfg = captured["config"]
    assert cfg.point_value == POINT_VALUE
    assert cfg.slippage_points == SLIPPAGE_POINTS
    assert cfg.commission_usd == COMMISSION_USD
    assert isinstance(result, BacktestResult)


def test_compute_atr_at_no_lookahead():
    """_compute_atr_at darf nur Bars bis idx (inklusiv) verwenden – kein Lookahead."""
    from types import SimpleNamespace
    from sb.engine.nautilus_bridge import NautilusBridge

    def make_bar(h, l, c):
        return SimpleNamespace(high=h, low=l, close=c, open=c, ts_event=0)

    # 20 Bars: erste 10 mit TR~4, dann TR~200 (Ausreißer)
    bars = [make_bar(h=102, l=98, c=100)] * 10
    bars += [make_bar(h=200, l=0, c=100)] * 10

    bridge = NautilusBridge.from_bars(
        bars=bars, cache_df=None, instrument=None, bar_type=None, venue=None
    )
    # idx=9, window=5 → nur Bars 5-9 → TR~4, kein Ausreißer aus Bars 10+
    atr_early = bridge._compute_atr_at(9, window=5)
    # idx=19, window=5 → Bars 15-19 → TR~200
    atr_late = bridge._compute_atr_at(19, window=5)
    assert atr_early < 10.0, f"kein Lookahead erwartet, atr={atr_early}"
    assert atr_late > 50.0


def test_compute_atr_fallback_for_insufficient_history():
    """_compute_atr_at gibt 10.0 zurück wenn idx < 1."""
    from sb.engine.nautilus_bridge import NautilusBridge

    bridge = NautilusBridge.from_bars(
        bars=[], cache_df=None, instrument=None, bar_type=None, venue=None
    )
    assert bridge._compute_atr_at(0) == 10.0
    assert bridge._compute_atr_at(-1) == 10.0


def test_run_atr_mode_fallback_when_atr_mult_is_none(sample_parquet, signal_cache):
    """run() fällt auf sl_points zurück wenn atr_mult nicht in params."""
    bridge = NautilusBridge(data_path=sample_parquet, cache_path=signal_cache)
    result = bridge.run(
        {
            "sl_points": 10.0,
            "tp_mult": 2.5,
            "entry_bar_offset": 0,
            "concepts": [],
            "direction": 1,
            # kein atr_mult → legacy-Modus
        }
    )
    assert isinstance(result, BacktestResult)


def test_run_atr_mode_explicit_none_does_not_crash(sample_parquet, signal_cache):
    """run() wirft keinen TypeError wenn atr_mult=None explizit übergeben wird."""
    bridge = NautilusBridge(data_path=sample_parquet, cache_path=signal_cache)
    result = bridge.run(
        {
            "atr_mult": None,  # explizit None → muss auf sl_points fallen
            "sl_points": 10.0,
            "tp_mult": 2.5,
            "entry_bar_offset": 0,
            "concepts": [],
            "direction": 1,
        }
    )
    assert isinstance(result, BacktestResult)
