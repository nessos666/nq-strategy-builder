from __future__ import annotations


from sb.engine.backtest_bridge import BacktestBridge
from sb.models import BacktestResult


def test_bridge_returns_result(sample_parquet):
    bridge = BacktestBridge(data_path=sample_parquet)
    params = {
        "sl_points": 10.0,
        "tp_mult": 2.5,
        "entry_bar_offset": 1,
        "signal_interval_bars": 30,
    }
    result = bridge.run(params)
    assert isinstance(result, BacktestResult)
    assert result.num_trades >= 0


def test_bridge_larger_sl_fewer_stops(sample_parquet):
    bridge = BacktestBridge(data_path=sample_parquet)
    small_sl = bridge.run(
        {
            "sl_points": 2.0,
            "tp_mult": 2.0,
            "entry_bar_offset": 0,
            "signal_interval_bars": 20,
        }
    )
    large_sl = bridge.run(
        {
            "sl_points": 50.0,
            "tp_mult": 2.0,
            "entry_bar_offset": 0,
            "signal_interval_bars": 20,
        }
    )
    assert small_sl.num_wins <= large_sl.num_wins


def test_bridge_zero_trades_for_no_data(tmp_path):
    bridge = BacktestBridge(data_path=tmp_path / "nonexistent.parquet")
    result = bridge.run(
        {
            "sl_points": 10.0,
            "tp_mult": 2.0,
            "entry_bar_offset": 0,
            "signal_interval_bars": 50,
        }
    )
    assert result.num_trades == 0
    assert result.profit_factor == 0.0  # kein Profit, kein Loss → 0
