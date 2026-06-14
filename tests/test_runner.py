# tests/test_runner.py
"""Tests für KombinatorikRunner."""

from unittest.mock import MagicMock, patch

import pytest
pytest.importorskip("tv_kombinatorik")

from tv_kombinatorik.runner import KombinatorikRunner
from tv_kombinatorik.tv_client import TVResult

FAKE_TEMPLATE = (
    "//@version=6\n"
    "tp_mult={{tp_mult}}\nsl_mult={{sl_mult}}\ndisp_mult={{disp_mult}}\n"
    "sw_lookback={{sw_lookback}}\nob_max_age={{ob_max_age}}\n"
    "atr_per={{atr_per}}\nmin_body={{min_body}}\n"
)


def make_client(pf: float = 1.2, trades: int = 80) -> MagicMock:
    mock = MagicMock()
    mock.inject_and_compile.return_value = True
    mock.read_result.return_value = TVResult(pf=pf, wr=55.0, trades=trades, max_dd=0.5)
    return mock


@pytest.fixture
def runner(tmp_path):
    client = make_client()
    return KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=3,
        min_trades=20,
        result_dir=tmp_path / "results",
    )


def test_runner_calls_tv_n_times(runner):
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        runner.run(storage=None)
    assert runner.tv_client.inject_and_compile.call_count == 3


def test_runner_returns_list(runner):
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        results = runner.run(storage=None)
    assert isinstance(results, list)


def test_runner_penalty_on_compile_fail(tmp_path):
    client = MagicMock()
    client.inject_and_compile.return_value = False  # Compile schlägt fehl
    r = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=2,
        min_trades=20,
        result_dir=tmp_path / "results",
    )
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        results = r.run(storage=None)
    # Alle Trials mit Penalty → keine Ergebnisse gespeichert
    assert results == []


def test_runner_penalty_on_too_few_trades(tmp_path):
    client = make_client(pf=2.5, trades=5)  # trades < min_trades=30
    r = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=2,
        min_trades=30,
        result_dir=tmp_path / "results",
    )
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        results = r.run(storage=None)
    assert results == []


def test_runner_saves_good_results(tmp_path):
    client = make_client(pf=1.5, trades=100)
    r = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=3,
        min_trades=20,
        result_dir=tmp_path / "results",
    )
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        results = r.run(storage=None)
    assert len(results) > 0
    assert results[0]["pf"] == pytest.approx(1.5)


def test_runner_result_dir_created(tmp_path):
    result_dir = tmp_path / "deep" / "nested" / "results"
    client = make_client()
    r = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        n_trials=1,
        min_trades=20,
        result_dir=result_dir,
    )
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        r.run(storage=None)
    assert result_dir.exists()


def test_quick_check_returns_best_result(tmp_path):
    client = make_client(pf=1.5, trades=55)

    runner = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        result_dir=tmp_path / "results",
    )

    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        result = runner.quick_check(n_trials=5, min_trades=30)

    assert result is not None
    assert result["pf"] == 1.5
    assert result["trades"] == 55
    assert "params" in result


def test_quick_check_all_trials_fail(tmp_path):
    """Wenn alle Trials fehlschlagen (compile fail), soll None zurückgegeben werden."""
    client = MagicMock()
    client.inject_and_compile.return_value = False

    runner = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        result_dir=tmp_path / "results",
    )
    with patch("tv_kombinatorik.runner.load_template", return_value=FAKE_TEMPLATE):
        result = runner.quick_check(n_trials=5, min_trades=30)

    assert result is None


def test_quick_check_invalid_n_trials(tmp_path):
    """n_trials < 1 soll ValueError werfen."""
    client = make_client()
    runner = KombinatorikRunner(
        strategy="s1_8_reborn",
        tv_client=client,
        result_dir=tmp_path / "results",
    )
    with pytest.raises(ValueError, match="n_trials"):
        runner.quick_check(n_trials=0)
