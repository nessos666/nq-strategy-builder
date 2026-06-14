# tests/test_result_store.py
"""Tests für ResultStore."""

import pytest
pytest.importorskip("tv_kombinatorik")
import pandas as pd
from tv_kombinatorik.result_store import ResultStore
from tv_kombinatorik.tv_client import TVResult


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "results.csv")


def test_save_creates_file(store, tmp_path):
    store.save(1, {"tp_mult": 2.0}, TVResult(pf=1.24, wr=58.3, trades=142, max_dd=0.8))
    assert (tmp_path / "results.csv").exists()


def test_save_and_load_single_trial(store):
    store.save(
        1,
        {"tp_mult": 2.0, "sl_mult": 1.5},
        TVResult(pf=1.24, wr=58.3, trades=142, max_dd=0.8),
    )
    df = store.load()
    assert len(df) == 1
    assert df.iloc[0]["pf"] == pytest.approx(1.24)
    assert df.iloc[0]["tp_mult"] == pytest.approx(2.0)
    assert df.iloc[0]["trades"] == 142


def test_save_multiple_appends(store):
    for i in range(3):
        store.save(
            i,
            {"tp_mult": float(i)},
            TVResult(pf=float(i), wr=50.0, trades=100, max_dd=1.0),
        )
    df = store.load()
    assert len(df) == 3


def test_load_empty_returns_empty_dataframe(tmp_path):
    store = ResultStore(tmp_path / "nonexistent.csv")
    df = store.load()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_top_n_sorted_by_pf(store):
    for i, pf in enumerate([0.8, 1.5, 1.2, 0.9, 1.8]):
        store.save(
            i, {"tp_mult": float(i)}, TVResult(pf=pf, wr=50.0, trades=100, max_dd=1.0)
        )
    top = store.top_n(3)
    assert len(top) == 3
    assert top.iloc[0]["pf"] == pytest.approx(1.8)
    assert top.iloc[1]["pf"] == pytest.approx(1.5)
    assert top.iloc[2]["pf"] == pytest.approx(1.2)


def test_top_n_empty_store(tmp_path):
    store = ResultStore(tmp_path / "empty.csv")
    top = store.top_n(5)
    assert top.empty


def test_save_creates_parent_dir(tmp_path):
    store = ResultStore(tmp_path / "subdir" / "results.csv")
    store.save(1, {}, TVResult(pf=1.0, wr=50.0, trades=80, max_dd=0.5))
    assert (tmp_path / "subdir" / "results.csv").exists()
