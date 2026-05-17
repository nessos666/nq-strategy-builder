from __future__ import annotations

import sys
import types

import optuna
import pytest

from sb.engine.backtest_bridge import BacktestBridge
from sb.engine.kombinator import Kombinator
from sb.models import BacktestResult, ParsedIdea


def test_kombinator_returns_results(sample_parquet):
    idea = ParsedIdea(raw="BOS + FVG London", concepts=["BOS", "FVG"], session="london")
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=5)
    results = kom.search(idea)
    assert len(results) == 5
    assert all(isinstance(r, BacktestResult) for r in results)


def test_kombinator_atr_mult_in_valid_range(sample_parquet):
    """atr_mult liegt immer im konfigurierten Bereich [0.8, 2.5]."""
    from sb.engine.kombinator import ATR_MULT_HIGH, ATR_MULT_LOW

    idea = ParsedIdea(raw="FVG London", concepts=["FVG"], session="london")
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=5)
    results = kom.search(idea)
    for r in results:
        atr = r.params.get("atr_mult", 0)
        assert ATR_MULT_LOW <= atr <= ATR_MULT_HIGH


def test_kombinator_best_result_has_highest_pf(sample_parquet):
    idea = ParsedIdea(raw="OB NY", concepts=["OB"], session="ny")
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=10)
    results = kom.search(idea)
    sorted_results = sorted(results, key=lambda r: r.profit_factor, reverse=True)
    assert sorted_results[0].profit_factor >= sorted_results[-1].profit_factor


def test_kombinator_exposes_last_study_after_search(sample_parquet):
    from sb.engine.backtest_bridge import BacktestBridge
    from sb.engine.kombinator import Kombinator
    from sb.models import ParsedIdea

    idea = ParsedIdea(raw="BOS", concepts=["BOS"], session="all")
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=3)
    kom.search(idea)
    assert kom.last_study is not None
    assert hasattr(kom.last_study, "best_trial")
    assert kom.last_study.best_trial is not None


def test_kombinator_persists_study_to_sqlite(tmp_path, sample_parquet):
    """Study-Trials werden in SQLite gespeichert und beim zweiten Run geladen."""
    from sb.engine.backtest_bridge import BacktestBridge
    from sb.engine.kombinator import Kombinator
    from sb.models import ParsedIdea

    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    study_name = "test_study_persist"
    idea = ParsedIdea(raw="BOS London", concepts=["BOS"], session="london")
    bridge = BacktestBridge(data_path=sample_parquet)

    # Erster Run: 3 Trials
    kom1 = Kombinator(bridge=bridge, n_trials=3, storage=storage, study_name=study_name)
    kom1.search(idea)
    assert kom1.last_study is not None
    assert len(kom1.last_study.trials) == 3

    # Zweiter Run: lädt Study und hängt 3 weitere Trials dran
    kom2 = Kombinator(bridge=bridge, n_trials=3, storage=storage, study_name=study_name)
    kom2.search(idea)
    assert kom2.last_study is not None
    assert len(kom2.last_study.trials) == 6  # 3 + 3


def test_kombinator_without_storage_works_as_before(sample_parquet):
    """Ohne storage-Parameter: Verhalten wie bisher (in-memory)."""
    from sb.engine.backtest_bridge import BacktestBridge
    from sb.engine.kombinator import Kombinator
    from sb.models import ParsedIdea

    idea = ParsedIdea(raw="OB NY", concepts=["OB"], session="ny")
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=3)
    results = kom.search(idea)
    assert len(results) == 3


def test_suggest_params_includes_role_fields():
    """_suggest_params muss entry/zone/context/timing in params weitergeben."""
    from unittest.mock import MagicMock
    import optuna
    from sb.engine.kombinator import Kombinator
    from sb.models import ParsedIdea

    bridge = MagicMock()
    bridge.run.return_value = MagicMock(
        profit_factor=1.5, winrate=0.5, num_trades=20, max_drawdown=5.0
    )
    kom = Kombinator(bridge=bridge, n_trials=1)

    idea = ParsedIdea(
        raw="BOS + FVG + Hurst NY",
        concepts=["BOS", "FVG", "HURST"],
        session="ny",
        entry=["BOS"],
        zone=["FVG"],
        context=["HURST"],
        timing=[],
    )

    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 10.0
    trial.suggest_int.return_value = 1

    params = kom._suggest_params(trial, idea)
    assert params["entry"] == ["BOS"]
    assert params["zone"] == ["FVG"]
    assert params["context"] == ["HURST"]
    assert params["timing"] == []
    assert params["concepts"] == ["BOS", "FVG", "HURST"]  # Backward-Compat


def test_suggest_params_backward_compat_empty_roles():
    """Wenn keine Rollen gesetzt, concepts bleibt wie gehabt."""
    from unittest.mock import MagicMock
    import optuna
    from sb.engine.kombinator import Kombinator
    from sb.models import ParsedIdea

    bridge = MagicMock()
    kom = Kombinator(bridge=bridge, n_trials=1)

    idea = ParsedIdea(raw="BOS NY", concepts=["BOS"], session="ny")

    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 10.0
    trial.suggest_int.return_value = 1

    params = kom._suggest_params(trial, idea)
    assert params["concepts"] == ["BOS"]
    assert params["entry"] == []
    assert params["zone"] == []
    assert params["context"] == []
    assert params["timing"] == []


def test_suggest_params_normalizes_none_roles_to_lists():
    from unittest.mock import MagicMock

    bridge = MagicMock()
    kom = Kombinator(bridge=bridge, n_trials=1)

    idea = ParsedIdea(
        raw="BOS NY",
        concepts=["BOS"],
        session="ny",
        entry=None,
        zone=None,
        context=None,
        timing=None,
    )

    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 10.0
    trial.suggest_int.return_value = 1

    params = kom._suggest_params(trial, idea)
    assert params["entry"] == []
    assert params["zone"] == []
    assert params["context"] == []
    assert params["timing"] == []


def test_build_search_space_default_bounds():
    """_build_search_space: atr_mult mit Default-Bounds 0.8-2.5."""
    from sb.engine.kombinator import ATR_MULT_HIGH, ATR_MULT_LOW, _build_search_space

    idea = ParsedIdea(raw="BOS NY", concepts=["BOS"], session="ny")
    space = _build_search_space(idea)
    assert "atr_mult" in space
    assert "tp_mult" in space
    assert "entry_bar_offset" in space
    assert "sl_points" not in space
    assert space["atr_mult"].low == ATR_MULT_LOW  # type: ignore[union-attr]
    assert space["atr_mult"].high == ATR_MULT_HIGH  # type: ignore[union-attr]
    assert space["tp_mult"].low == 1.5  # type: ignore[union-attr]
    assert space["tp_mult"].high == 5.0  # type: ignore[union-attr]
    assert space["entry_bar_offset"].low == 0  # type: ignore[union-attr]
    assert space["entry_bar_offset"].high == 5  # type: ignore[union-attr]


def test_build_search_space_sl_hint_removes_atr_mult():
    """sl_hint_points gesetzt → atr_mult NICHT im Suchraum (fixer SL)."""
    from sb.engine.kombinator import _build_search_space

    idea = ParsedIdea(
        raw="FVG NY SL 15", concepts=["FVG"], session="ny", sl_hint_points=15.0
    )
    space = _build_search_space(idea)
    assert "atr_mult" not in space
    assert "tp_mult" in space
    assert "entry_bar_offset" in space


def test_build_search_space_no_sl_points_key():
    """sl_points darf NICHT mehr im Suchraum sein."""
    from sb.engine.kombinator import _build_search_space

    idea = ParsedIdea(raw="FVG NY", concepts=["FVG"], session="ny")
    space = _build_search_space(idea)
    assert "sl_points" not in space


def test_suggest_params_uses_atr_mult_bounds():
    from unittest.mock import MagicMock

    from sb.engine.kombinator import ATR_MULT_HIGH, ATR_MULT_LOW

    bridge = MagicMock()
    kom = Kombinator(bridge=bridge, n_trials=1)
    idea = ParsedIdea(raw="FVG NY", concepts=["FVG"], session="ny")
    trial = MagicMock(spec=optuna.Trial)
    trial.suggest_float.return_value = 1.5
    trial.suggest_int.return_value = 1

    params = kom._suggest_params(trial, idea)

    trial.suggest_float.assert_any_call("atr_mult", ATR_MULT_LOW, ATR_MULT_HIGH)
    assert params["atr_mult"] == 1.5


def test_make_sampler_returns_confopt_when_available(monkeypatch):
    """Wenn verfügbar, wird ConfOptSampler mit search_space instanziiert."""
    from sb.engine.kombinator import _make_sampler

    space = {
        "sl_points": optuna.distributions.FloatDistribution(4.0, 30.0),
        "tp_mult": optuna.distributions.FloatDistribution(1.5, 5.0),
    }
    captured: dict[str, object] = {}

    class FakeConfOptSampler:
        def __init__(self, *, search_space):
            captured["search_space"] = search_space

    module = types.SimpleNamespace(ConfOptSampler=FakeConfOptSampler)
    monkeypatch.setitem(
        sys.modules, "optunahub", types.SimpleNamespace(load_module=lambda _: module)
    )

    sampler = _make_sampler(None, "study", search_space=space, use_confopt=True)

    assert isinstance(sampler, FakeConfOptSampler)
    assert captured["search_space"] == space


def test_make_sampler_confopt_falls_back_to_meta_learn_tpe(monkeypatch):
    """Wenn ConfOpt fehlschlägt, geht die Kette weiter zu MetaLearnTPE."""
    from sb.engine.kombinator import _make_sampler

    class FakeStudySummary:
        def __init__(self, study_name, best_trial):
            self.study_name = study_name
            self.best_trial = best_trial

    class FakeMetaLearnTPESampler:
        def __init__(self, *, source_studies):
            self.source_studies = source_studies

    def fake_load_module(name):
        if name == "samplers/confopt_sampler":
            raise RuntimeError("confopt broken")
        if name == "samplers/meta_learn_tpe":
            return types.SimpleNamespace(MetaLearnTPESampler=FakeMetaLearnTPESampler)
        raise AssertionError(name)

    monkeypatch.setitem(
        sys.modules, "optunahub", types.SimpleNamespace(load_module=fake_load_module)
    )
    monkeypatch.setattr(
        "sb.engine.kombinator.optuna.get_all_study_summaries",
        lambda storage: [
            FakeStudySummary("current", object()),
            FakeStudySummary("source", object()),
        ],
    )
    monkeypatch.setattr(
        "sb.engine.kombinator.optuna.load_study",
        lambda *, study_name, storage: {"study_name": study_name, "storage": storage},
    )

    sampler = _make_sampler(
        "sqlite:///tmp.db",
        "current",
        search_space={"sl_points": optuna.distributions.FloatDistribution(4.0, 30.0)},
        use_confopt=True,
    )

    assert isinstance(sampler, FakeMetaLearnTPESampler)
    assert sampler.source_studies == [
        {"study_name": "source", "storage": "sqlite:///tmp.db"}
    ]


def test_make_sampler_confopt_fallback_to_tpe(monkeypatch):
    """Wenn use_confopt=True aber weder ConfOpt noch MetaLearnTPE klappen: TPE."""
    from sb.engine.kombinator import _make_sampler

    monkeypatch.setitem(
        sys.modules,
        "optunahub",
        types.SimpleNamespace(
            load_module=lambda _: (_ for _ in ()).throw(RuntimeError("boom"))
        ),
    )

    sampler = _make_sampler(
        None,
        None,
        search_space={"sl_points": optuna.distributions.FloatDistribution(4.0, 30.0)},
        use_confopt=True,
    )
    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_make_sampler_confopt_without_search_space_uses_tpe():
    from sb.engine.kombinator import _make_sampler

    sampler = _make_sampler(None, "study", search_space=None, use_confopt=True)

    assert isinstance(sampler, optuna.samplers.TPESampler)


def test_kombinator_passes_confopt_flag_and_search_space(monkeypatch, sample_parquet):
    captured: dict[str, object] = {}

    def fake_make_sampler(storage, study_name, search_space=None, use_confopt=False):
        captured["storage"] = storage
        captured["study_name"] = study_name
        captured["search_space"] = search_space
        captured["use_confopt"] = use_confopt
        return optuna.samplers.RandomSampler(seed=0)

    monkeypatch.setattr("sb.engine.kombinator._make_sampler", fake_make_sampler)

    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(
        bridge=bridge,
        n_trials=1,
        storage="sqlite:///tmp.db",
        study_name="confopt-study",
        use_confopt=True,
    )
    idea = ParsedIdea(raw="BOS NY", concepts=["BOS"], session="ny")

    kom.search(idea)

    assert captured["storage"] == "sqlite:///tmp.db"
    assert captured["study_name"] == "confopt-study"
    assert captured["use_confopt"] is True
    assert isinstance(captured["search_space"], dict)
    assert set(captured["search_space"]) == {"atr_mult", "tp_mult", "entry_bar_offset"}


def test_kombinator_with_confopt_flag(sample_parquet):
    """Kombinator mit use_confopt=True funktioniert (egal ob ConfOpt oder Fallback)."""
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=3, use_confopt=True)
    idea = ParsedIdea(raw="BOS NY", concepts=["BOS"], session="ny")
    results = kom.search(idea)
    assert len(results) == 3
    assert all(isinstance(r, BacktestResult) for r in results)


def test_kombinator_rejects_non_positive_trial_count(sample_parquet):
    bridge = BacktestBridge(data_path=sample_parquet)

    with pytest.raises(ValueError, match="n_trials must be > 0"):
        Kombinator(bridge=bridge, n_trials=0)


def test_kombinator_fixed_sl_skips_atr_mult(sample_parquet):
    """Wenn sl_hint_points gesetzt → atr_mult=None in allen Results."""
    idea = ParsedIdea(
        raw="OB Session short AM",
        concepts=["OB_SESSION", "MANIP_BEAR"],
        session="am",
        sl_hint_points=10.0,
        direction=-1,
    )
    bridge = BacktestBridge(data_path=sample_parquet)
    kom = Kombinator(bridge=bridge, n_trials=3)
    results = kom.search(idea)
    for r in results:
        assert r.params.get("atr_mult") is None
        assert r.params.get("sl_points") == 10.0
