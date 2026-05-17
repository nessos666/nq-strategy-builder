from __future__ import annotations


class _FakeBridge:
    """Minimale Bridge für Tests – gibt immer dasselbe BacktestResult zurück."""

    def __init__(self, pf: float = 1.5, num_trades: int = 20) -> None:
        from sb.models import BacktestResult

        self._result = BacktestResult(
            params={},
            gross_profit=pf * 100.0,
            gross_loss=100.0,
            num_trades=num_trades,
            num_wins=int(num_trades * 0.6),
        )
        # Simuliere geladene Bars
        from unittest.mock import MagicMock

        self._bars = list(range(1000))  # type: ignore[assignment]
        self._cache_df = None
        self._instrument = MagicMock()
        self._bar_type = MagicMock()
        self._venue = MagicMock()

    def run(self, params: dict):  # type: ignore[override]
        return self._result


def test_split_windows_returns_three_windows():
    from sb.engine.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(_FakeBridge(), n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
    bars = list(range(1000))
    windows = engine._split_windows(bars)  # type: ignore[arg-type]
    assert len(windows) == 3


def test_split_windows_oos_non_overlapping():
    from sb.engine.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(_FakeBridge(), n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
    bars = list(range(1000))
    windows = engine._split_windows(bars)  # type: ignore[arg-type]
    # OOS-Perioden dürfen sich nicht überlappen
    oos_starts = [w[1][0] for w in windows]
    assert oos_starts == sorted(oos_starts), "OOS-Fenster müssen chronologisch sein"


def test_split_windows_is_larger_than_oos():
    from sb.engine.walk_forward import WalkForwardEngine

    engine = WalkForwardEngine(_FakeBridge(), n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
    bars = list(range(1000))
    windows = engine._split_windows(bars)  # type: ignore[arg-type]
    for is_bars, oos_bars in windows:
        assert len(is_bars) > len(oos_bars), "IS muss größer als OOS sein"


def test_average_importances_correct():
    from sb.engine.walk_forward import _average_importances

    imps = [
        {"sl_points": 0.8, "tp_mult": 0.2},
        {"sl_points": 0.6, "tp_mult": 0.4},
    ]
    result = _average_importances(imps)
    assert abs(result["sl_points"] - 0.7) < 0.001
    assert abs(result["tp_mult"] - 0.3) < 0.001


def test_average_importances_empty_returns_empty():
    from sb.engine.walk_forward import _average_importances

    assert _average_importances([]) == {}


def test_average_importances_ignores_empty_windows():
    from sb.engine.walk_forward import _average_importances

    imps = [{}, {"sl_points": 0.6, "tp_mult": 0.4}]
    result = _average_importances(imps)
    assert result == {"sl_points": 0.6, "tp_mult": 0.4}


def test_walk_forward_engine_run_returns_three_windows():
    """Integration: WalkForwardEngine mit FakeBridge läuft durch ohne Fehler."""
    from unittest.mock import MagicMock, patch
    from sb.engine.walk_forward import WalkForwardEngine
    from sb.models import BacktestResult, ParsedIdea

    fake_result = BacktestResult(
        params={"sl_points": 10.0, "tp_mult": 2.5, "entry_bar_offset": 0},
        gross_profit=150.0,
        gross_loss=100.0,
        num_trades=20,
        num_wins=12,
    )

    # Mock NautilusBridge.from_bars() damit kein Datei-IO passiert
    fake_is_bridge = MagicMock()
    fake_is_bridge.run.return_value = fake_result
    fake_oos_bridge = MagicMock()
    fake_oos_bridge.run.return_value = fake_result

    # Mock Kombinator.search() + last_study
    import optuna

    mock_study = MagicMock(spec=optuna.Study)
    mock_study.best_trial = MagicMock()
    mock_study.best_trial.params = {
        "sl_points": 10.0,
        "tp_mult": 2.5,
        "entry_bar_offset": 0,
    }

    mock_kom = MagicMock()
    mock_kom.search.return_value = [fake_result]
    mock_kom.last_study = mock_study

    bridge = _FakeBridge()

    with (
        patch(
            "sb.engine.walk_forward.NautilusBridge.from_bars",
            side_effect=[fake_is_bridge, fake_oos_bridge] * 3,
        ),
        patch("sb.engine.walk_forward.Kombinator", return_value=mock_kom),
        patch(
            "sb.engine.walk_forward.get_param_importances",
            return_value={"sl_points": 0.7, "tp_mult": 0.3},
        ),
    ):
        engine = WalkForwardEngine(bridge, n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
        idea = ParsedIdea(raw="BOS", concepts=["bos"], session="all")
        result = engine.run(idea, n_trials=3)

    assert len(result.windows) == 3
    assert "sl_points" in result.importances


def test_walk_forward_engine_returns_empty_result_when_dataset_too_small():
    from sb.engine.walk_forward import WalkForwardEngine
    from sb.models import ParsedIdea

    bridge = _FakeBridge()
    bridge._bars = list(range(3))

    engine = WalkForwardEngine(bridge, n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
    result = engine.run(
        ParsedIdea(raw="BOS", concepts=["bos"], session="all"), n_trials=3
    )

    assert result.windows == []
    assert result.best_params == {}
    assert result.is_robust is False


def test_walk_forward_engine_handles_missing_best_trial():
    from unittest.mock import MagicMock, patch

    from sb.engine.walk_forward import WalkForwardEngine
    from sb.models import BacktestResult, ParsedIdea

    fake_result = BacktestResult(
        params={},
        gross_profit=0.0,
        gross_loss=0.0,
        num_trades=0,
        num_wins=0,
    )
    fake_bridge = MagicMock()
    fake_bridge.run.return_value = fake_result

    class _StudyWithoutBestTrial:
        @property
        def best_trial(self):
            raise ValueError("No trials are completed yet.")

    mock_kom = MagicMock()
    mock_kom.search.return_value = []
    mock_kom.last_study = _StudyWithoutBestTrial()

    with (
        patch(
            "sb.engine.walk_forward.NautilusBridge.from_bars",
            side_effect=[fake_bridge, fake_bridge] * 3,
        ),
        patch("sb.engine.walk_forward.Kombinator", return_value=mock_kom),
        patch("sb.engine.walk_forward.get_param_importances", return_value={}),
    ):
        engine = WalkForwardEngine(_FakeBridge(), n_windows=3, in_sample_ratio=0.75)  # type: ignore[arg-type]
        result = engine.run(
            ParsedIdea(raw="BOS", concepts=["bos"], session="all"),
            n_trials=0,
        )

    assert len(result.windows) == 3
    assert all(window.best_params == {} for window in result.windows)


def test_walk_forward_engine_rejects_invalid_window_configuration():
    import pytest

    from sb.engine.walk_forward import WalkForwardEngine

    with pytest.raises(ValueError):
        WalkForwardEngine(_FakeBridge(), n_windows=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        WalkForwardEngine(_FakeBridge(), in_sample_ratio=1.0)  # type: ignore[arg-type]


def test_walk_forward_passes_storage_to_kombinator(tmp_path):
    """WalkForwardEngine reicht storage an jeden Kombinator-Aufruf weiter."""
    from unittest.mock import MagicMock, patch

    import optuna

    from sb.engine.walk_forward import WalkForwardEngine
    from sb.models import BacktestResult, ParsedIdea

    storage = f"sqlite:///{tmp_path / 'test_studies.db'}"
    bridge = _FakeBridge()
    idea = ParsedIdea(raw="BOS London", concepts=["BOS"], session="london")

    fake_result = BacktestResult(
        params={"sl_points": 10.0, "tp_mult": 2.5, "entry_bar_offset": 0},
        gross_profit=150.0,
        gross_loss=100.0,
        num_trades=20,
        num_wins=12,
    )

    mock_study = MagicMock(spec=optuna.Study)
    mock_study.best_trial = MagicMock()
    mock_study.best_trial.params = {
        "sl_points": 10.0,
        "tp_mult": 2.5,
        "entry_bar_offset": 0,
    }

    fake_is_bridge = MagicMock()
    fake_is_bridge.run.return_value = fake_result
    fake_oos_bridge = MagicMock()
    fake_oos_bridge.run.return_value = fake_result

    created_storages: list[str | None] = []

    class CapturingKombinator:
        def __init__(
            self, bridge, n_trials=50, storage=None, study_name=None, use_confopt=False
        ):
            created_storages.append(storage)
            self.last_study = mock_study
            self.last_results = [fake_result]

        def search(self, idea):
            return [fake_result]

    with (
        patch(
            "sb.engine.walk_forward.NautilusBridge.from_bars",
            side_effect=[fake_is_bridge, fake_oos_bridge] * 3,
        ),
        patch("sb.engine.walk_forward.get_param_importances", return_value={}),
        patch("sb.engine.walk_forward.Kombinator", CapturingKombinator),
    ):
        wfe = WalkForwardEngine(bridge=bridge, storage=storage)  # type: ignore[arg-type]
        wfe.run(idea, n_trials=2)

    assert len(created_storages) > 0, "Kein Kombinator erstellt"
    assert all(s == storage for s in created_storages), (
        f"Nicht alle Kombinator-Instanzen bekamen storage: {created_storages}"
    )
