from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from sb.engine.worker import WorkerConfig, run_worker_task


def _make_cfg(tmp_path: Path) -> WorkerConfig:
    data = tmp_path / "data.parquet"
    data.touch()
    return WorkerConfig(
        idea="BOS + FVG London",
        trials=5,
        data_path=data,
        cache_path=tmp_path / "cache",
        db_path=tmp_path / "builder.db",
        output_dir=tmp_path / "output",
        min_trades=5,
    )


def test_worker_result_has_mc_pct_profitable_key(tmp_path):
    """run_worker_task Ergebnis muss 'mc_pct_profitable' key enthalten (kann None sein)."""
    cfg = _make_cfg(tmp_path)

    # Stub WalkForwardResult
    wf_stub = MagicMock()
    wf_stub.pbo_score = float("nan")
    wf_stub.oos_pf = 2.5
    wf_stub.windows = []
    wf_stub.oos_pnl_series = []

    with (
        patch("sb.engine.worker.parse_idea") as mock_parse,
        patch("sb.engine.worker.NautilusBridge") as mock_bridge_cls,
        patch("sb.engine.worker.WalkForwardEngine") as mock_wfe_cls,
        patch("sb.engine.worker.generate_wf_report"),
    ):
        mock_parse.return_value = MagicMock(
            raw="BOS + FVG London", concepts=["BOS", "FVG"], session="london"
        )
        mock_bridge_cls.return_value = MagicMock()
        mock_wfe = MagicMock()
        mock_wfe.run.return_value = wf_stub
        mock_wfe_cls.return_value = mock_wfe

        result = run_worker_task(cfg)

    assert "mc_pct_profitable" in result, (
        "run_worker_task Ergebnis muss 'mc_pct_profitable' key enthalten"
    )
