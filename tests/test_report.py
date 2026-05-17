from __future__ import annotations

from sb.models import BacktestResult, EvalResult, ParsedIdea
from sb.report import generate_report


def _make_results() -> list[EvalResult]:
    return [
        EvalResult(
            rank=1,
            score=4.5,
            result=BacktestResult(
                params={
                    "sl_points": 10.0,
                    "tp_mult": 2.5,
                    "concepts": ["BOS", "FVG"],
                    "session": "london",
                },
                gross_profit=1200.0,
                gross_loss=400.0,
                num_trades=80,
                num_wins=50,
            ),
            warnings=[],
        ),
        EvalResult(
            rank=2,
            score=2.1,
            result=BacktestResult(
                params={
                    "sl_points": 8.0,
                    "tp_mult": 2.0,
                    "concepts": ["BOS"],
                    "session": "london",
                },
                gross_profit=600.0,
                gross_loss=400.0,
                num_trades=50,
                num_wins=28,
            ),
            warnings=["Winrate 56% – unter Ziel"],
        ),
    ]


def test_generate_report_creates_file(tmp_path):
    idea = ParsedIdea(raw="BOS + FVG London", concepts=["BOS", "FVG"], session="london")
    report_path = generate_report(
        idea=idea, results=_make_results(), output_dir=tmp_path
    )
    assert report_path.exists()
    content = report_path.read_text()
    assert "BOS + FVG London" in content
    assert "PF" in content


def test_generate_report_contains_best_params(tmp_path):
    idea = ParsedIdea(raw="FVG NY", concepts=["FVG"], session="ny")
    report_path = generate_report(
        idea=idea, results=_make_results(), output_dir=tmp_path
    )
    assert "10.0" in report_path.read_text()


def test_generate_report_shows_warnings(tmp_path):
    idea = ParsedIdea(raw="OB London", concepts=["OB"], session="london")
    report_path = generate_report(
        idea=idea, results=_make_results(), output_dir=tmp_path
    )
    assert "Winrate" in report_path.read_text()


def test_table_shows_full_pf(capsys):
    """PF-Wert darf nicht abgeschnitten werden."""
    from sb.models import BacktestResult, EvalResult, ParsedIdea
    from sb.report import _print_terminal

    idea = ParsedIdea(raw="BOS", concepts=["BOS"], session="all")
    r = BacktestResult(
        params={"sl_points": 5.0, "tp_mult": 2.5},
        gross_profit=1132.0,
        gross_loss=1000.0,
        num_trades=100,
        num_wins=20,
    )
    ev = EvalResult(rank=1, score=1.132, result=r, warnings=[])
    _print_terminal(idea, [ev])
    captured = capsys.readouterr()
    assert "1.13" in captured.out


def test_generate_wf_report_creates_file(tmp_path):
    from sb.models import BacktestResult, ParsedIdea, WalkForwardResult, WindowResult
    from sb.report import generate_wf_report

    idea = ParsedIdea(raw="BOS London", concepts=["bos"], session="london")

    def make_r(gp):
        return BacktestResult(
            params={"sl_points": 10.0, "tp_mult": 2.5, "entry_bar_offset": 0},
            gross_profit=gp,
            gross_loss=100.0,
            num_trades=20,
            num_wins=12,
        )

    windows = [
        WindowResult(
            window_idx=i,
            in_sample=make_r(150.0),
            oos=make_r(140.0 + i * 5),
            best_params={"sl_points": 10.0},
        )
        for i in range(3)
    ]
    wf_result = WalkForwardResult(
        windows=windows,
        importances={"sl_points": 0.76, "tp_mult": 0.20, "entry_bar_offset": 0.04},
    )

    path = generate_wf_report(idea=idea, wf_result=wf_result, output_dir=tmp_path)
    assert path.exists()
    content = path.read_text()
    assert "Walk-Forward" in content
    assert "OOS" in content
    assert "sl_points" in content


def test_print_walk_forward_shows_oos_pf(capsys):
    from sb.models import BacktestResult, ParsedIdea, WalkForwardResult, WindowResult
    from sb.report import _print_walk_forward

    idea = ParsedIdea(raw="BOS", concepts=["bos"], session="london")

    def make_r(gp):
        return BacktestResult(
            params={}, gross_profit=gp, gross_loss=100.0, num_trades=20, num_wins=12
        )

    windows = [
        WindowResult(
            window_idx=i, in_sample=make_r(160.0), oos=make_r(145.0), best_params={}
        )
        for i in range(3)
    ]
    wf_result = WalkForwardResult(
        windows=windows, importances={"sl_points": 0.8, "tp_mult": 0.2}
    )
    _print_walk_forward(idea, wf_result)
    captured = capsys.readouterr()
    assert "1.45" in captured.out or "OOS" in captured.out


def test_print_walk_forward_zero_trade_window_is_not_marked_as_pass(capsys):
    from sb.models import BacktestResult, ParsedIdea, WalkForwardResult, WindowResult
    from sb.report import _print_walk_forward

    idea = ParsedIdea(raw="BOS", concepts=["bos"], session="london")
    profitable = BacktestResult(
        params={}, gross_profit=160.0, gross_loss=100.0, num_trades=20, num_wins=12
    )
    no_trades = BacktestResult(
        params={}, gross_profit=0.0, gross_loss=0.0, num_trades=0, num_wins=0
    )
    wf_result = WalkForwardResult(
        windows=[
            WindowResult(
                window_idx=0,
                in_sample=profitable,
                oos=profitable,
                best_params={},
            ),
            WindowResult(window_idx=1, in_sample=profitable, oos=no_trades, best_params={}),
        ],
        importances={},
    )

    _print_walk_forward(idea, wf_result)
    captured = capsys.readouterr()
    assert "NICHT ROBUST" in captured.out
