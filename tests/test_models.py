from __future__ import annotations

from sb.models import EvalResult, ParsedIdea, BacktestResult


def test_parsed_idea_defaults():
    idea = ParsedIdea(raw="BOS + FVG London Open", concepts=["BOS", "FVG"])
    assert idea.session == "london"
    assert idea.sl_hint_points is None
    assert idea.concepts == ["BOS", "FVG"]


def test_backtest_result_profit_factor():
    result = BacktestResult(
        params={"sl": 10, "tp_mult": 2.5},
        gross_profit=1000.0,
        gross_loss=400.0,
        num_trades=50,
        num_wins=30,
    )
    assert abs(result.profit_factor - 2.5) < 0.01
    assert abs(result.winrate - 0.6) < 0.01


def test_backtest_result_zero_loss():
    result = BacktestResult(
        params={},
        gross_profit=500.0,
        gross_loss=0.0,
        num_trades=10,
        num_wins=10,
    )
    assert result.profit_factor == float("inf")


def test_eval_result_ordering():
    r1 = EvalResult(
        rank=1, score=4.5, result=BacktestResult({}, 1000, 200, 40, 30), warnings=[]
    )
    r2 = EvalResult(
        rank=2,
        score=2.1,
        result=BacktestResult({}, 500, 300, 20, 10),
        warnings=["Warnung"],
    )
    assert r1 > r2


def test_walk_forward_result_oos_pf_average():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    def make_wfr(pfs):
        windows = []
        for i, pf in enumerate(pfs):
            gp = pf * 100.0
            is_r = BacktestResult(
                params={},
                gross_profit=100.0,
                gross_loss=100.0,
                num_trades=10,
                num_wins=5,
            )
            oos_r = BacktestResult(
                params={}, gross_profit=gp, gross_loss=100.0, num_trades=10, num_wins=5
            )
            windows.append(
                WindowResult(window_idx=i, in_sample=is_r, oos=oos_r, best_params={})
            )
        return WalkForwardResult(windows=windows, importances={})

    wfr = make_wfr([1.5, 2.0, 1.0])
    assert abs(wfr.oos_pf - 1.5) < 0.01


def test_walk_forward_result_is_robust_true():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    windows = []
    for i in range(3):
        r = BacktestResult(
            params={}, gross_profit=150.0, gross_loss=100.0, num_trades=20, num_wins=12
        )
        windows.append(WindowResult(window_idx=i, in_sample=r, oos=r, best_params={}))
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.is_robust is True


def test_walk_forward_result_is_robust_false_when_one_window_fails():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    def make_r(gp):
        return BacktestResult(
            params={}, gross_profit=gp, gross_loss=100.0, num_trades=20, num_wins=10
        )

    windows = [
        WindowResult(
            window_idx=0, in_sample=make_r(150.0), oos=make_r(150.0), best_params={}
        ),
        WindowResult(
            window_idx=1, in_sample=make_r(150.0), oos=make_r(80.0), best_params={}
        ),
        WindowResult(
            window_idx=2, in_sample=make_r(150.0), oos=make_r(120.0), best_params={}
        ),
    ]
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.is_robust is False


def test_walk_forward_result_best_params_from_last_window():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    r = BacktestResult(
        params={}, gross_profit=100.0, gross_loss=100.0, num_trades=10, num_wins=5
    )
    windows = [
        WindowResult(window_idx=0, in_sample=r, oos=r, best_params={"sl_points": 8.0}),
        WindowResult(window_idx=1, in_sample=r, oos=r, best_params={"sl_points": 12.0}),
        WindowResult(window_idx=2, in_sample=r, oos=r, best_params={"sl_points": 10.0}),
    ]
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.best_params["sl_points"] == 10.0


def test_walk_forward_result_oos_trades_sum():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    def make_r(trades):
        return BacktestResult(
            params={},
            gross_profit=100.0,
            gross_loss=80.0,
            num_trades=trades,
            num_wins=trades // 2,
        )

    windows = [
        WindowResult(window_idx=i, in_sample=make_r(10), oos=make_r(t), best_params={})
        for i, t in enumerate([20, 25, 30])
    ]
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.oos_trades == 75


def test_walk_forward_result_oos_pf_uses_aggregate_pnl():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    is_r = BacktestResult(
        params={}, gross_profit=100.0, gross_loss=50.0, num_trades=10, num_wins=5
    )
    windows = [
        WindowResult(
            window_idx=0,
            in_sample=is_r,
            oos=BacktestResult(
                params={},
                gross_profit=300.0,
                gross_loss=100.0,
                num_trades=20,
                num_wins=12,
            ),
            best_params={},
        ),
        WindowResult(
            window_idx=1,
            in_sample=is_r,
            oos=BacktestResult(
                params={},
                gross_profit=40.0,
                gross_loss=200.0,
                num_trades=20,
                num_wins=8,
            ),
            best_params={},
        ),
    ]
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.oos_pf == 1.133


def test_walk_forward_result_is_not_robust_with_zero_trade_window():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    profitable = BacktestResult(
        params={}, gross_profit=150.0, gross_loss=100.0, num_trades=20, num_wins=12
    )
    no_trades = BacktestResult(
        params={}, gross_profit=0.0, gross_loss=0.0, num_trades=0, num_wins=0
    )
    wfr = WalkForwardResult(
        windows=[
            WindowResult(
                window_idx=0,
                in_sample=profitable,
                oos=profitable,
                best_params={},
            ),
            WindowResult(
                window_idx=1, in_sample=profitable, oos=no_trades, best_params={}
            ),
        ],
        importances={},
    )
    assert wfr.is_robust is False


def test_parsed_idea_has_role_fields():
    idea = ParsedIdea(raw="BOS + FVG NY", concepts=["BOS", "FVG"])
    assert hasattr(idea, "entry")
    assert hasattr(idea, "zone")
    assert hasattr(idea, "context")
    assert hasattr(idea, "timing")


def test_parsed_idea_role_fields_default_empty():
    idea = ParsedIdea(raw="BOS NY", concepts=["BOS"])
    assert idea.entry == []
    assert idea.zone == []
    assert idea.context == []
    assert idea.timing == []


def test_parsed_idea_role_fields_settable():
    idea = ParsedIdea(
        raw="BOS + FVG NY",
        concepts=["BOS", "FVG"],
        entry=["BOS"],
        zone=["FVG"],
    )
    assert idea.entry == ["BOS"]
    assert idea.zone == ["FVG"]
    assert idea.context == []
    assert idea.timing == []


def test_parsed_idea_concepts_unchanged():
    """Existierender concepts-Field bleibt unverändert (Backward-Compat)."""
    idea = ParsedIdea(raw="BOS NY", concepts=["BOS", "FVG"])
    assert idea.concepts == ["BOS", "FVG"]


def test_backtest_result_has_pnl_series():
    from sb.models import BacktestResult

    r = BacktestResult(
        params={},
        gross_profit=100.0,
        gross_loss=50.0,
        num_trades=3,
        num_wins=2,
        pnl_series=[50.0, -25.0, 75.0],
    )
    assert r.pnl_series == [50.0, -25.0, 75.0]


def test_backtest_result_pnl_series_default_empty():
    from sb.models import BacktestResult

    r = BacktestResult(
        params={}, gross_profit=0.0, gross_loss=0.0, num_trades=0, num_wins=0
    )
    assert r.pnl_series == []


def test_walk_forward_result_oos_pnl_series():
    from sb.models import BacktestResult, WalkForwardResult, WindowResult

    def make_result(pnl_series):
        return BacktestResult(
            params={},
            gross_profit=100.0,
            gross_loss=50.0,
            num_trades=2,
            num_wins=1,
            pnl_series=pnl_series,
        )

    windows = [
        WindowResult(0, make_result([10.0, -5.0]), make_result([20.0, -8.0]), {}),
        WindowResult(1, make_result([5.0]), make_result([15.0, 12.0]), {}),
    ]
    wfr = WalkForwardResult(windows=windows, importances={})
    assert wfr.oos_pnl_series == [20.0, -8.0, 15.0, 12.0]


def test_trade_record_creation():
    from sb.models import TradeRecord
    t = TradeRecord(
        entry_ns=1_700_000_000_000_000_000,
        exit_ns=1_700_000_060_000_000_000,
        direction=1,
        entry_price=19000.0,
        exit_price=19020.0,
        pnl_pts=10.0,
    )
    assert t.direction == 1
    assert t.pnl_pts == 10.0


def test_backtest_result_raw_trades_default_empty():
    from sb.models import BacktestResult
    r = BacktestResult(params={}, gross_profit=100.0, gross_loss=50.0,
                       num_trades=2, num_wins=1)
    assert r.raw_trades == []


def test_backtest_result_with_trades():
    from sb.models import BacktestResult, TradeRecord
    trade = TradeRecord(
        entry_ns=1_700_000_000_000_000_000,
        exit_ns=1_700_000_060_000_000_000,
        direction=-1,
        entry_price=19000.0,
        exit_price=18980.0,
        pnl_pts=10.0,
    )
    r = BacktestResult(params={}, gross_profit=100.0, gross_loss=0.0,
                       num_trades=1, num_wins=1, raw_trades=[trade])
    assert len(r.raw_trades) == 1
    assert r.raw_trades[0].direction == -1
