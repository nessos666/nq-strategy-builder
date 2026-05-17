from __future__ import annotations

from sb.engine.evaluator import Evaluator
from sb.models import BacktestResult, KnowledgeCtx


def _ctx() -> KnowledgeCtx:
    return KnowledgeCtx(
        pda_algos=[],
        known_errors=[],
        feedback_rules=[],
        learnings=[],
        ideas=[],
    )


def _result(pf: float, wr: float, trades: int) -> BacktestResult:
    wins = int(trades * wr)
    losses = trades - wins
    return BacktestResult(
        params={"sl_points": 10, "tp_mult": pf},
        gross_profit=wins * 10.0,
        gross_loss=losses * 10.0,
        num_trades=trades,
        num_wins=wins,
    )


def test_evaluator_ranks_by_score():
    ev = Evaluator(_ctx())
    results = [_result(1.5, 0.45, 50), _result(4.2, 0.60, 80), _result(2.8, 0.55, 60)]
    ranked = ev.rank(results)
    assert ranked[0].score > ranked[1].score > ranked[2].score


def test_evaluator_warns_on_too_few_trades():
    ev = Evaluator(_ctx())
    ranked = ev.rank([_result(5.0, 0.9, 5)])
    assert any("wenige Trades" in w for w in ranked[0].warnings)


def test_evaluator_warns_on_low_winrate():
    ev = Evaluator(_ctx())
    ranked = ev.rank([_result(3.0, 0.38, 100)])
    assert any("Winrate" in w for w in ranked[0].warnings)


def test_evaluator_filters_by_min_trades():
    ev = Evaluator(_ctx(), min_trades=30)
    results = [_result(5.0, 0.9, 5), _result(2.0, 0.55, 50)]
    ranked = ev.rank(results)
    assert len(ranked) == 1
    assert ranked[0].result.num_trades == 50
