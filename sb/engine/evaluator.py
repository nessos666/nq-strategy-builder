from __future__ import annotations

from sb.models import BacktestResult, EvalResult, KnowledgeCtx

TARGET_WINRATE = 0.58
TARGET_PF = 2.5
MAX_DRAWDOWN = 15.0
MIN_TRADES_DEFAULT = 30


class Evaluator:
    def __init__(self, ctx: KnowledgeCtx, min_trades: int = MIN_TRADES_DEFAULT) -> None:
        self.ctx = ctx
        self.min_trades = min_trades

    def rank(self, results: list[BacktestResult]) -> list[EvalResult]:
        filtered = [r for r in results if r.num_trades >= self.min_trades]
        if not filtered:
            filtered = results
        scored = sorted(
            [(self._score(r), r) for r in filtered], key=lambda x: x[0], reverse=True
        )
        return [
            EvalResult(
                rank=i + 1,
                score=score,
                result=result,
                warnings=self._check_warnings(result),
            )
            for i, (score, result) in enumerate(scored)
        ]

    def _score(self, r: BacktestResult) -> float:
        if r.num_trades == 0:
            return 0.0
        wr_bonus = 1.0 + max(0.0, r.winrate - TARGET_WINRATE)
        dd_penalty = r.max_drawdown / 100.0
        trade_bonus = min(1.0, r.num_trades / 100.0) * 0.1
        return round(r.profit_factor * wr_bonus - dd_penalty + trade_bonus, 4)

    def _check_warnings(self, r: BacktestResult) -> list[str]:
        warnings: list[str] = []
        if r.num_trades < self.min_trades:
            warnings.append(
                f"Nur {r.num_trades} Trades – zu wenige Trades für statistisch verlässliche Ergebnisse (min. {self.min_trades})"
            )
        if r.winrate < 0.45:
            warnings.append(
                f"Winrate {r.winrate:.0%} – sehr niedrig. Ziel: >={TARGET_WINRATE:.0%}"
            )
        if r.profit_factor < TARGET_PF:
            warnings.append(f"PF {r.profit_factor:.2f} – unter Ziel von {TARGET_PF}")
        if r.max_drawdown > MAX_DRAWDOWN:
            warnings.append(
                f"Max Drawdown {r.max_drawdown:.1f} Punkte – über Limit von {MAX_DRAWDOWN}"
            )
        return warnings
