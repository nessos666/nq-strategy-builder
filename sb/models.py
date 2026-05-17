from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedIdea:
    """Ergebnis des Parsers: Was hat der User eingegeben?"""

    raw: str
    concepts: list[str]
    session: str = "london"
    sl_hint_points: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Rollen-Felder (werden vom Parser gesetzt wenn Rollen erkannt wurden)
    entry: list[str] = field(default_factory=list)
    zone: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    timing: list[str] = field(default_factory=list)
    direction: int = 0  # 0 = beide, +1 = Long only, -1 = Short only
    use_trail: bool = False  # True → Trail-SL in Optuna-Suchraum aufnehmen
    exit_mode: str = "fixed"  # fixed, atr_trail, breakeven, next_zone, session_level, breakeven_trail


@dataclass
class KnowledgeCtx:
    """Geladene Wissensbasis für den Kombinator."""

    pda_algos: list[str]
    known_errors: list[str]
    feedback_rules: list[str]
    learnings: list[str]
    ideas: list[str]


@dataclass
class BacktestResult:
    """Ergebnis eines einzelnen Backtest-Laufs."""

    params: dict[str, Any]
    gross_profit: float
    gross_loss: float
    num_trades: int
    num_wins: int
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    pnl_series: list[float] = field(default_factory=list)
    raw_trades: list["TradeRecord"] = field(default_factory=list)

    @property
    def profit_factor(self) -> float:
        if self.gross_loss == 0:
            return float("inf") if self.gross_profit > 0 else 0.0
        return round(self.gross_profit / self.gross_loss, 3)

    @property
    def winrate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        return round(self.num_wins / self.num_trades, 3)


@dataclass
class TradeRecord:
    """Einzelner Trade aus einem Backtest-Lauf."""

    entry_ns: int  # Einstiegs-Timestamp in Nanosekunden UTC
    exit_ns: int  # Ausstiegs-Timestamp in Nanosekunden UTC
    direction: int  # +1 = Long, -1 = Short
    entry_price: float
    exit_price: float
    pnl_pts: float  # Netto PnL in Punkten (nach Slippage+Kommission)


@dataclass
class EvalResult:
    """Bewertetes und geranktes Backtest-Ergebnis."""

    rank: int
    score: float
    result: BacktestResult
    warnings: list[str]

    def __gt__(self, other: "EvalResult") -> bool:
        return self.score > other.score


@dataclass
class WindowResult:
    """Ergebnis eines einzelnen Walk-Forward-Fensters."""

    window_idx: int
    in_sample: BacktestResult
    oos: BacktestResult
    best_params: dict[str, Any]


@dataclass
class WalkForwardResult:
    """Aggregiertes Ergebnis aller Walk-Forward-Fenster + PED-ANOVA."""

    windows: list[WindowResult]
    importances: dict[str, float]
    pbo_score: float = field(default_factory=lambda: float("nan"))

    @property
    def oos_pf(self) -> float:
        total_profit = sum(w.oos.gross_profit for w in self.windows)
        total_loss = sum(w.oos.gross_loss for w in self.windows)
        if total_loss == 0:
            return float("inf") if total_profit > 0 else 0.0
        return round(total_profit / total_loss, 3)

    @property
    def oos_winrate(self) -> float:
        total_wins = sum(w.oos.num_wins for w in self.windows)
        total_trades = sum(w.oos.num_trades for w in self.windows)
        return round(total_wins / total_trades, 3) if total_trades > 0 else 0.0

    @property
    def oos_trades(self) -> int:
        return sum(w.oos.num_trades for w in self.windows)

    @property
    def is_robust(self) -> bool:
        """True wenn jedes OOS-Fenster Trades hat und PF > 1.0 erreicht."""
        return bool(self.windows) and all(
            w.oos.num_trades > 0 and w.oos.profit_factor > 1.0 for w in self.windows
        )

    @property
    def best_params(self) -> dict[str, Any]:
        """Beste Parameter aus dem letzten Fenster (aktuellste Daten)."""
        return self.windows[-1].best_params if self.windows else {}

    @property
    def oos_pnl_series(self) -> list[float]:
        """Alle OOS Trade-PnLs über alle Fenster concateniert."""
        result: list[float] = []
        for w in self.windows:
            result.extend(w.oos.pnl_series)
        return result
