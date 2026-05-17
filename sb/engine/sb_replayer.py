from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy


from sb.models import TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class SignalEntry:
    """Ein einzelner Signal-Eintrag für den SBReplayer."""

    timestamp: pd.Timestamp
    direction: int  # +1 = Long, -1 = Short
    entry_price: float
    sl_price: float
    tp_price: float


class SBReplayerConfig(StrategyConfig, frozen=True):
    """Konfiguration für den generischen SB Signal-Replayer."""

    instrument_id: str = "MNQM6.SIM"
    bar_type: str = "MNQM6.SIM-1-MINUTE-LAST-EXTERNAL"
    signals: tuple = ()
    """Signale als tuple von (ts_ns, direction, entry, sl, tp[, exit_idx])."""
    sl_points: float = 10.0
    tp_mult: float = 2.5
    point_value: float = 2.0  # MNQ: $2/Punkt
    slippage_points: float = 0.5  # 0.5 Punkte pro Seite
    commission_usd: float = 0.70  # Round-Trip Kommission in USD
    trail_activation: float = 0.0  # 0 = deaktiviert, >0 = nach X Punkten Profit
    trail_distance: float = 0.0  # Trail-Abstand in Punkten
    exit_mode: str = "fixed"  # fixed, atr_trail, breakeven, next_zone, session_level, breakeven_trail
    breakeven_rr: float = 0.0  # RR-Ratio für Breakeven (0 = deaktiviert)
    exit_levels: dict | None = None  # Dynamische Exit-Spalten aus Cache


class SBReplayer(Strategy):
    """Replayed vorberechnete Signale durch die Nautilus-BacktestEngine.

    Platziert keine echten Orders – prüft SL/TP manuell per Bar (High/Low).
    Zählt Wins/Losses für spätere Auswertung durch NautilusBridge.
    """

    def __init__(self, config: SBReplayerConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = BarType.from_str(config.bar_type)
        self.cfg = config

        # Signal-Lookup: minute_ns → (direction, entry, sl, tp, exit_idx)
        self._signal_lookup: dict[int, tuple[int, float, float, float, int]] = {}
        for signal in config.signals:
            if len(signal) == 5:
                ts_ns, direction, entry, sl, tp = signal
                exit_idx = 0
            else:
                ts_ns, direction, entry, sl, tp, exit_idx = signal
            minute_ns = self._to_minute_ns(int(ts_ns))
            self._signal_lookup[minute_ns] = (
                int(direction),
                float(entry),
                float(sl),
                float(tp),
                int(exit_idx),
            )

        # Trade State
        self._in_trade: bool = False
        self._direction: int = 0
        self._entry_price: float = 0.0
        self._sl_price: float = 0.0
        self._tp_price: float = 0.0
        self._sl_dist: float = 0.0
        self._best_price: float = 0.0  # Höchster/Tiefster Preis seit Entry

        # Ergebnis-Tracking
        self.wins: int = 0
        self.losses: int = 0
        self.gross_profit_pts: float = 0.0
        self.gross_loss_pts: float = 0.0
        self.pnl_series: list[float] = []

        # Trade-Level Tracking
        self._entry_ns: int = 0
        self._trade_records: list[TradeRecord] = []
        self._breakeven_reached: bool = False
        self._bar_index: int = 0

    def on_start(self) -> None:
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar) -> None:
        minute_ns = self._to_minute_ns(bar.ts_event)

        if self._in_trade:
            self._check_exit(bar)
            return

        if minute_ns in self._signal_lookup:
            direction, entry, sl, tp, exit_idx = self._signal_lookup[minute_ns]
            self._entry_ns = bar.ts_event
            self._enter_trade(direction, entry, sl, tp, exit_idx)

    def _enter_trade(
        self,
        direction: int,
        entry: float,
        sl: float,
        tp: float,
        exit_idx: int = 0,
    ) -> None:
        self._in_trade = True
        self._direction = direction
        self._entry_price = entry
        self._sl_price = sl
        self._tp_price = tp
        self._sl_dist = abs(entry - sl)
        self._best_price = entry
        self._breakeven_reached = False
        self._bar_index = max(exit_idx, 0)  # Absoluter Cache-Index für Exit-Level-Lookup

    def _update_trailing_stop(self, bar: Bar) -> None:
        """Trail-SL nachziehen – je nach exit_mode."""
        high = float(bar.high)
        low = float(bar.low)
        em = self.cfg.exit_mode

        # Best-Price tracken (für alle Modi die Trail nutzen)
        if self._direction == 1:
            self._best_price = max(self._best_price, high)
        else:
            self._best_price = min(self._best_price, low)

        # --- Breakeven-Check (für breakeven und breakeven_trail) ---
        if em in ("breakeven", "breakeven_trail") and not self._breakeven_reached:
            rr = self.cfg.breakeven_rr
            if rr > 0:
                be_target = self._sl_dist * rr
                profit = self._direction * (self._best_price - self._entry_price)
                if profit >= be_target:
                    # SL auf Entry verschieben (Breakeven)
                    self._sl_price = self._entry_price
                    self._breakeven_reached = True

        # --- Punkt-Trail (fixed + use_trail ODER breakeven_trail nach BE) ---
        if em == "fixed" and self.cfg.trail_activation > 0:
            self._apply_point_trail()
        elif em == "breakeven_trail" and self._breakeven_reached:
            self._apply_point_trail()

        # --- ATR-Trail: SL aus Exit-Level-Spalte ---
        if em == "atr_trail" and self.cfg.exit_levels:
            col = "trail_stop_bull" if self._direction == 1 else "trail_stop_bear"
            levels = self.cfg.exit_levels.get(col)
            if levels and self._bar_index < len(levels):
                new_sl = levels[self._bar_index]
                if new_sl and new_sl == new_sl:  # not NaN
                    if self._direction == 1 and new_sl > self._sl_price:
                        self._sl_price = new_sl
                    elif self._direction == -1 and new_sl < self._sl_price:
                        self._sl_price = new_sl

        # --- Dynamischer TP aus Exit-Levels ---
        if em in ("next_zone", "session_level") and self.cfg.exit_levels:
            col = (
                ("tp_bull_best" if self._direction == 1 else "tp_bear_best")
                if em == "next_zone"
                else ("tp_bull_target" if self._direction == 1 else "tp_bear_target")
            )
            levels = self.cfg.exit_levels.get(col)
            if levels and self._bar_index < len(levels):
                new_tp = levels[self._bar_index]
                if new_tp and new_tp == new_tp:  # not NaN
                    self._tp_price = new_tp

        self._bar_index += 1

    def _apply_point_trail(self) -> None:
        """Einfacher Punkt-Trail (original-Logik)."""
        if self.cfg.trail_activation <= 0:
            return
        if self._direction == 1:  # Long
            if self._best_price - self._entry_price >= self.cfg.trail_activation:
                trail_sl = self._best_price - self.cfg.trail_distance
                if trail_sl > self._sl_price:
                    self._sl_price = trail_sl
        else:  # Short
            if self._entry_price - self._best_price >= self.cfg.trail_activation:
                trail_sl = self._best_price + self.cfg.trail_distance
                if trail_sl < self._sl_price:
                    self._sl_price = trail_sl

    def _check_exit(self, bar: Bar) -> None:
        high = float(bar.high)
        low = float(bar.low)
        open_ = float(bar.open)
        if self._direction == 1:  # Long
            sl_hit = low <= self._sl_price
            tp_hit = high >= self._tp_price
            if sl_hit and tp_hit:
                dist_sl = abs(open_ - self._sl_price)
                dist_tp = abs(open_ - self._tp_price)
                if dist_sl <= dist_tp:
                    self._record_loss(bar)
                else:
                    self._record_win(bar)
            elif sl_hit:
                self._record_loss(bar)
            elif tp_hit:
                self._record_win(bar)
            else:
                self._update_trailing_stop(bar)
        else:  # Short
            sl_hit = high >= self._sl_price
            tp_hit = low <= self._tp_price
            if sl_hit and tp_hit:
                dist_sl = abs(open_ - self._sl_price)
                dist_tp = abs(open_ - self._tp_price)
                if dist_sl <= dist_tp:
                    self._record_loss(bar)
                else:
                    self._record_win(bar)
            elif sl_hit:
                self._record_loss(bar)
            elif tp_hit:
                self._record_win(bar)
            else:
                self._update_trailing_stop(bar)

    def _record_win(self, bar: Bar) -> None:
        friction_pts = (
            self.cfg.slippage_points * 2
            + self.cfg.commission_usd / self.cfg.point_value
        )
        if self.cfg.exit_mode in ("next_zone", "session_level"):
            profit_pts = self._direction * (self._tp_price - self._entry_price) - friction_pts
            exit_price = self._tp_price
        elif self.cfg.tp_mult > 0:
            profit_pts = self._sl_dist * self.cfg.tp_mult - friction_pts
            exit_price = self._tp_price
        else:
            # Kein festes TP → Gewinn = Distanz Entry→SL (Trail-Exit)
            profit_pts = abs(self._sl_price - self._entry_price) - friction_pts
            exit_price = self._sl_price
        if profit_pts <= 0:
            # TP-Treffer aber Friction frisst den Gewinn → als Loss buchen
            self.losses += 1
            self.gross_loss_pts += abs(profit_pts)
            self.pnl_series.append(profit_pts * self.cfg.point_value)
        else:
            self.wins += 1
            self.gross_profit_pts += profit_pts
            self.pnl_series.append(profit_pts * self.cfg.point_value)
        actual_pnl = profit_pts if profit_pts > 0 else -abs(profit_pts)
        self._trade_records.append(
            TradeRecord(
                entry_ns=self._entry_ns,
                exit_ns=bar.ts_event,
                direction=self._direction,
                entry_price=self._entry_price,
                exit_price=exit_price,
                pnl_pts=actual_pnl,
            )
        )
        self._reset_trade()

    def _record_loss(self, bar: Bar) -> None:
        friction_pts = (
            self.cfg.slippage_points * 2
            + self.cfg.commission_usd / self.cfg.point_value
        )
        # Trail hat SL auf Gewinnseite oder auf Entry gezogen → Exit-PnL direkt buchen
        # Short (d=-1): Gewinn wenn sl_price < entry  → d*(sl-entry) = -1*(neg) = pos
        # Long  (d=+1): Gewinn wenn sl_price > entry  → d*(sl-entry) = +1*(pos) = pos
        trail_profit = self._direction * (self._sl_price - self._entry_price)
        if self._direction != 0 and self._entry_price != 0.0 and trail_profit >= 0:
            profit_pts = trail_profit - friction_pts
            if profit_pts > 0:
                self.wins += 1
                self.gross_profit_pts += profit_pts
            else:
                self.losses += 1
                self.gross_loss_pts += abs(profit_pts)
            self.pnl_series.append(profit_pts * self.cfg.point_value)
            self._trade_records.append(
                TradeRecord(
                    entry_ns=self._entry_ns,
                    exit_ns=bar.ts_event,
                    direction=self._direction,
                    entry_price=self._entry_price,
                    exit_price=self._sl_price,
                    pnl_pts=profit_pts,
                )
            )
            self._reset_trade()
            return
        # Normal-Loss
        self.losses += 1
        loss_pts = self._sl_dist + friction_pts
        self.gross_loss_pts += loss_pts
        self.pnl_series.append(-loss_pts * self.cfg.point_value)
        self._trade_records.append(
            TradeRecord(
                entry_ns=self._entry_ns,
                exit_ns=bar.ts_event,
                direction=self._direction,
                entry_price=self._entry_price,
                exit_price=self._sl_price,
                pnl_pts=-loss_pts,
            )
        )
        self._reset_trade()

    def _reset_trade(self) -> None:
        self._in_trade = False
        self._direction = 0
        self._entry_price = 0.0
        self._sl_price = 0.0
        self._tp_price = 0.0
        self._sl_dist = 0.0
        self._best_price = 0.0
        self._breakeven_reached = False
        self._bar_index = 0

    @staticmethod
    def _to_minute_ns(ts_ns: int) -> int:
        """Rundet Nanosekunden-Timestamp auf die volle Minute."""
        ns_per_minute = 60_000_000_000
        return (ts_ns // ns_per_minute) * ns_per_minute
