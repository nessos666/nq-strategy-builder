from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarAggregation, BarSpecification, BarType
from nautilus_trader.model.enums import AccountType, OmsType, PriceType
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.test_kit.providers import TestInstrumentProvider

from sb.cache.cache_query import query_signal_bars, query_signal_bars_roles
from sb.cache.signal_cache import SignalCache, SignalCacheConfig
from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig
from sb.models import BacktestResult

logger = logging.getLogger(__name__)

VENUE_NAME = "SIM"


def _compute_max_drawdown(pnl_series: list[float]) -> float:
    """Berechnet maximalen Drawdown (in Dollar) aus einer PnL-Sequenz."""
    if not pnl_series:
        return 0.0
    peak = 0.0
    cumulative = 0.0
    max_dd = 0.0
    for pnl in pnl_series:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _compute_sharpe(pnl_series: list[float], risk_free: float = 0.0) -> float:
    """Berechnet Sharpe Ratio aus einer PnL-Sequenz (annualisiert nicht)."""
    if len(pnl_series) < 2:
        return 0.0
    n = len(pnl_series)
    mean = sum(pnl_series) / n
    variance = sum((x - mean) ** 2 for x in pnl_series) / (n - 1)
    std = variance**0.5
    if std == 0.0:
        return 0.0
    return (mean - risk_free) / std


STARTING_CAPITAL = 100_000.0
POINT_VALUE = 2.0  # MNQ: $2/Punkt
NS_PER_MINUTE = 60_000_000_000
SLIPPAGE_POINTS = 0.5  # 0.5 Punkte pro Seite (NQ 1-Min Schätzung)
COMMISSION_USD = 0.70  # Rithmic MNQ Round-Trip Kommission

# ICT Kill Zone UTC-Stunden (start_inclusive, end_exclusive)
# Halbe Stunden als Dezimal: 13.5 = 13:30 UTC
_SESSION_HOURS: dict[str, tuple[float, float]] = {
    "london": (7.0, 11.0),
    "ny": (13.5, 16.0),
    "asia": (1.0, 4.0),
}


def _filter_by_session(timestamps: pd.Index, session: str) -> pd.Index:
    """Filtert Timestamps nach UTC-Stunden der Session. 'all' = kein Filter."""
    if session == "all" or session not in _SESSION_HOURS:
        return timestamps
    if not isinstance(timestamps, pd.DatetimeIndex) or len(timestamps) == 0:
        return timestamps
    start, end = _SESSION_HOURS[session]
    if timestamps.tz is None:  # type: ignore[union-attr]
        ts_utc = timestamps.tz_localize("UTC")  # type: ignore[union-attr]
    else:
        ts_utc = timestamps.tz_convert("UTC")  # type: ignore[union-attr]
    hours = ts_utc.hour + ts_utc.minute / 60.0  # type: ignore[union-attr]
    mask = (hours >= start) & (hours < end)
    return timestamps[mask]


def _build_mnq_instrument(venue: Venue) -> FuturesContract:
    """Baut MNQ FuturesContract mit korrekter Expiration und Multiplier."""
    mnq_base = TestInstrumentProvider.future(
        symbol="MNQM6", underlying="MNQ", venue=str(venue), exchange="XCME"
    )
    d = FuturesContract.to_dict(mnq_base)
    d["expiration_ns"] = int(pd.Timestamp("2027-06-18", tz="UTC").value)
    d["activation_ns"] = int(pd.Timestamp("2023-01-01", tz="UTC").value)
    d["multiplier"] = "2"
    d["price_precision"] = 2
    d["price_increment"] = "0.25"
    return FuturesContract.from_dict(d)


class NautilusBridge:
    """Ersetzt BacktestBridge – nutzt echte Nautilus BacktestEngine.

    Einmal laden (Bars + Cache), dann run() beliebig oft aufrufen.
    """

    def __init__(
        self,
        data_path: Path,
        cache_path: Path,
        concepts: list[str] | None = None,
        algo_dirs: list[Path] | None = None,
        algo_pattern: str = "*.py",
        max_date: str | None = None,
        min_date: str | None = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.cache_path = Path(cache_path)
        self._concepts = concepts
        self._algo_dirs = [Path(d) for d in algo_dirs] if algo_dirs else None
        self._algo_pattern = algo_pattern
        self._max_date = max_date
        self._min_date = min_date
        self._bars: list[Bar] = []
        self._cache_df: pd.DataFrame | None = None
        self._instrument: FuturesContract | None = None
        self._bar_type: BarType | None = None
        self._venue = Venue(VENUE_NAME)
        self._load()

    @classmethod
    def from_bars(
        cls,
        bars: list[Bar],
        cache_df: pd.DataFrame | None,
        instrument: FuturesContract | None,
        bar_type: BarType | None,
        venue: Venue | None,
    ) -> "NautilusBridge":
        """Erstellt NautilusBridge mit vorgegebenen Bars – kein Parquet-Laden."""
        obj = object.__new__(cls)
        obj.data_path = Path(".")
        obj.cache_path = Path(".")
        obj._bars = bars
        obj._cache_df = cache_df
        obj._instrument = instrument
        obj._bar_type = bar_type
        obj._venue = venue if venue is not None else Venue(VENUE_NAME)
        return obj

    # ── Public ──────────────────────────────────────────────────────────────

    def run(self, params: dict) -> BacktestResult:
        if not self._bars or self._instrument is None:
            return BacktestResult(
                params=params,
                gross_profit=0.0,
                gross_loss=0.0,
                num_trades=0,
                num_wins=0,
            )

        # ATR-Modus wenn atr_mult in params und nicht None, sonst legacy sl_points
        _raw_atr_mult = params.get("atr_mult")
        atr_mult: float | None = (
            float(_raw_atr_mult) if _raw_atr_mult is not None else None
        )
        sl_points = float(params.get("sl_points", 10.0))
        tp_mult = float(params.get("tp_mult", 2.5))
        offset = max(int(params.get("entry_bar_offset", 1)), 1)
        concepts = list(params.get("concepts", []))
        direction = int(params.get("direction", 1))

        # Rollen-bewusste Abfrage wenn entry-Feld gesetzt, sonst legacy
        entry = list(params.get("entry", []))
        if self._cache_df is not None and entry:
            zone = list(params.get("zone", []))
            context = list(params.get("context", []))
            timing = list(params.get("timing", []))
            timestamps = query_signal_bars_roles(
                self._cache_df, entry, zone, context, timing
            )
        elif self._cache_df is not None:
            concepts = list(params.get("concepts", []))
            if not concepts:
                return BacktestResult(
                    params=params,
                    gross_profit=0.0,
                    gross_loss=0.0,
                    num_trades=0,
                    num_wins=0,
                )
            timestamps = query_signal_bars(self._cache_df, concepts, mode="any")
        else:
            return BacktestResult(
                params=params,
                gross_profit=0.0,
                gross_loss=0.0,
                num_trades=0,
                num_wins=0,
            )

        session = str(params.get("session", "all"))

        # Zone-Levels für OB_KL: SL aus Zone-Boundary statt ATR/fixed
        zone_levels: pd.DataFrame | None = None
        if self._cache_df is not None and any(
            z.lower() == "ob_kl" for z in list(params.get("zone", []))
        ):
            level_cols = [
                c for c in ["ob_kl_high", "ob_kl_low"] if c in self._cache_df.columns
            ]
            if len(level_cols) == 2:
                zone_levels = self._cache_df[level_cols]

        signals = self._build_signals(
            timestamps,
            sl_points,
            tp_mult,
            offset,
            direction,
            session,
            atr_mult=atr_mult,
            zone_levels=zone_levels,
            sl_buffer=float(params.get("sl_buffer", 5.0)),
        )
        if not signals:
            return BacktestResult(
                params=params,
                gross_profit=0.0,
                gross_loss=0.0,
                num_trades=0,
                num_wins=0,
            )

        engine_sl = atr_mult if atr_mult is not None else sl_points
        trail_act = float(params.get("trail_activation", 0.0))
        trail_dist = float(params.get("trail_distance", 0.0))
        exit_mode = str(params.get("exit_mode", "fixed"))
        breakeven_rr = float(params.get("breakeven_rr", 0.0))

        # Exit-Spalten aus Cache laden (für atr_trail, next_zone, session_level)
        exit_levels: dict[str, list[float]] | None = None
        if self._cache_df is not None and exit_mode in (
            "atr_trail",
            "next_zone",
            "session_level",
        ):
            _EXIT_COLS: dict[str, list[str]] = {
                "atr_trail": ["trail_stop_bull", "trail_stop_bear"],
                "next_zone": ["tp_bull_best", "tp_bear_best"],
                "session_level": ["tp_bull_target", "tp_bear_target"],
            }
            needed = _EXIT_COLS.get(exit_mode, [])
            found = [c for c in needed if c in self._cache_df.columns]
            if found:
                exit_levels = {c: self._cache_df[c].values.tolist() for c in found}

        return self._run_engine(
            signals,
            engine_sl,
            tp_mult,
            params,
            trail_act,
            trail_dist,
            exit_mode=exit_mode,
            breakeven_rr=breakeven_rr,
            exit_levels=exit_levels,
        )

    # ── Private ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Lädt Bar-Daten und Signal-Cache."""
        if not self.data_path.exists():
            return
        try:
            df = pd.read_parquet(self.data_path)
            df.columns = [c.capitalize() for c in df.columns]
            if df.index.tz is None:  # type: ignore[union-attr]
                df.index = df.index.tz_localize("UTC")  # type: ignore[union-attr]
            else:
                df.index = df.index.tz_convert("UTC")  # type: ignore[union-attr]

            if self._max_date:
                cutoff = pd.Timestamp(self._max_date, tz="UTC")
                df = df[df.index < cutoff]
            if self._min_date:
                start = pd.Timestamp(self._min_date, tz="UTC")
                df = df[df.index >= start]

            instrument = _build_mnq_instrument(self._venue)
            bar_type = BarType(
                instrument_id=instrument.id,
                bar_spec=BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST),
            )
            self._instrument = instrument
            self._bar_type = bar_type

            bars: list[Bar] = []
            for ts, row in df.iterrows():
                ts_ns = int(pd.Timestamp(ts).value)  # type: ignore[arg-type]
                bars.append(
                    Bar(
                        bar_type=bar_type,
                        open=Price.from_str(f"{float(row['Open']):.2f}"),
                        high=Price.from_str(f"{float(row['High']):.2f}"),
                        low=Price.from_str(f"{float(row['Low']):.2f}"),
                        close=Price.from_str(f"{float(row['Close']):.2f}"),
                        volume=Quantity.from_int(max(1, int(row.get("Volume", 1)))),  # type: ignore[arg-type]
                        ts_event=ts_ns,
                        ts_init=ts_ns,
                    )
                )
            self._bars = bars
            logger.info("NautilusBridge: %d Bars geladen", len(bars))
        except Exception as exc:
            logger.error("NautilusBridge Ladefehler: %s", exc)

        # Signal-Cache laden: erst Shards versuchen, dann monolithisches Parquet
        shard_dir = self.cache_path.parent / "signal_shards"
        if shard_dir.exists() or (self._algo_dirs is not None):
            try:
                sc_cfg = SignalCacheConfig(
                    bars_path=self.data_path,
                    cache_path=self.cache_path,
                    algo_dirs=self._algo_dirs or [],
                    algo_pattern=self._algo_pattern,
                )
                sc = SignalCache(sc_cfg)
                if self._algo_dirs:
                    sc.build(concepts=self._concepts, force=False)
                cache_df = sc.load(concepts=self._concepts)
                if not cache_df.empty:
                    self._cache_df = cache_df
            except Exception as exc:
                logger.error("Signal-Cache (Shards) Ladefehler: %s", exc)
        elif self.cache_path.exists():
            # Fallback: altes monolithisches Parquet
            try:
                df = pd.read_parquet(self.cache_path)
                # Prüfe ob es ein echtes Cache-Parquet ist (hat bool-Spalten)
                bool_cols = [c for c in df.columns if df[c].dtype == bool]
                if bool_cols:
                    self._cache_df = df
            except Exception as exc:
                logger.error("Signal-Cache (Monolith) Ladefehler: %s", exc)

    def _compute_atr_at(self, idx: int, window: int = 14) -> float:
        """Berechnet ATR(window) als Simple MA der True Range bis Bar idx (inklusiv)."""
        if idx < 1 or not self._bars:
            return 10.0
        start = max(1, idx - window + 1)
        tr_sum = 0.0
        count = 0
        for i in range(start, min(idx + 1, len(self._bars))):
            h = float(self._bars[i].high)
            low = float(self._bars[i].low)
            prev_c = float(self._bars[i - 1].close)
            tr = max(h - low, abs(h - prev_c), abs(low - prev_c))
            tr_sum += tr
            count += 1
        return tr_sum / count if count > 0 else 10.0

    def _build_signals(
        self,
        timestamps: pd.Index,
        sl_points: float,
        tp_mult: float,
        offset: int,
        direction: int,
        session: str = "all",
        atr_mult: float | None = None,
        zone_levels: pd.DataFrame | None = None,
        sl_buffer: float = 5.0,
    ) -> list[tuple]:
        """Baut Signals als (ts_ns, direction, entry, sl, tp, exit_idx) Tupel.

        ATR-Modus: atr_mult gesetzt → sl = ATR(14) × atr_mult pro Signal.
        Zone-Modus: zone_levels gesetzt → sl = Zone-Boundary ± sl_buffer.
        Legacy-Modus: atr_mult=None → fixer sl_points.
        """
        timestamps = _filter_by_session(timestamps, session)
        # Bar-Index: minute_ns → Bar-Index in self._bars
        bar_by_minute: dict[int, int] = {}
        for i, bar in enumerate(self._bars):
            minute_ns = (bar.ts_event // NS_PER_MINUTE) * NS_PER_MINUTE
            bar_by_minute[minute_ns] = i

        signals: list[tuple] = []
        for ts in timestamps:
            ts_ns = int(pd.Timestamp(ts).value)  # type: ignore[arg-type]
            minute_ns = (ts_ns // NS_PER_MINUTE) * NS_PER_MINUTE
            idx = bar_by_minute.get(minute_ns)
            if idx is None:
                continue
            entry_idx = min(idx + offset, len(self._bars) - 1)
            entry_bar = self._bars[entry_idx]
            entry_price = float(entry_bar.open)

            if atr_mult is not None:
                atr_val = self._compute_atr_at(entry_idx)
                if not (atr_val > 0.0):  # guard: ATR==0 oder NaN → Skip
                    continue
                effective_sl = atr_val * atr_mult
            else:
                effective_sl = sl_points

            dirs = [1, -1] if direction == 0 else [direction]
            for d in dirs:
                # Zone-Modus: SL = Zone-Boundary ± Buffer
                if zone_levels is not None and ts in zone_levels.index:
                    z_high = float(zone_levels.at[ts, "ob_kl_high"])
                    z_low = float(zone_levels.at[ts, "ob_kl_low"])
                    if d == 1:  # Long → SL unter Zone
                        sl_price = z_low - sl_buffer
                    else:  # Short → SL über Zone
                        sl_price = z_high + sl_buffer
                    effective_sl = abs(entry_price - sl_price)
                else:
                    sl_price = entry_price - d * effective_sl
                tp_price = entry_price + d * effective_sl * tp_mult
                signals.append(
                    (
                        entry_bar.ts_event,
                        d,
                        entry_price,
                        sl_price,
                        tp_price,
                        min(entry_idx + 1, len(self._bars)),
                    )
                )

        return signals

    def _run_engine(
        self,
        signals: list[tuple],
        sl_points: float,
        tp_mult: float,
        params: dict,
        trail_activation: float = 0.0,
        trail_distance: float = 0.0,
        exit_mode: str = "fixed",
        breakeven_rr: float = 0.0,
        exit_levels: dict[str, list[float]] | None = None,
    ) -> BacktestResult:
        """Baut BacktestEngine, führt SBReplayer aus, gibt BacktestResult zurück."""
        assert self._instrument is not None
        assert self._bar_type is not None

        engine_cfg = BacktestEngineConfig(
            logging=LoggingConfig(log_level="ERROR", bypass_logging=True),
        )
        engine = BacktestEngine(config=engine_cfg)
        engine.add_venue(
            venue=self._venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(STARTING_CAPITAL, USD)],
        )
        engine.add_instrument(self._instrument)
        engine.add_data(self._bars)

        replayer_cfg = SBReplayerConfig(
            instrument_id=str(self._instrument.id),
            bar_type=str(self._bar_type),
            signals=tuple(signals),
            sl_points=sl_points,
            tp_mult=tp_mult,
            point_value=POINT_VALUE,
            slippage_points=SLIPPAGE_POINTS,
            commission_usd=COMMISSION_USD,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            exit_mode=exit_mode,
            breakeven_rr=breakeven_rr,
            exit_levels=exit_levels,
        )
        replayer = SBReplayer(config=replayer_cfg)
        engine.add_strategy(replayer)
        engine.run()

        gp = replayer.gross_profit_pts * POINT_VALUE
        gl = replayer.gross_loss_pts * POINT_VALUE

        return BacktestResult(
            params=params,
            gross_profit=round(gp, 2),
            gross_loss=round(gl, 2),
            num_trades=replayer.wins + replayer.losses,
            num_wins=replayer.wins,
            max_drawdown=round(_compute_max_drawdown(replayer.pnl_series), 2),
            sharpe_ratio=round(_compute_sharpe(replayer.pnl_series), 3),
            pnl_series=list(replayer.pnl_series),
            raw_trades=list(replayer._trade_records),
        )
