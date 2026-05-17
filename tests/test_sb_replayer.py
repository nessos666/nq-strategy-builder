from __future__ import annotations

import pandas as pd
import pytest

from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig, SignalEntry


def test_replayer_config_defaults():
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
    )
    assert cfg.sl_points == 10.0
    assert cfg.tp_mult == 2.5
    assert cfg.point_value == 2.0


def test_signal_entry_direction():
    ts = pd.Timestamp("2024-01-15 14:30:00", tz="UTC")
    entry = SignalEntry(
        timestamp=ts,
        direction=1,
        entry_price=19000.0,
        sl_price=18990.0,
        tp_price=19025.0,
    )
    assert entry.direction == 1
    assert entry.sl_price < entry.entry_price
    assert entry.tp_price > entry.entry_price


def test_replayer_instantiates_without_error():
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
    )
    replayer = SBReplayer(config=cfg)
    assert replayer.wins == 0
    assert replayer.losses == 0


def test_check_exit_ambiguous_bar_open_near_tp_wins():
    """Wenn bar.open näher am TP als am SL liegt, soll bei ambiguer Bar TP gewinnen."""
    from unittest.mock import MagicMock
    from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.0,
        point_value=2.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._sl_price = 18990.0
    replayer._tp_price = 19020.0
    replayer._sl_dist = 10.0

    # Bar trifft beide: low=18988 (SL hit), high=19025 (TP hit)
    # open=19015 → näher am TP (19020) als am SL (18990) → TP gewinnt
    bar = MagicMock()
    bar.open = MagicMock(return_value=None)
    bar.open.__float__ = lambda s: 19015.0
    bar.high = MagicMock()
    bar.high.__float__ = lambda s: 19025.0
    bar.low = MagicMock()
    bar.low.__float__ = lambda s: 18988.0

    replayer._check_exit(bar)
    assert replayer.wins == 1
    assert replayer.losses == 0


def test_check_exit_ambiguous_bar_open_near_sl_loses():
    """Wenn bar.open näher am SL liegt, soll bei ambiguer Bar SL gewinnen."""
    from unittest.mock import MagicMock
    from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.0,
        point_value=2.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._sl_price = 18990.0
    replayer._tp_price = 19020.0
    replayer._sl_dist = 10.0

    # open=18993 → näher am SL (18990) als am TP (19020) → SL gewinnt
    bar = MagicMock()
    bar.open = MagicMock(return_value=None)
    bar.open.__float__ = lambda s: 18993.0
    bar.high = MagicMock()
    bar.high.__float__ = lambda s: 19025.0
    bar.low = MagicMock()
    bar.low.__float__ = lambda s: 18988.0

    replayer._check_exit(bar)
    assert replayer.wins == 0
    assert replayer.losses == 1


def test_replayer_config_slippage_defaults():
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
    )
    assert cfg.slippage_points == pytest.approx(0.5)
    assert cfg.commission_usd == pytest.approx(0.70)


def test_record_win_deducts_friction():
    """Win-PnL muss um Friction (Slippage + Kommission) reduziert sein."""
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.5,
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
    )
    from unittest.mock import MagicMock

    replayer = SBReplayer(config=cfg)
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 0

    replayer._record_win(_bar)

    # friction = 0.5*2 + 0.70/2.0 = 1.35 Punkte
    # profit_pts = 10.0 * 2.5 - 1.35 = 23.65
    assert replayer.gross_profit_pts == pytest.approx(23.65)
    assert replayer.pnl_series[0] == pytest.approx(47.30)  # 23.65 * 2.0 USD


def test_record_loss_adds_friction():
    """Loss-PnL muss um Friction erhöht sein (Verlust ist größer)."""
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.5,
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
    )
    from unittest.mock import MagicMock

    replayer = SBReplayer(config=cfg)
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 0

    replayer._record_loss(_bar)

    # friction = 1.35 Punkte
    # loss_pts = 10.0 + 1.35 = 11.35
    assert replayer.gross_loss_pts == pytest.approx(11.35)
    assert replayer.pnl_series[0] == pytest.approx(-22.70)  # -11.35 * 2.0 USD


def test_record_win_with_tiny_sl_counts_as_loss():
    """TP-Treffer mit sl <= friction wird als Loss gezählt, nicht als Win."""
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=0.1,
        tp_mult=2.5,
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
    )
    from unittest.mock import MagicMock

    replayer = SBReplayer(config=cfg)
    replayer._sl_dist = 0.1  # sl_dist * tp_mult = 0.25 < friction 1.35 → negativ
    _bar = MagicMock()
    _bar.ts_event = 0

    replayer._record_win(_bar)

    # profit_pts = 0.1 * 2.5 - 1.35 = -1.1 → negativ → als Loss zählen
    assert replayer.wins == 0
    assert replayer.losses == 1
    assert replayer.gross_profit_pts == pytest.approx(0.0)
    assert replayer.gross_loss_pts == pytest.approx(1.1)
    assert replayer.pnl_series[0] == pytest.approx(-2.20)


def test_record_win_breakeven_counts_as_loss():
    """TP-Treffer mit profit_pts == 0 (Break-even) zählt als Loss – kein Gewinn = kein Win."""
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=0.54,
        tp_mult=2.5,
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
    )
    from unittest.mock import MagicMock

    replayer = SBReplayer(config=cfg)
    replayer._sl_dist = 0.54  # 0.54 * 2.5 - 1.35 = 0.0 → Break-even
    _bar = MagicMock()
    _bar.ts_event = 0

    replayer._record_win(_bar)

    assert replayer.wins == 0
    assert replayer.losses == 1
    assert replayer.pnl_series[0] == pytest.approx(0.0)


def test_replayer_records_win_trade():
    from unittest.mock import MagicMock
    from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.0,
        point_value=2.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 19000.0
    replayer._sl_price = 18990.0
    replayer._tp_price = 19020.0
    replayer._sl_dist = 10.0
    replayer._entry_ns = 1_700_000_000_000_000_000
    bar = MagicMock()
    bar.high.__float__ = lambda s: 19025.0
    bar.low.__float__ = lambda s: 18995.0
    bar.open.__float__ = lambda s: 19010.0
    bar.ts_event = 1_700_000_060_000_000_000
    replayer._check_exit(bar)
    assert replayer.wins == 1
    assert len(replayer._trade_records) == 1
    tr = replayer._trade_records[0]
    assert tr.direction == 1
    assert tr.entry_price == 19000.0
    assert tr.exit_price == 19020.0
    assert tr.entry_ns == 1_700_000_000_000_000_000
    assert tr.exit_ns == 1_700_000_060_000_000_000
    assert tr.pnl_pts > 0


def test_replayer_records_loss_trade():
    from unittest.mock import MagicMock
    from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        sl_points=10.0,
        tp_mult=2.0,
        point_value=2.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 19000.0
    replayer._sl_price = 18990.0
    replayer._tp_price = 19020.0
    replayer._sl_dist = 10.0
    replayer._entry_ns = 1_700_000_000_000_000_000
    bar = MagicMock()
    bar.high.__float__ = lambda s: 19005.0
    bar.low.__float__ = lambda s: 18985.0
    bar.open.__float__ = lambda s: 18993.0
    bar.ts_event = 1_700_000_060_000_000_000
    replayer._check_exit(bar)
    assert replayer.losses == 1
    assert len(replayer._trade_records) == 1
    tr = replayer._trade_records[0]
    assert tr.direction == 1
    assert tr.exit_price == 18990.0
    assert tr.pnl_pts < 0


def test_replayer_entry_ns_set_before_record():
    from sb.engine.sb_replayer import SBReplayer, SBReplayerConfig

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
    )
    replayer = SBReplayer(config=cfg)
    assert hasattr(replayer, "_entry_ns")
    assert hasattr(replayer, "_trade_records")
    assert replayer._trade_records == []


def test_record_loss_with_long_trailing_profit_counts_as_win():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
        trail_activation=10.0,
        trail_distance=5.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 20000.0
    replayer._sl_price = 20030.0
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 123

    replayer._record_loss(_bar)

    assert replayer.wins == 1
    assert replayer.losses == 0
    assert replayer.gross_profit_pts == pytest.approx(28.65)
    assert replayer.gross_loss_pts == pytest.approx(0.0)
    assert replayer.pnl_series == [pytest.approx(57.3)]
    assert replayer._trade_records[0].exit_price == pytest.approx(20030.0)
    assert replayer._trade_records[0].pnl_pts == pytest.approx(28.65)


def test_record_loss_with_short_trailing_profit_counts_as_win():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
        trail_activation=10.0,
        trail_distance=5.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = -1
    replayer._entry_price = 20000.0
    replayer._sl_price = 19970.0
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 456

    replayer._record_loss(_bar)

    assert replayer.wins == 1
    assert replayer.losses == 0
    assert replayer.gross_profit_pts == pytest.approx(28.65)
    assert replayer.gross_loss_pts == pytest.approx(0.0)
    assert replayer.pnl_series == [pytest.approx(57.3)]
    assert replayer._trade_records[0].exit_price == pytest.approx(19970.0)
    assert replayer._trade_records[0].pnl_pts == pytest.approx(28.65)


def test_record_loss_with_trailing_breakeven_counts_only_friction_loss():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
        trail_activation=10.0,
        trail_distance=10.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 20000.0
    replayer._sl_price = 20000.0
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 789

    replayer._record_loss(_bar)

    assert replayer.wins == 0
    assert replayer.losses == 1
    assert replayer.gross_profit_pts == pytest.approx(0.0)
    assert replayer.gross_loss_pts == pytest.approx(1.35)
    assert replayer.pnl_series == [pytest.approx(-2.7)]
    assert replayer._trade_records[0].exit_price == pytest.approx(20000.0)
    assert replayer._trade_records[0].pnl_pts == pytest.approx(-1.35)


def test_update_trailing_stop_for_short_only_tightens_stop():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        trail_activation=10.0,
        trail_distance=5.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._direction = -1
    replayer._entry_price = 20000.0
    replayer._sl_price = 20010.0
    replayer._best_price = 20000.0

    bar = MagicMock()
    bar.high.__float__ = lambda s: 20002.0
    bar.low.__float__ = lambda s: 19980.0

    replayer._update_trailing_stop(bar)

    assert replayer._best_price == pytest.approx(19980.0)
    assert replayer._sl_price == pytest.approx(19985.0)


def test_record_loss_after_breakeven_only_books_friction():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
        exit_mode="breakeven",
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 20000.0
    replayer._sl_price = 20000.0
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 999

    replayer._record_loss(_bar)

    assert replayer.wins == 0
    assert replayer.losses == 1
    assert replayer.gross_loss_pts == pytest.approx(1.35)
    assert replayer.pnl_series == [pytest.approx(-2.7)]


def test_record_win_next_zone_uses_dynamic_tp_distance():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        tp_mult=2.5,
        point_value=2.0,
        slippage_points=0.5,
        commission_usd=0.70,
        exit_mode="next_zone",
    )
    replayer = SBReplayer(config=cfg)
    replayer._direction = 1
    replayer._entry_price = 20000.0
    replayer._tp_price = 20040.0
    replayer._sl_dist = 10.0
    _bar = MagicMock()
    _bar.ts_event = 111

    replayer._record_win(_bar)

    assert replayer.gross_profit_pts == pytest.approx(38.65)
    assert replayer._trade_records[0].exit_price == pytest.approx(20040.0)


def test_signal_exit_index_seeds_exit_level_alignment():
    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=((1_700_000_000_000_000_000, 1, 20000.0, 19990.0, 20025.0, 42),),
    )
    replayer = SBReplayer(config=cfg)
    minute_ns = replayer._to_minute_ns(1_700_000_000_000_000_000)
    direction, entry, sl, tp, exit_idx = replayer._signal_lookup[minute_ns]

    replayer._enter_trade(direction, entry, sl, tp, exit_idx)

    assert replayer._bar_index == 42


def test_breakeven_does_not_use_same_bar_high_as_lookahead_for_exit():
    from unittest.mock import MagicMock

    cfg = SBReplayerConfig(
        instrument_id="MNQM6.SIM",
        bar_type="MNQM6.SIM-1-MINUTE-LAST-EXTERNAL",
        signals=(),
        exit_mode="breakeven",
        breakeven_rr=1.0,
    )
    replayer = SBReplayer(config=cfg)
    replayer._in_trade = True
    replayer._direction = 1
    replayer._entry_price = 20000.0
    replayer._sl_price = 19990.0
    replayer._tp_price = 20050.0
    replayer._sl_dist = 10.0
    replayer._best_price = 20000.0

    bar = MagicMock()
    bar.open.__float__ = lambda s: 20000.0
    bar.high.__float__ = lambda s: 20012.0
    bar.low.__float__ = lambda s: 19995.0

    replayer._check_exit(bar)

    assert replayer._in_trade is True
    assert replayer._sl_price == pytest.approx(20000.0)
