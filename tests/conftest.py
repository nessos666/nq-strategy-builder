from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _install_nautilus_trader_stub() -> None:
    if "nautilus_trader" in sys.modules:
        return

    root = types.ModuleType("nautilus_trader")

    config_mod = types.ModuleType("nautilus_trader.config")

    class _ConfigBase:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

        def __init__(self, **kwargs):
            fields: dict[str, object] = {}
            for base in reversed(type(self).__mro__):
                annotations = getattr(base, "__annotations__", {})
                for name in annotations:
                    if name.startswith("_"):
                        continue
                    if hasattr(base, name):
                        fields[name] = getattr(base, name)
            unknown = set(kwargs) - set(fields)
            if unknown:
                names = ", ".join(sorted(unknown))
                raise TypeError(f"Unexpected config fields: {names}")
            for name, default in fields.items():
                setattr(self, name, kwargs.get(name, default))

    class StrategyConfig(_ConfigBase):
        pass

    class LoggingConfig(_ConfigBase):
        log_level: str = "INFO"
        bypass_logging: bool = False

    config_mod.StrategyConfig = StrategyConfig
    config_mod.LoggingConfig = LoggingConfig

    identifiers_mod = types.ModuleType("nautilus_trader.model.identifiers")

    class InstrumentId(str):
        @classmethod
        def from_str(cls, value: str) -> "InstrumentId":
            return cls(value)

    class Venue(str):
        pass

    identifiers_mod.InstrumentId = InstrumentId
    identifiers_mod.Venue = Venue

    data_mod = types.ModuleType("nautilus_trader.model.data")

    class BarAggregation:
        MINUTE = "MINUTE"

    class BarType:
        def __init__(self, instrument_id=None, bar_spec=None):
            self.instrument_id = instrument_id
            self.bar_spec = bar_spec

        @classmethod
        def from_str(cls, value: str) -> "BarType":
            obj = cls()
            obj._value = value
            return obj

        def __str__(self) -> str:
            return getattr(self, "_value", "BAR-TYPE")

    @dataclass
    class BarSpecification:
        step: int
        aggregation: str
        price_type: str

    @dataclass
    class Bar:
        bar_type: BarType
        open: object
        high: object
        low: object
        close: object
        volume: object
        ts_event: int
        ts_init: int

    data_mod.Bar = Bar
    data_mod.BarAggregation = BarAggregation
    data_mod.BarSpecification = BarSpecification
    data_mod.BarType = BarType

    strategy_mod = types.ModuleType("nautilus_trader.trading.strategy")

    class Strategy:
        def __init__(self, config) -> None:
            self.config = config

        def subscribe_bars(self, bar_type) -> None:
            self._subscribed_bar_type = bar_type

    strategy_mod.Strategy = Strategy

    enums_mod = types.ModuleType("nautilus_trader.model.enums")

    class AccountType:
        MARGIN = "MARGIN"

    class OmsType:
        NETTING = "NETTING"

    class PriceType:
        LAST = "LAST"

    enums_mod.AccountType = AccountType
    enums_mod.OmsType = OmsType
    enums_mod.PriceType = PriceType

    currencies_mod = types.ModuleType("nautilus_trader.model.currencies")
    currencies_mod.USD = "USD"

    objects_mod = types.ModuleType("nautilus_trader.model.objects")

    class Money:
        def __init__(self, amount, currency) -> None:
            self.amount = amount
            self.currency = currency

    class Price(float):
        @classmethod
        def from_str(cls, value: str) -> "Price":
            return cls(float(value))

    class Quantity(int):
        @classmethod
        def from_int(cls, value: int) -> "Quantity":
            return cls(value)

    objects_mod.Money = Money
    objects_mod.Price = Price
    objects_mod.Quantity = Quantity

    instruments_mod = types.ModuleType("nautilus_trader.model.instruments")

    class FuturesContract:
        def __init__(self, symbol: str, venue: str, **kwargs) -> None:
            self.symbol = symbol
            self.venue = venue
            self.id = InstrumentId(f"{symbol}.{venue}")
            self.fields = kwargs

        @staticmethod
        def to_dict(contract: "FuturesContract") -> dict[str, object]:
            return {
                "symbol": contract.symbol,
                "venue": contract.venue,
                **contract.fields,
            }

        @classmethod
        def from_dict(cls, data: dict[str, object]) -> "FuturesContract":
            symbol = str(data["symbol"])
            venue = str(data["venue"])
            kwargs = {k: v for k, v in data.items() if k not in {"symbol", "venue"}}
            return cls(symbol=symbol, venue=venue, **kwargs)

    instruments_mod.FuturesContract = FuturesContract

    providers_mod = types.ModuleType("nautilus_trader.test_kit.providers")

    class TestInstrumentProvider:
        @staticmethod
        def future(symbol: str, underlying: str, venue: str, exchange: str):
            return FuturesContract(
                symbol=symbol,
                venue=venue,
                underlying=underlying,
                exchange=exchange,
            )

    providers_mod.TestInstrumentProvider = TestInstrumentProvider

    backtest_mod = types.ModuleType("nautilus_trader.backtest.engine")

    class BacktestEngineConfig:
        def __init__(self, logging=None) -> None:
            self.logging = logging

    class BacktestEngine:
        def __init__(self, config) -> None:
            self.config = config
            self._bars = []
            self._strategies = []

        def add_venue(self, **kwargs) -> None:
            self._venue = kwargs

        def add_instrument(self, instrument) -> None:
            self._instrument = instrument

        def add_data(self, bars) -> None:
            self._bars = list(bars)

        def add_strategy(self, strategy) -> None:
            self._strategies.append(strategy)

        def run(self) -> None:
            for strategy in self._strategies:
                if hasattr(strategy, "on_start"):
                    strategy.on_start()
                for bar in self._bars:
                    strategy.on_bar(bar)

    backtest_mod.BacktestEngine = BacktestEngine
    backtest_mod.BacktestEngineConfig = BacktestEngineConfig

    sys.modules["nautilus_trader"] = root
    sys.modules["nautilus_trader.config"] = config_mod
    sys.modules["nautilus_trader.model.identifiers"] = identifiers_mod
    sys.modules["nautilus_trader.model.data"] = data_mod
    sys.modules["nautilus_trader.trading.strategy"] = strategy_mod
    sys.modules["nautilus_trader.model.enums"] = enums_mod
    sys.modules["nautilus_trader.model.currencies"] = currencies_mod
    sys.modules["nautilus_trader.model.objects"] = objects_mod
    sys.modules["nautilus_trader.model.instruments"] = instruments_mod
    sys.modules["nautilus_trader.test_kit.providers"] = providers_mod
    sys.modules["nautilus_trader.backtest.engine"] = backtest_mod


_install_nautilus_trader_stub()


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Temporäre SQLite DB für Tests."""
    return tmp_path / "test.db"


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Minimales OHLCV Parquet für Backtest-Tests."""
    n = 5000
    dates = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    base = 17000 + np.cumsum(rng.standard_normal(n))
    wick = np.abs(rng.standard_normal(n)) * 5 + 1.0
    open_ = base + rng.standard_normal(n) * 2
    close = base + rng.standard_normal(n) * 2
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 1000, n),
        },
        index=dates,
    )
    path = tmp_path / "test_data.parquet"
    df.to_parquet(path)
    return path
