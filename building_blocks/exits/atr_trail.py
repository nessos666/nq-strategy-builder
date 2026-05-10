"""
ATR Trailing Stop (Exit Block)
================================
Dynamic trailing stop based on Average True Range (ATR).

After entry, the stop is placed at:
    Long:  entry - (ATR * multiplier)
    Short: entry + (ATR * multiplier)

The stop only moves in the direction of profit — never against it.
This allows winners to run while cutting losers at a defined risk level.

Integration in the builder:
    ./sb.py "FVG + NY + ATR Trail"
"""

import pandas as pd

BLOCK_META = {
    "name": "ATR Trailing Stop",
    "category": "exit",
    "description": "Dynamic trailing stop using Average True Range",
    "tags": ["exit", "trail", "ATR", "risk management"],
    "params": {
        "atr_period": {
            "type": int,
            "default": 14,
            "range": [5, 30],
            "description": "ATR calculation period",
        },
        "atr_multiplier": {
            "type": float,
            "default": 1.5,
            "range": [0.5, 4.0],
            "description": "ATR multiplier for initial stop distance",
        },
        "session_exit": {
            "type": bool,
            "default": True,
            "description": "Force close at session end regardless of trail",
        },
    },
}


def compute_stop(
    bars: pd.DataFrame,
    entry_price: float,
    direction: str,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
) -> pd.Series:
    """
    Args:
        bars: OHLCV DataFrame from entry bar onward
        entry_price: Price at which position was entered
        direction: "long" or "short"
    Returns:
        Series of stop prices for each bar (trail moves, never reverses)
    """
    # --- implement your ATR trail here ---
    # 1. Compute ATR over atr_period
    # 2. Set initial stop = entry ± ATR * multiplier
    # 3. Each bar: if long, stop = max(prev_stop, high - ATR*mult)
    #              if short: stop = min(prev_stop, low + ATR*mult)
    stops = pd.Series(entry_price, index=bars.index)
    return stops
