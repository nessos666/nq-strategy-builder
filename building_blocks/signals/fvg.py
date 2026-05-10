"""
Fair Value Gap (FVG) Signal Block
==================================
Detects imbalance zones where price moved too fast,
leaving a gap between candle bodies that price tends to revisit.

A bullish FVG forms when:
    low[i] > high[i-2]   (gap between candle i-2's high and candle i's low)

A bearish FVG forms when:
    high[i] < low[i-2]   (gap between candle i-2's low and candle i's high)
"""

import pandas as pd

BLOCK_META = {
    "name": "Fair Value Gap",
    "category": "signal",
    "description": "Detects bullish/bearish imbalance zones (FVG)",
    "tags": ["ICT", "price action", "imbalance"],
    "params": {
        "min_size": {
            "type": float,
            "default": 3.0,
            "range": [1.0, 20.0],
            "description": "Minimum FVG size in NQ points",
        },
        "direction": {
            "type": str,
            "default": "both",
            "options": ["bull", "bear", "both"],
            "description": "Which FVG direction to detect",
        },
        "max_age_bars": {
            "type": int,
            "default": 50,
            "range": [10, 200],
            "description": "FVG expires after N bars (0 = never)",
        },
    },
}


def compute(
    bars: pd.DataFrame,
    min_size: float = 3.0,
    direction: str = "both",
    max_age_bars: int = 50,
) -> pd.Series:
    """
    Args:
        bars: OHLCV DataFrame with DatetimeTzAware index
    Returns:
        Boolean Series — True where an active FVG is present
    """
    result = pd.Series(False, index=bars.index)

    # --- implement your FVG logic here ---
    # Example skeleton:
    #
    # for i in range(2, len(bars)):
    #     bull_fvg = bars["low"].iloc[i] > bars["high"].iloc[i - 2]
    #     bear_fvg = bars["high"].iloc[i] < bars["low"].iloc[i - 2]
    #     gap_size = abs(bars["low"].iloc[i] - bars["high"].iloc[i - 2])
    #
    #     if gap_size >= min_size:
    #         if direction in ("bull", "both") and bull_fvg:
    #             result.iloc[i] = True
    #         if direction in ("bear", "both") and bear_fvg:
    #             result.iloc[i] = True

    return result
