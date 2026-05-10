"""
Displacement Entry Block
==========================
Enter only after a strong directional candle confirms the move.

Displacement = a candle with:
  - Body > ATR * threshold
  - Close near the extreme (top 20% for bull, bottom 20% for bear)
  - Higher than average volume (optional)

Used after a signal (e.g. FVG) to avoid entering in choppy conditions.
Displacement confirms that institutional participation is present.

Integration in the builder:
    ./sb.py "FVG + Displacement entry + ATR Trail"
"""

import pandas as pd

BLOCK_META = {
    "name": "Displacement Entry",
    "category": "entry",
    "description": "Enter only after a strong institutional candle",
    "tags": ["entry", "displacement", "ICT", "momentum"],
    "params": {
        "body_atr_mult": {
            "type": float,
            "default": 1.2,
            "range": [0.5, 3.0],
            "description": "Minimum candle body as multiple of ATR",
        },
        "close_percentile": {
            "type": float,
            "default": 0.75,
            "range": [0.5, 0.95],
            "description": "Close must be in top/bottom X% of candle range",
        },
        "require_volume": {
            "type": bool,
            "default": False,
            "description": "Require above-average volume for confirmation",
        },
    },
}


def compute(
    bars: pd.DataFrame,
    body_atr_mult: float = 1.2,
    close_percentile: float = 0.75,
    require_volume: bool = False,
) -> pd.Series:
    """
    Args:
        bars: OHLCV DataFrame
    Returns:
        Boolean Series — True where a displacement candle is present
    """
    result = pd.Series(False, index=bars.index)

    # --- implement your displacement logic here ---
    # body = abs(close - open)
    # atr = rolling_atr(bars, 14)
    # bull_displacement = (close > open) and (body > atr * body_atr_mult)
    #                     and ((close - low) / (high - low) > close_percentile)
    # bear_displacement = inverse

    return result
