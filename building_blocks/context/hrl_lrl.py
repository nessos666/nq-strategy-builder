"""
HRL / LRL Regime Filter (Context Block)
=========================================
High Resistance Liquidity (HRL) vs Low Resistance Liquidity (LRL)

This block classifies the current market regime based on how price
interacts with liquidity pools and structural zones on a higher timeframe.

HRL = price is fighting against strong liquidity resistance → avoid trades
LRL = price is moving through thin liquidity → favorable conditions

Used as a FILTER: when regime = LRL, allow trades. When HRL, skip.

Integration in the builder:
    ./sb.py "FVG + LRL context + NY"
"""

import pandas as pd

BLOCK_META = {
    "name": "HRL/LRL Regime Filter",
    "category": "context",
    "description": "High/Low Resistance Liquidity regime classification",
    "tags": ["ICT", "regime", "liquidity", "context"],
    "params": {
        "context_timeframe": {
            "type": str,
            "default": "15min",
            "options": ["5min", "15min", "1h"],
            "description": "Higher timeframe for regime assessment",
        },
        "lookback": {
            "type": int,
            "default": 20,
            "range": [10, 50],
            "description": "Bars to look back for regime classification",
        },
        "lrl_threshold": {
            "type": float,
            "default": 0.80,
            "range": [0.60, 0.95],
            "description": "Below this → LRL (favorable), above → HRL",
        },
    },
}


def compute(
    bars: pd.DataFrame,
    context_timeframe: str = "15min",
    lookback: int = 20,
    lrl_threshold: float = 0.80,
) -> pd.Series:
    """
    Args:
        bars: OHLCV DataFrame (execution timeframe)
    Returns:
        Boolean Series — True where regime = LRL (safe to trade)
    """
    result = pd.Series(False, index=bars.index)

    # --- implement your HRL/LRL logic here ---
    # Resample to context_timeframe, compute regime metric,
    # forward-fill back to execution timeframe.
    #
    # Example approach:
    #   ctx = bars.resample(context_timeframe).agg({...})
    #   regime_score = ctx.apply(your_regime_metric, axis=1)
    #   lrl_mask = regime_score < lrl_threshold
    #   result = lrl_mask.reindex(bars.index, method="ffill").fillna(False)

    return result
