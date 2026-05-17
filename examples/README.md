# Example Signal Generators

This folder contains 9 example signal generators compatible with the Strategy Builder framework.

Each file implements one or more `compute_*()` functions that accept a pandas DataFrame of OHLCV bars
and return a pandas Series (or DataFrame) with signal values.

## Signal Contract

```python
def compute_my_signal(bars: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Args:
        bars: DataFrame with columns Open, High, Low, Close, Volume
              Index: DatetimeIndex (UTC, 1-minute bars)
    Returns:
        pd.Series with boolean or float values, same index as bars
    """
```

## Included Examples

| File | Concept | Category |
|------|---------|----------|
| `algo_09_fair_value_gap_v2.py` | Fair Value Gap (FVG) detection | ICT/PDA |
| `algo_20_sessions_v2.py` | Asia / London / NY session ranges | Time |
| `algo_21_opening_range_v2.py` | Opening Range (first 30 min) | Time |
| `algo_25_bos_v2.py` | Break of Structure (BOS) | Market Structure |
| `algo_30_displacement_v2.py` | Strong impulsive moves | Momentum |
| `algo_31_premium_discount_v2.py` | Premium / Discount zones | Price Level |
| `algo_33_dealing_range_v2.py` | Weekly dealing range | Price Level |
| `algo_inside_day_v2.py` | Inside Day pattern | Price Pattern |
| `science_hurst_exponent_v2.py` | Hurst Exponent (mean-reversion detector) | Statistical |

## Usage with Strategy Builder

Place your signal generator files in a directory and reference it in `sources.yaml`:

```yaml
pda_library:
  paths:
    - /path/to/your/algos
  pattern: "*.py"
```

Then run:

```bash
./sb.py "FVG + BOS combination"
```
