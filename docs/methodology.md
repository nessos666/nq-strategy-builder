# Test Methodology

## Why Three Phases?

Most retail backtests use 100% of available data for both optimization and validation. This guarantees overfitting. A strategy optimized and validated on the same data will always look better than it actually is.

The NQ Strategy Builder enforces a strict three-phase split that mirrors institutional research practice.

## The Split

```
Full data timeline (example: 3 years of NQ 1-min bars ≈ 580k bars)
│
├── Phase 1 — In-Sample (IS): 60% of data
│   Used for: parameter optimization, signal tuning
│   Overfitting is expected and acceptable here.
│
├── Phase 2 — Out-of-Sample (OOS): 20% of data
│   Used for: validation of IS results
│   The IS strategy is run here unchanged.
│   If PF drops >25% from IS: discard.
│
└── Phase 3 — Holdout: 20% of data
    Used for: final decision
    NEVER touched during research.
    Only opened after OOS passes.
```

## Walk-Forward (Optional)

For longer datasets, the engine supports rolling walk-forward validation:

```
Window 1:  [====IS====|=OOS=]
Window 2:       [====IS====|=OOS=]
Window 3:            [====IS====|=OOS=]
Window 4:                 [====IS====|=OOS=|HO]
```

Each window generates a result. A strategy passes walk-forward only if it shows edge across all windows, not just one.

## Degradation Check

The engine computes the IS→OOS degradation:

```
degradation = (PF_IS - PF_OOS) / PF_IS
```

| Degradation | Assessment |
|-------------|-----------|
| < 10% | Excellent — real edge |
| 10–20% | Acceptable |
| 20–30% | Borderline — use caution |
| > 30% | Likely overfit — discard |

## Why NQ Futures?

- Liquid: $5–10B daily volume, tight spreads
- Structured: clear institutional order flow patterns
- Levered: 1 MNQ contract = $2 per point, NQ = $20 per point
- Continuous: near 24h trading (Sunday 18:00 ET to Friday 17:00 ET)
- Commission matters: $12.50/side per NQ contract — every backtest includes this

## What Gets Tested

The framework runs backtests at the trade level, not bar level:

```python
Trade(
    entry_time   = ...,
    entry_price  = ...,
    direction    = "long" | "short",
    stop_loss    = ...,
    take_profit  = ...,    # or trailing stop
    exit_time    = ...,
    exit_price   = ...,
    pnl          = ...,    # after commission + slippage
)
```

Slippage is modeled as 2 ticks (0.5 NQ points) per side — conservative for NQ.

## Metrics

| Metric | Formula | Minimum |
|--------|---------|---------|
| Profit Factor | Gross Profit / Gross Loss | > 1.3 |
| Win Rate | Winning Trades / Total Trades | context-dependent |
| Sharpe Ratio | Annualized (trade-level) | > 0.8 |
| Max Drawdown | Largest peak-to-trough | < 20% of capital |
| Expectancy | Avg P&L per trade | > $50/trade (NQ) |

**Note:** Win Rate alone is meaningless. A 40% WR strategy with 3:1 R:R beats a 60% WR strategy with 1:2 R:R. The framework reports all metrics — PF and Expectancy are the primary signals.
