# NQ Strategy Builder

<div align="center">

```
███╗   ██╗ ██████╗     ███████╗████████╗██████╗  █████╗ ████████╗███████╗ ██████╗██╗   ██╗
████╗  ██║██╔═══██╗    ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔════╝╚██╗ ██╔╝
██╔██╗ ██║██║   ██║    ███████╗   ██║   ██████╔╝███████║   ██║   █████╗  ██║  ███╗ ╚████╔╝
██║╚██╗██║██║▄▄ ██║    ╚════██║   ██║   ██╔══██╗██╔══██║   ██║   ██╔══╝  ██║   ██║  ╚██╔╝
██║ ╚████║╚██████╔╝    ███████║   ██║   ██║  ██║██║  ██║   ██║   ███████╗╚██████╔╝   ██║
╚═╝  ╚═══╝ ╚══▀▀═╝     ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝ ╚═════╝    ╚═╝
                 B U I L D E R
```

**A systematic, anti-overfitting framework for discovering edge in Nasdaq-100 futures.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Market](https://img.shields.io/badge/Market-NQ%2FMNQ%20Futures-orange?style=flat-square)](https://www.cmegroup.com)
[![Framework](https://img.shields.io/badge/Framework-IS%2FOOS%2FHoldout-blue?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen?style=flat-square)]()

</div>

---

## The Problem with Most Backtests

Most traders backtest wrong. They optimize on all available data, get a great-looking equity curve, go live — and lose. The reason: **overfitting**.

This framework enforces a strict research discipline that mirrors how quantitative hedge funds actually work.

```bash
./sb.py "Fair Value Gap with Order Block in NY session"
```

```
Strategy: FVG + OB NY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase       Bars      Trades    Win%    PF      Sharpe
────────────────────────────────────────────────────
IS          432k      187       54.0%   1.61    1.42   ← trained on this
OOS          85k       41       51.2%   1.38    1.18   ← validated here
Holdout      62k       28       50.0%   1.21    0.97   ← never touched until final
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tier: B  |  Degradation IS→OOS: -14.3%  |  run_id: 247
```

---

## The Architecture

```
 You type:   "./sb.py 'FVG + OB NY'"
                      │
           ┌──────────▼──────────┐
           │     NLP Parser      │  "FVG" → signal token
           │                     │  "OB"  → signal token
           │  engine/parser.py   │  "NY"  → session filter
           └──────────┬──────────┘
                      │  structured tokens
           ┌──────────▼──────────┐
           │  Knowledge Engine   │  FVG → your_fvg_block.py
           │                     │  OB  → your_ob_block.py
           │  engine/knowledge.py│  NY  → session_filter=NY
           └──────────┬──────────┘
                      │  algo references + param spaces
           ┌──────────▼──────────┐
           │    Combinator       │  generates N param combinations
           │                     │  via Optuna TPE sampler
           │ engine/kombinator.py│  fvg_min=[3,5,8] × ob_mode=[strict,loose]
           └──────────┬──────────┘
                      │  for each combination:
           ┌──────────▼──────────┐
           │    Signal Cache     │  compute_fvg(bars) → boolean Series
           │                     │  compute_ob(bars)  → boolean Series
           │    sb/cache/        │  cached as parquet → skip on re-run
           └──────────┬──────────┘
                      │  signal columns
           ┌──────────▼──────────┐
           │   Backtest Engine   │  IS → OOS → Holdout
           │                     │  walk-forward splits
           │   sb/backtest.py    │  commission + slippage applied
           └──────────┬──────────┘
                      │  trades DataFrame
           ┌──────────▼──────────┐
           │     Evaluator       │  PF, Sharpe, Win%, MaxDD
           │                     │  Tier assignment (S/A/B/C/F)
           │   sb/evaluator.py   │  degradation check IS→OOS
           └──────────┬──────────┘
                      │  results
           ┌──────────▼──────────┐
           │   SQLite + Report   │  builder.db — every run stored
           │                     │  ranked leaderboard
           │   sb/results.py     │  reproducible via run_id
           └─────────────────────┘
```

---

## Building Blocks — The DLC System

> Think of the Strategy Builder as the **game engine**.
> Building Blocks are the **DLC packs** — modular, plug-and-play, independently testable.

Each block does exactly one thing and exposes a single interface. The engine wires them together.

```
┌─────────────────────────────────────────────────────────────┐
│                    STRATEGY BUILDER (Engine)                 │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ SIGNAL   │  │ CONTEXT  │  │  ENTRY   │  │   EXIT   │   │
│  │   DLC    │→ │   DLC    │→ │   DLC    │→ │   DLC    │   │
│  │          │  │          │  │          │  │          │   │
│  │ FVG      │  │ Regime   │  │ 1st Touch│  │ ATR Trail│   │
│  │ OB       │  │ Session  │  │ 2nd Touch│  │ Breakeven│   │
│  │ Sweep    │  │ HRL/LRL  │  │ Displace │  │ SessionEX│   │
│  │ BOS/MSS  │  │ Trend    │  │ CE       │  │ NextZone │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Pack 1 — Signal Blocks

Signal blocks detect market structures and generate binary signals.

| Block | What it detects | File |
|-------|----------------|------|
| Fair Value Gap (FVG) | Imbalance zones between candles | `signals/fvg.py` |
| Order Block (OB) | Institutional order flow origins | `signals/orderblock.py` |
| Liquidity Sweep | Stop hunt above/below structure | `signals/sweep.py` |
| Break of Structure (BOS) | Confirmed trend shift | `signals/bos.py` |
| Market Structure Shift (MSS) | Aggressive reversal signal | `signals/mss.py` |
| Inverse FVG (iFVG) | Refilled imbalance becomes support/resistance | `signals/ifvg.py` |

### Pack 2 — Context Blocks

Context blocks act as filters. They don't generate entries — they decide *when* the engine is allowed to trade.

| Block | What it filters | File |
|-------|----------------|------|
| Session Filter | NY AM / London / Asia only | `context/session.py` |
| HRL/LRL Regime | High Resistance / Low Resistance Liquidity | `context/hrl_lrl.py` |
| Trend Filter | Higher timeframe trend direction | `context/trend.py` |
| Macro Window | High-volatility news windows | `context/macro.py` |
| Day-of-Week | Mon-Fri bias filter | `context/day_of_week.py` |

### Pack 3 — Entry Blocks

Entry blocks define *how* to get into a position once signal + context agree.

| Block | Entry logic | File |
|-------|-------------|------|
| First Touch | Enter on first price return to zone | `entries/first_touch.py` |
| Second Touch | Wait for confirmation, enter on 2nd test | `entries/second_touch.py` |
| Displacement | Enter only after strong directional candle | `entries/displacement.py` |
| CE (50%) | Enter at the 50% midpoint of a structure | `entries/ce_entry.py` |

### Pack 4 — Exit Blocks

Exit blocks manage the open trade once in position.

| Block | Exit logic | File |
|-------|-----------|------|
| ATR Trail | Dynamic trailing stop based on ATR | `exits/atr_trail.py` |
| Breakeven | Move stop to entry after R:R threshold | `exits/breakeven.py` |
| Next Zone | Target next FVG/OB level | `exits/next_zone.py` |
| Session Exit | Close at fixed session end time | `exits/session_exit.py` |

---

## How to Write Your Own Block

Every block follows the same interface — a single function that takes OHLCV bars and returns a boolean Series.

```python
# building_blocks/signals/your_block.py

import pandas as pd

BLOCK_META = {
    "name": "Your Block Name",
    "category": "signal",          # signal | context | entry | exit
    "description": "What it detects",
    "params": {
        "lookback": {"type": int, "default": 20, "range": [5, 50]},
        "threshold": {"type": float, "default": 0.5, "range": [0.1, 1.0]},
    }
}

def compute(bars: pd.DataFrame, lookback: int = 20, threshold: float = 0.5) -> pd.Series:
    """
    Args:
        bars: DataFrame with columns [open, high, low, close, volume]
              Index: DatetimeTzAware (UTC or ET)
    Returns:
        Boolean Series — True where condition is active
    """
    result = pd.Series(False, index=bars.index)

    # your logic here

    return result
```

That's it. The engine handles everything else — caching, combination, backtest splits, evaluation.

---

## Test Methodology

The framework uses a **three-phase walk-forward** design to prevent overfitting:

```
Timeline ──────────────────────────────────────────────────────►
         │◄──── In-Sample (IS) ────►│◄─ OOS ─►│◄─ Holdout ─►│
         │      Optimize here       │  Validate │  Final test  │
         │      (60% of data)       │  (20%)    │  (20%)       │
         │                          │           │              │
         │  Overfitting dies here ──►           │              │
         │                          │           │              │
         │                          └───────────►              │
         │                            Only if OOS passes       │
         │                                       └─────────────►
         │                                         Locked until
         │                                         final decision
```

**Tier system:**

| Tier | Criteria | Action |
|------|----------|--------|
| S | PF > 2.0, IS→OOS degradation < 10% | Candidate for live |
| A | PF > 1.6, degradation < 15% | Strong, needs more testing |
| B | PF > 1.3, degradation < 20% | Promising, keep researching |
| C | PF > 1.1 | Weak edge, discard |
| F | PF < 1.1 or OOS fails | No edge, discard |

---

## Quick Start

```bash
git clone https://github.com/nessos666/nq-strategy-builder
cd nq-strategy-builder
pip install -r requirements.txt

# Run a test
./sb.py "Fair Value Gap in NY session"

# Combine multiple blocks
./sb.py "FVG + Order Block + NY + ATR Trail"

# Run a full parameter sweep
./sb.py "FVG" --sweep --trials 50

# Show leaderboard
./sb.py --leaderboard
```

---

## Requirements

```
python >= 3.11
pandas
numpy
optuna
vectorbt (or your own backtest engine)
pyarrow
rich
loguru
```

---

## Philosophy

1. **No curve-fitting** — holdout data is locked until the final decision
2. **One idea at a time** — each block does exactly one thing
3. **Combinatorial search** — the engine finds combinations, not you
4. **Every run is saved** — full reproducibility via `run_id`
5. **Degradation is the signal** — a strategy that holds up OOS has real edge

---

## License

MIT — build on it, test your own ideas, publish your results.

---

<div align="center">

*Built for systematic NQ futures research.*
*If you find a block that works — share it.*

**github.com/nessos666**

</div>
