<p align="center">

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
*Modular building blocks · NLP-driven strategy composition · Walk-forward validation · 60 tests*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Market](https://img.shields.io/badge/Market-NQ%2FMNQ%20Futures-orange?style=flat-square)](https://www.cmegroup.com)
[![Tests](https://img.shields.io/badge/Tests-60_suites-brightgreen?style=flat-square)]()
[![Stars](https://img.shields.io/github/stars/nessos666/nq-strategy-builder?style=social)]()

</p>

---

## What is This?

NQ Strategy Builder is a **research framework for algorithmic futures trading** that forces you to test ideas correctly.

Most traders backtest wrong: optimize on all available data, get a great-looking equity curve, go live — and lose. The reason is **overfitting**.

This framework enforces the same research discipline used by quantitative hedge funds:

1. **Split data into three periods** — IS (train), OOS (validate), Holdout (final verify)
2. **Describe strategies in plain English** — the NLP parser converts your idea into signal modules
3. **Search parameter spaces systematically** — Optuna finds combinations, not guesswork
4. **Grade every result** — S/A/B/C/F tier system based on degradation from IS to OOS
5. **Reproduce everything** — every run is saved in SQLite with a unique `run_id`

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

## Why This Exists

I spent years building and testing NQ futures strategies. Most frameworks are either:

- **Too simplistic** — single backtest, no walk-forward, no overfitting checks
- **Too academic** — complex APIs, steep learning curve, no practical workflow
- **Black-box** — you don't know what's happening under the hood

This framework is the middle ground: **correct math with a practical CLI**. Type a strategy idea in plain English, get a tier-graded result with IS/OOS/Holdout separation, and know whether your edge is real or just noise.

---

## Architecture

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

### Engine Components (150+ files)

| Component | Files | Purpose |
|-----------|-------|---------|
| `sb/engine/` | 9 modules | Parser, knowledge, combinator, evaluator, walk-forward, meta-learner, worker |
| `sb/cache/` | 4 modules | Signal caching (parquet-based, skip on re-run) |
| `sb/memory/` | 1 module | SQLite-backed run storage with full reproducibility |
| `sb/filters/` | 1 module | News/macro event filter |
| `sb/research/` | 1 module | Entry statistics and analysis |
| `helfer/` | 9 modules | Cache guard, quality gate, batch pilot, status board |
| `scripts/` | 18 scripts | Research automation, backfill, param sensitivity, settlement analysis |
| `tests/` | 60 test files | Full coverage of engine, cache, filters, research |

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

### Available Building Blocks

| Pack | Blocks | What They Do |
|------|--------|-------------|
| **Signal** | FVG, Order Block, Liquidity Sweep, BOS, MSS, iFVG | Detect market structures and generate binary signals |
| **Context** | Session Filter, HRL/LRL Regime, Trend, Macro Window, Day-of-Week | Decide *when* the engine is allowed to trade |
| **Entry** | First Touch, Second Touch, Displacement, CE (50%) | Define *how* to enter a position |
| **Exit** | ATR Trail, Breakeven, Next Zone, Session Exit | Manage the open trade once in position |

Each with its own parameter space explored by Optuna during sweep mode.

---

## How to Write Your Own Block

Every block follows the same interface — a single function that takes OHLCV bars and returns a boolean Series:

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

### Tier System

| Tier | Criteria | Action |
|------|----------|--------|
| **S** | PF > 2.0, IS→OOS degradation < 10% | Candidate for live |
| **A** | PF > 1.6, degradation < 15% | Strong, needs more testing |
| **B** | PF > 1.3, degradation < 20% | Promising, keep researching |
| **C** | PF > 1.1 | Weak edge, discard |
| **F** | PF < 1.1 or OOS fails | No edge, discard |

The degradation metric (how much profit factor drops from IS to OOS) is **the real signal**. A strategy with PF 1.8 IS and 1.7 OOS (5.6% degradation) is far more trustworthy than one with 3.0 IS and 1.2 OOS (60% degradation).

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

# Show leaderboard of all past runs
./sb.py --leaderboard
```

---

## Requirements

```
python >= 3.11
pandas
numpy
optuna
pyarrow
rich
loguru
```

---

## Project Structure (158 files)

```
nq-strategy-builder/
├── sb.py                     # CLI entrypoint
├── sb/                       # Engine core (37 files)
│   ├── engine/               #   Parser, combinator, evaluator, walk-forward
│   ├── cache/                #   Signal caching (parquet-based)
│   ├── memory/               #   SQLite run storage
│   ├── filters/              #   News/macro filters
│   └── research/             #   Entry statistics
├── building_blocks/          # DLC system
│   ├── signals/              #   FVG, OB, Sweep, BOS, MSS, iFVG
│   ├── context/              #   Session, HRL/LRL, Trend, Macro
│   ├── entries/              #   First/2nd Touch, Displacement, CE
│   └── exits/                #   ATR Trail, Breakeven, Next Zone, Session
├── helfer/                   # Support modules (9 files)
├── scripts/                  # Research automation (18 scripts)
├── tests/                    # 60 test files
├── examples/                 # Example building block implementations
├── docs/                     # Methodology and research docs
└── Makefile                  # Build and test commands
```

---

## Testing

```bash
pytest tests/ -v           # Run all 60 test suites
./sb.py --validate          # Validate current building blocks
./sb.py --diagnose          # System health check
```

---

## Related Projects

Check out the broader ecosystem:

- [quant-tools](https://github.com/nessos666/quant-tools) — OU process MLE, simulation, bias correction
- [baustein-tester](https://github.com/nessos666/baustein-tester) — Scan and compare signal concepts
- [tv-watch-agent](https://github.com/nessos666/tv-watch-agent) — CDP-based TradingView automation
- [api-health-trust-system](https://github.com/nessos666/api-health-trust-system) — API monitoring for algo trading

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

<p align="center">
  <small>*Built for systematic NQ futures research. 158 files, 60 tests, 0 token costs.*<br>
  <strong>github.com/nessos666</strong></small>
</p>
