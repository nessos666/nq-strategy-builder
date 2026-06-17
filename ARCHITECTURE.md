# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         NQ Strategy Builder                          │
│                                                                      │
│   User Input                                                         │
│   ──────────                                                         │
│   "./sb.py 'FVG + BOS NY'"                                          │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │
│   │   Parser    │───▶│  Knowledge  │───▶│    Combinator       │    │
│   │             │    │   Engine    │    │  (Optuna TPE)       │    │
│   │ text → tokens│   │tokens→algos │    │ param combinations  │    │
│   └─────────────┘    └─────────────┘    └──────────┬──────────┘    │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │    Signal Cache      │   │
│                                          │  compute_*(bars)     │   │
│                                          │  → parquet cache     │   │
│                                          └──────────┬──────────┘   │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │   Backtest Engine   │   │
│                                          │  IS → OOS → HO      │   │
│                                          └──────────┬──────────┘   │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │     Evaluator       │   │
│                                          │  PF, Sharpe, Tier   │   │
│                                          └──────────┬──────────┘   │
│                                                     │               │
│                                          ┌──────────▼──────────┐   │
│                                          │  SQLite + Report    │   │
│                                          │   builder.db        │   │
│                                          └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Deep-Dives

### 1. Parser (`engine/parser.py`)

Converts a free-text idea into structured concept tokens.

```
Input:   "FVG with BOS in NY session"
         │
         ▼
Tokenizer:
  - "FVG"      → concept: FAIR_VALUE_GAP
  - "BOS"      → concept: BREAK_OF_STRUCTURE
  - "NY"       → filter:  session=NY
  - "with/in"  → ignored (stopwords)
         │
         ▼
Output:  StrategySpec(
             concepts=["FVG", "BOS"],
             session_filter="NY",
             raw="FVG with BOS in NY session"
         )
```

---

### 2. Knowledge Engine (`engine/knowledge.py`)

Maps concept names to the actual Python files in your algo library.

```
Concept registry (auto-built from sources.yaml):

  "FVG"              → algo_09_fair_value_gap_v2.py
  "BOS"              → algo_25_bos_v2.py
  "INSIDE_DAY"       → algo_inside_day_v2.py
  "HURST"            → science_hurst_exponent_v2.py
  "SESSIONS"         → algo_20_sessions_v2.py
  "OPENING_RANGE"    → algo_21_opening_range_v2.py
  "DISPLACEMENT"     → algo_30_displacement_v2.py
  "PREMIUM_DISCOUNT" → algo_31_premium_discount_v2.py
  "DEALING_RANGE"    → algo_33_dealing_range_v2.py
  ...

Discovery:
  - Scans all directories listed in sources.yaml
  - Looks for files matching the pattern (e.g. "algo_*_v2.py")
  - Extracts concept name from filename + docstring
  - Builds concept → file mapping at startup
```

---

### 3. Combinator + Optuna (`engine/kombinator.py`)

Generates parameter combinations using Bayesian optimization.

```
Signal A (FVG):
  param_space = {
      "min_gap_pts":   CategoricalParam([2.0, 4.0, 6.0, 8.0]),
      "max_age_bars":  IntParam(20, 100, step=10),
  }

Signal B (BOS):
  param_space = {
      "lookback":      IntParam(5, 30, step=5),
      "min_strength":  FloatParam(0.3, 0.8),
  }

Optuna TPE Sampler:
  Trial 1:  {min_gap_pts=4.0, max_age_bars=50, lookback=10, min_strength=0.5}
  Trial 2:  {min_gap_pts=2.0, max_age_bars=30, lookback=20, min_strength=0.6}
  Trial 3:  (guided by results of Trials 1-2)
  ...
  Trial N:  best params found
```

---

### 4. Signal Cache (`sb/cache/`)

Signals are computed **once** and stored as parquet shards — never recomputed unless parameters change.

```
First run:
  compute_fvg(bars, min_gap_pts=4.0)
       │
       ├── runs on all 717k bars  (~2-5 seconds)
       │
       └── saves to:
           sb/cache/fvg__min_gap_pts=4.0__max_age_bars=50.parquet
                │
                └── columns: ["fvg_bull", "fvg_bear", "fvg_age"]

Second run (same params):
  load sb/cache/fvg__min_gap_pts=4.0__max_age_bars=50.parquet
       │
       └── instant (~50ms) — no recomputation needed

Cache invalidation:
  - Any param change → new filename → recompute
  - Source file mtime change → invalidate all shards for that algo
```

---

### 5. Backtest Engine (`engine/backtest_bridge.py`)

Runs three sequential phases on a fixed timeline.

```
Jan 2024                          Oct 2025    Jan 2026      Mar 2026
   │                                  │           │              │
   ├──────────────────────────────────┼───────────┼──────────────┤
   │                                  │           │              │
   │         IN-SAMPLE (IS)           │    OOS    │   HOLDOUT    │
   │         ~600,000 bars            │  ~85,000  │   ~62,000    │
   │                                  │   bars    │    bars      │
   │  Optuna optimizes params here    │           │              │
   │  → finds best entry/exit combo   │  Honest   │   Locked     │
   │                                  │  check    │   until A/B  │
   └──────────────────────────────────┴───────────┴──────────────┘

Trade simulation per phase:
  For each bar where combined_signal == True:
    entry  = Close[t]
    stop   = entry - (ATR[t] × sl_multiplier)
    target = entry + (ATR[t] × tp_multiplier)
    exit   = first of: target hit, stop hit, max_hold_bars elapsed

Slippage + commission applied per trade:
  net_pnl = raw_pnl - (slippage × 2) - commission
```

---

### 6. Evaluator (`engine/evaluator.py`)

Calculates all performance metrics and assigns tier.

```
From trade list → metrics:

  Profit Factor  =  Sum(winning trades) / |Sum(losing trades)|
  Win Rate       =  winning trades / total trades
  Sharpe Ratio   =  mean(daily_returns) / std(daily_returns) × √252
  Kelly %        =  win_rate - (1 - win_rate) / avg_win_loss_ratio
  Max Drawdown   =  max peak-to-trough decline in equity curve
  Degradation    =  (PF_IS - PF_OOS) / PF_IS × 100%

Tier assignment logic:

  if OOS_trades < 30:          → Tier C  (insufficient data)
  elif OOS_PF >= 1.5:          → Tier A
  elif OOS_PF >= 1.2:          → Tier B
  elif OOS_PF >= 1.0:          → Tier C
  else:                         → Tier D

  Additional red flags (auto-downgrade):
    - Degradation > 40%        → one tier lower
    - Max DD > 30%             → one tier lower
    - Win rate < 30%           → Tier D
```

---

### 7. Meta-Learner (`engine/meta_learner.py`)

Predicts whether a strategy will reach Tier A or B **before** running the full backtest. Saves compute time by skipping low-probability combinations early.

```
Training data: all previously run strategies
               (stored in builder.db)

Features per strategy:
  - signal_activity_rate     (how often signals fire)
  - signal_correlation       (correlation between signal A and B)
  - avg_bars_in_trade        (average trade duration)
  - session_concentration    (% trades in NY vs other sessions)
  - IS_profit_factor         (early IS result, first 20% of data)
  - param_distance_from_default

Target: is_tier_A_or_B (binary)

Model: LightGBM classifier
       trained after every 50 new runs
       → predicts P(Tier A/B) for new candidates

If P(Tier A/B) < 0.15 after 20% IS bars → early abort
  → saves ~80% of compute for near-certain losers
```

---

### 8. Parallel Workers (`engine/worker.py` + `queue_runner.sh`)

```
queue_runner.sh 3
       │
       ├──── spawn Worker 1 (PID 1234)
       │          │
       │          ├── poll builder.db for next pending run
       │          ├── claim run_id=47 (atomic UPDATE SET status='running')
       │          ├── execute full pipeline
       │          └── write results, release claim
       │
       ├──── spawn Worker 2 (PID 1235)
       │          │  (same logic, different run_id)
       │
       └──── spawn Worker 3 (PID 1236)
                  │  (same logic, different run_id)

SQLite WAL mode:
  Multiple readers + one writer at a time — safe without file locking.

temp_guard.sh:
  Monitors CPU temperature.
  If T > 85°C for 3 minutes → SIGSTOP all workers (pause)
  If T < 50°C again         → SIGCONT all workers (resume)
```

---

## TradingView Integration Architecture

```
Python Side                          TradingView Desktop
──────────────────────────           ───────────────────
                                     Chrome DevTools
runner.py                            Protocol (CDP)
    │                                port 9222
    │  1. render template                 │
    │     {{min_gap}} = 4.0               │
    ▼                                     │
pine_template.py                          │
    │                                     │
    │  2. inject Pine Script              │
    ▼                                     │
mcp_bridge.py ──────── MCP ────────────▶ │
    OR                                    │
cdp_bridge.py ──── WebSocket/CDP ──────▶ │
    │                                     │    ┌─────────────┐
    │  3. read result                     │    │  Live Chart │
    │     (PF from strategy tester)       │    │  with Pine  │
    ▼                                     │    │  Script     │
result_store.py                           │    └─────────────┘
    │
    └── saves to tv_kombinatorik_results/
        └── {strategy_name}_results.csv
            columns: [params..., net_profit, pf, win_rate, trades]
```

---

## Signal Generator Interface (Full Spec)

```python
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Config:
    """
    Optional: typed parameter definitions for Optuna search.
    If not provided, the combinator uses default values only.
    """
    min_gap_pts: float = 4.0    # minimum FVG size in NQ points
    max_age_bars: int = 50      # bars until FVG expires


def compute_fvg(
    bars: pd.DataFrame,
    min_gap_pts: float = Config.min_gap_pts,
    max_age_bars: int = Config.max_age_bars,
) -> pd.Series:
    """
    Fair Value Gap detector.

    REQUIRED CONTRACT:
    - First argument must be `bars: pd.DataFrame`
    - DataFrame has columns: Open, High, Low, Close, Volume
    - Index: DatetimeIndex (UTC timezone)
    - Returns: pd.Series with same index as bars
    - Return dtype: bool (active/not active) OR float (score 0–1)
    - No side effects, no global state
    - Must be deterministic given same inputs

    OPTIONAL CONTRACT:
    - Additional keyword arguments = tunable parameters
    - @dataclass Config defines types + defaults for Optuna
    - Multiple compute_*() functions per file are allowed
      (each becomes a separate signal variant)
    """
    # ... implementation
    result = pd.Series(False, index=bars.index)
    return result
```

---

## Data Models (`sb/models.py`)

```
StrategySpec
  ├── idea: str              "FVG + BOS NY"
  ├── concepts: list[str]    ["FVG", "BOS"]
  ├── session: str | None    "NY"
  └── params: dict           {"min_gap_pts": 4.0, ...}

BacktestResult
  ├── run_id: int
  ├── spec: StrategySpec
  ├── is_metrics: PhaseMetrics
  ├── oos_metrics: PhaseMetrics
  ├── ho_metrics: PhaseMetrics | None   (None if not yet opened)
  ├── tier: Literal["A","B","C","D"]
  ├── degradation_pct: float
  └── created_at: datetime

PhaseMetrics
  ├── profit_factor: float
  ├── win_rate: float
  ├── trade_count: int
  ├── sharpe: float
  ├── kelly_pct: float
  └── max_drawdown_pct: float
```

---

## Database Schema

```sql
-- builder.db (SQLite, WAL mode)

CREATE TABLE strategies (
    run_id          INTEGER PRIMARY KEY,
    idea            TEXT NOT NULL,
    params_json     TEXT NOT NULL,         -- JSON: {"min_gap_pts": 4.0, ...}
    tier            TEXT NOT NULL,         -- "A" | "B" | "C" | "D"
    is_pf           REAL,
    oos_pf          REAL,
    ho_pf           REAL,
    is_trades       INTEGER,
    oos_trades      INTEGER,
    ho_trades       INTEGER,
    is_winrate      REAL,
    oos_winrate     REAL,
    sharpe_oos      REAL,
    degradation_pct REAL,
    status          TEXT DEFAULT 'pending', -- 'pending'|'running'|'done'|'error'
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_tier    ON strategies(tier);
CREATE INDEX idx_oos_pf  ON strategies(oos_pf DESC);
CREATE INDEX idx_status  ON strategies(status);
```

---

## Module Dependency Graph

```
sb.py
  └── sb/cli.py
        ├── sb/combinator.py
        │     ├── sb/engine/parser.py
        │     ├── sb/engine/knowledge.py
        │     │     └── sb/algo_paths.py
        │     └── sb/engine/kombinator.py
        │           ├── sb/cache/signal_cache.py
        │           │     └── sb/cache/concept_algo_map.py
        │           └── sb/engine/backtest_bridge.py
        │                 ├── sb/engine/nautilus_bridge.py
        │                 ├── sb/engine/evaluator.py
        │                 │     └── sb/filters/news_filter.py
        │                 └── sb/engine/walk_forward.py
        │
        ├── sb/memory/db.py
        ├── sb/report.py
        ├── sb/analyse.py
        ├── sb/inspect.py
        └── sb/diagnose.py
```

---

## Test Coverage

```
tests/
│
├── Core logic
│   ├── test_parser.py              ← text → concept tokens
│   ├── test_knowledge.py           ← concept → algo file mapping
│   ├── test_combinator.py          ← combination generation
│   ├── test_kombinator.py          ← Optuna param search
│   ├── test_models.py              ← Pydantic validation
│   └── test_db.py                  ← SQLite storage
│
├── Backtest engine
│   ├── test_backtest_bridge.py     ← IS/OOS/HO phase execution
│   ├── test_nautilus_bridge.py     ← NautilusTrader integration
│   ├── test_evaluator.py           ← metrics + tier assignment
│   ├── test_walk_forward.py        ← walk-forward splits
│   └── test_news_filter.py         ← news window filtering
│
├── Signal cache
│   ├── test_signal_cache.py        ← cache build + load
│   ├── test_cache_query.py         ← cache query interface
│   └── test_concept_algo_map.py    ← file discovery
│
├── CLI & tools
│   ├── test_cli_validate.py
│   ├── test_cli_export.py
│   ├── test_inspect.py             ← signal rate + heatmap
│   └── test_diagnose.py
│
├── TradingView
│   ├── test_pine_template.py       ← {{param}} substitution
│   ├── test_param_space.py         ← Optuna search spaces
│   ├── test_runner.py              ← sweep orchestration
│   ├── test_result_store.py        ← CSV result storage
│   └── test_tv_client.py
│
├── Integration
│   ├── test_integration.py         ← full pipeline dry run
│   └── test_integration_dry.py     ← without market data
│
└── External algos (auto-skipped if TRADINGPROJEKT_PATH not set)
    ├── test_mtf_align.py
    ├── test_hawkes_ofi.py
    ├── test_ou_mean_reversion.py
    └── test_regime_gate.py

Total: 523 tests
```

---

## Hot Path / Slow Path Separation

Inspired by Jennifer Lewis's [FoxML_Trader_v2](https://github.com/Jennyfirrr/FoxML_Trader_v2) per-core sharded engine architecture.

### Concept

The engine separates fast, per-tick decisions (Hot Path) from slow, analytical computation (Slow Path):

| | Hot Path | Slow Path |
|---|----------|-----------|
| **What** | Signal lookup, gate evaluation | Walk-forward, ML inference, parameter optimization |
| **When** | Every tick | Every N minutes / on parameter change |
| **Latency** | Microseconds (cache read) | Seconds to minutes |
| **Writes?** | No (read only) | Yes (updates signal cache) |
| **Handoff** | Reads from sb/cache/ (Parquet) | Writes to sb/cache/ via SignalCache |

### Current Implementation

```
HOT PATH (live trading — future):
  sb/cache/signal_cache.py  <- reads pre-computed signals
  sb/engine/evaluator.py    <- gate evaluation only
  NO database access, NO ML inference

SLOW PATH (research — current):
  sb/engine/walk_forward.py <- IS/OOS/Holdout computation
  sb/engine/kombinator.py   <- Optuna parameter search
  sb/memory/db.py           <- SQLite persistence
  sb/cache/signal_cache.py  <- writes computed signals
```

### Future: Seqlock Handoff

When live trading is added, the hot path and slow path will communicate through a lock-free double-buffer:

- Producer (Slow Path): Writes new GateParameters when strategy updates
- Consumer (Hot Path): Reads current parameters on every tick (1 ns steady state)
- No waiting, no locks, no torn reads.

### The Core Insight

> "The hot path is immune to model complexity." — Jennifer Lewis

Swap the ML model behind the slow path and the hot path's per-tick cost stays identical. This is the architecture that makes live trading feasible without sacrificing research depth.
