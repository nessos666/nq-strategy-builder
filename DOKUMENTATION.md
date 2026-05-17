# Strategie Builder – Systemdokumentation
Stand: 2026-05-12 (18:00 Uhr)
Systempfad: <PROJECT_ROOT> (Standard: aktuelles Verzeichnis)

---

## 1. ÜBERBLICK

Der **NQ Strategy Builder** ist ein systematisches Framework zum Entdecken, Testen und Evaluieren von algorithmischen Trading-Strategien auf Nasdaq-100 (NQ/MNQ) Futures mittels 1-Minuten-Bars (717.000 Bars, Jan 2024 – Mar 2026).

Kernprinzip: Der Trader beschreibt eine Idee in natürlicher Sprache (z.B. "FVG + BOS NY"), das System findet passende Signal-Generatoren, kombiniert sie, führt einen 3-Phasen-Backtest durch und bewertet die Strategie mit einem Tier-System (A/B/C/D).

---

## 2. ARCHITEKTUR

### 2.1 Pipeline (vereinfacht)

```
Idee (Text) → Parser → Knowledge Engine → Kombinator (Optuna)
              → Signal Cache → Backtest Engine → Evaluator → SQLite + Report
```

### 2.2 Verzeichnisstruktur

```
/24_Strategie_Builder/
├── sb.py                          ← CLI Entry Point (Typer)
├── CLAUDE.md                       ← Regeln für AI-Bearbeitung
├── ARCHITECTURE.md                 ← Technische Architektur
├── DOKUMENTATION.md                ← Diese Datei
├── README.md                       ← Projekt-README
├── Makefile                        ← make test / make install
│
├── sb/                             ← Core Framework
│   ├── cli.py                      ← CLI-Kommandos (build, batch, lager, ...)
│   ├── engine/
│   │   ├── parser.py               ← Text → Konzept-Tokens
│   │   ├── knowledge.py            ← Tokens → Algo-Dateien
│   │   ├── kombinator.py           ← Optuna TPE-Parametersuche
│   │   ├── walk_forward.py         ← Walk-Forward mit 3 Fenstern
│   │   ├── nautilus_bridge.py      ← NautilusTrader Backtest-Engine
│   │   ├── evaluator.py            ← Metriken + Tier-Vergabe
│   │   ├── meta_learner.py         ← ML-Prädiktion (LightGBM)
│   │   ├── worker.py               ← Task-Runner (picklable)
│   │   ├── sb_replayer.py          ← Bar-Replay
│   │   └── backtest_bridge.py      ← Alte IS/OOS/HO-Bridge
│   ├── cache/
│   │   ├── signal_cache.py         ← Signal-Caching (Parquet-Shards)
│   │   └── concept_algo_map.py     ← Konzept↔Algo-Mapping
│   ├── memory/
│   │   └── db.py                   ← SQLite-Datenbank (WAL-Modus)
│   ├── filters/
│   │   └── news_filter.py          ← Nachrichten-Filter (FOMC/NFP)
│   ├── combinator.py               ← Kombinationslogik
│   ├── analyse.py                  ← Post-hoc-Analyse
│   ├── inspect.py                  ← Signal-Rate + Heatmap
│   ├── diagnose.py                 ← Debug-Tool
│   ├── report.py                   ← Rich-Report-Generator
│   └── models.py                   ← Pydantic-Datenmodelle
│
├── david_bibliothek/               ← Algo-Bibliothek (Signal-Generatoren)
│   ├── 01_Hoch_Tief/               ← (2 algos, gesperrt)
│   ├── 02_FVG_Zonen/               ← (4 algos, gesperrt)
│   ├── 03_Order_Blocks/            ← (4 algos, gesperrt)
│   ├── 03_Context/                 ← (4 algos: HRL/LRL, premium_discount)
│   ├── 04_Opening_Gaps_NDOG_NWOG/  ← (3 algos, NICHT gesperrt)
│   ├── 05_Stoploss_TakeProfit/     ← (6 algos, 3 gesperrt + dynamic)
│   ├── 06_Time_Zeit/               ← (2 algos, gesperrt)
│   ├── 09_Entry_Logik/             ← (3 algos, gesperrt)
│   ├── 09_Concept_Algos/           ← (1 algo, gesperrt)
│   ├── 10_Exit_Logik/              ← (4 algos, gesperrt)
│   ├── 11_ICT_Konzepte/            ← (1 algo: manip, gesperrt)
│   └── 99_Alte_Algos_Noch_Nicht_Getestet/ ← (4 algos: cbdr, dealing_range, ote, quarterly_theory)
│
├── ideas/queue/                    ← Batch-Queue-System
│   ├── new50_batch_09.txt – 20.txt ← Aktuelle Queue (10 Batches, 50+ Ideen)
│   └── done/                       ← Abgearbeitete Batches
│
├── output_david_1/                 ← Worker-Slot "david" (Haupt-Ergebnisse)
│   ├── builder.db                  ← Ergebnisse (altes Schema: build_runs)
│   ├── objektbaum_20260512.md      ← Vollständiger Ergebnis-Objektbaum (105 Runs)
│   ├── kombi_7d_ob_gefiltert_...md ← Portfolio-Kombination (3 Strategien parallel)
│   └── report_*.md                 ← Einzel-Reports (ca. 100+ Stück)
│
├── output_worker_1/                ← Worker-Slot 1 (aktuell 6 Reports)
├── output_worker_2/                ← Worker-Slot 2 (aktuell 6 Reports)
├── output_worker_3/                ← Worker-Slot 3 (aktuell 6 Reports)
│
├── queue_runner.sh                 ← Automatische Worker-Verwaltung (3 Slots)
├── temp_guard.sh                   ← CPU-Temperatur-Wächter
├── auto_next50.sh                  ← Automatische Nachqueue von 50 Ideen
├── start_phase1.sh                 ← Phase-1-Starter
├── run_batches_13_14_15.sh         ← Historischer Batch-Runner
├── run_recovery_batches.sh         ← Recovery nach DB-Reset
│
├── batch_*.txt                     ← Alte Batch-Dateien (Exit-Modi, Trails, etc.)
├── knowledge_sources/
│   └── sources.yaml                ← Quell-Pfad-Konfiguration
├── tests/                          ← Pytest-Tests (523 Tests)
├── scripts/                        ← Hilfsskripte
├── tv_kombinatorik/                ← TradingView-Integration
├── tv_pine_bibliothek/             ← Pine Script v6 Indikatoren
└── data/                           ← Market-Daten-Pfad
```

---

## 3. WORKER-SYSTEM

### 3.1 Drei parallele Worker-Slots

Das System hat **3 Workerslots**, die parallel arbeiten. Der `queue_runner.sh` managt sie automatisch:

| Slot | Verzeichnis | Status |
|------|-------------|--------|
| Worker 1 | `output_worker_1/` | Aktiv (6 Reports am 12.05.2026) |
| Worker 2 | `output_worker_2/` | Aktiv (6 Reports am 12.05.2026) |
| Worker 3 | `output_worker_3/` | Aktiv (6 Reports am 12.05.2026) |
| David | `output_david_1/` | Frühere Runs, 100+ Reports |

**Ablauf:**
1. `queue_runner.sh` sucht im 30-Sekunden-Takt freie Slots
2. Findet den nächsten Batch in `ideas/queue/*.txt`
3. Verschiebt ihn nach `ideas/queue/done/`
4. Startet ein Pilot-Prüfscript (`helfer.batch_pilot`)
5. Startet den Batch via `.venv/bin/python sb.py batch "ideas/queue/done/..." --output "output_worker_X"`
6. Der `temp_guard.sh` überwacht die CPU-Temperatur:
   - ≥ 85°C: Akkumulation (120s bis STOP)
   - ≥ 90°C: Sofortiger STOP aller Batches
   - ≤ 50°C: Batches fortsetzen (SIGCONT)
   - > 5 Min pausiert + weiter heiß: SIGKILL

### 3.2 Worker-Implementierung (`sb/engine/worker.py`)

Jeder Worker ruft `run_worker_task(WorkerConfig)` auf:
1. Duplikat-Check gegen `builder.db`
2. Idee parsen → NautilusBridge initialisieren
3. WalkForwardEngine mit Optuna-Parametersuche (3 Fenster, 75/25 IS/OOS)
4. Monte Carlo Robustheitstest (über science_robustness_tools_v2)
5. Ergebnisse in DB speichern + Tier berechnen
6. Report generieren

### 3.3 Ausgabe pro Worker

Jeder Worker produziert:
- `builder.db` – eigene SQLite-DB mit Ergebnissen
- `studies.db` – Optuna-Studies
- `report_YYYY-MM-DD_HH-MM.md` – Einzelbericht pro Idee
- `trades/run_XX_trades.parquet` – Trade-Daten (nur output_david_1)

---

## 4. BACKTEST-METHODIK

### 4.1 Walk-Forward mit 3 Fenstern

```
Jan 2024                    Oct 2025    Jan 2026    Mar 2026
   │                            │           │           │
   ├────────────────────────────┼───────────┼───────────┤
   │                            │           │           │
   │   IN-SAMPLE (IS)           │    OOS    │  HOLDOUT  │
   │   ~600k bars               │  ~85k bars│  ~62k bars│
   │   Optuna optimiert hier    │  1. Check │  LOCKED   │
```

- IS: Optuna findet beste Parameter (TPE Sampler)
- OOS: Erster ehrlicher Check
- HO: Komplett gesperrt, nur für Tier A/B geöffnet

### 4.2 Tier-System

| Tier | OOS PF | Min Trades | Bedeutung |
|------|--------|------------|-----------|
| A | ≥ 1.5 | ≥ 30 | Starker Edge – HO öffnen, Live-Trading |
| B | ≥ 1.2 | ≥ 30 | Vielversprechend – mehr Daten sammeln |
| C | ≥ 1.0 | ≥ 30 | Kein klarer Edge – archivieren |
| D | < 1.0 | beliebig | Verlierend – verwerfen |

Zusätzliche Red Flags: Degradation > 40%, Max DD > 30%, Win-Rate < 30%

---

## 5. AKTUELL GETESTETE STRATEGIEN

### 5.1 Aktuelle Queue (12.05.2026, 18:00 Uhr)

**10 Batches in Warteschlange** (`ideas/queue/new50_batch_09.txt` bis `20.txt`):

| Batch | Thema | Ideen |
|-------|-------|-------|
| Batch 09 | SHORT + Macro Short/HRLLL | 7d short NY macro short trail, fvg 2tage short NY hrl_lrl trail, ifvg sameday short NY hrl_lrl trail, fvg standard short NY hrl_lrl trail, fvg standard first touch NY trail |
| Batch 10 | First/Second Touch | fvg 2tage first touch NY trail, ifvg sameday first touch NY trail, fvg standard second touch NY trail, fvg 2tage second touch NY trail, fvg 1-2wochen first touch NY trail |
| Batch 11 | LONG + HRL/LRL (Entry-Filter) | ob session hrl lrl NY trail, fvg standard hrl lrl NY trail, fvg 2tage hrl lrl NY trail, ifvg 1woche hrl lrl NY trail, fvg 1-2wochen hrl lrl NY trail |
| Batch 12 | SHORT + HRL/LRL | ob session short hrl lrl NY trail, fvg standard short hrl lrl NY trail, fvg 2tage short hrl lrl NY trail, ifvg 1woche short hrl lrl NY trail, ob chaos short hrl lrl NY trail |
| Batch 13 | Breakeven Trail LONG | 7d NY breakeven trail, ob session NY breakeven trail, fvg standard NY breakeven trail, ifvg 1woche NY breakeven trail, fvg 2tage NY breakeven trail |
| Batch 14 | Breakeven Trail SHORT | 7d short NY breakeven trail, ob session short NY breakeven trail, fvg standard short NY breakeven trail, manip bear NY breakeven trail, ifvg 1woche short NY breakeven trail |
| Batch 15 | 3er: Entry + Macro Long + HRL/LRL | ob session macro long hrl lrl NY trail, 7d macro long hrl lrl NY trail, fvg standard macro long hrl lrl NY trail, fvg 2tage macro long hrl lrl NY trail, ifvg 1woche macro long hrl lrl NY trail |
| Batch 16 | 3er: SHORT + Macro Short + HRL/LRL | ob session short macro short hrl lrl NY trail, fvg standard short macro short hrl lrl NY trail, 7d short macro short hrl lrl NY trail, ifvg 1woche short macro short hrl lrl NY trail, fvg 2tage short macro short hrl lrl NY trail |
| Batch 17 | MANIP LONG | manip long NY trail, manip long hrl lrl NY trail, manip long macro long NY trail, manip long NY breakeven trail, manip long hrl lrl macro long NY trail |
| Batch 18 | OB Chaos + OB Tageshoch mit Filtern | ob chaos hrl lrl NY trail, ob tageshoch hrl lrl NY trail, ob tageshoch short hrl lrl NY trail, ob chaos macro long hrl lrl NY trail, ob tageshoch macro long hrl lrl NY trail |
| Batch 19 | SHORT + Macro Short (3er ohne HRL/LRL) | ob session short macro short NY trail, fvg standard short macro short NY trail, 7d short macro short NY trail, ifvg 1woche short macro short NY trail, ob chaos short macro short NY trail |
| Batch 20 | 3er LONG/SHORT mit FVG 1-2W | fvg 1-2wochen macro long hrl lrl NY trail, fvg 1-2wochen short hrl lrl NY trail, fvg 1-2wochen short macro short NY trail, ob tageshoch short macro short NY trail, ob session macro short hrl lrl NY trail |

Schwerpunkt aktuell: **Exit-Varianten** (Breakeven, Trail, Session Level, Next Zone) in 2er- und 3er-Kombinationen mit Kontext-Filtern (HRL/LRL, Macro Time, Manipulation).

### 5.2 Bereits getestet (105 Runs, Objektbaum)

Siehe `output_david_1/objektbaum_20260512.md` für die vollständige Taxonomie.

**Beste Ergebnisse (HO PF ≥ 1.5, Tier A):**

| ID | Strategie | OOS PF | HO PF | Richtung |
|----|-----------|--------|-------|----------|
| 30 | 7d bull NY trail | 2.20 | 4.06 | LONG |
| 53 | ob session NY breakeven trail | 1.98 | 3.72 | LONG |
| 33 | ob session bull NY trail | 1.65 | 3.72 | LONG |
| 22 | 7d manip bear NY trail | 1.62 | 3.10 | LONG |
| 55 | 7d NY trail | 1.52 | 1.90 | LONG |
| 73 | 7d NY manip trail | 1.43 | 1.76 | LONG |
| 83 | 7d NY manip bear trail | 1.45 | 1.75 | LONG |
| 106 | ob session NY hrl_lrl trail | 1.45 | 1.71 | LONG |
| 25 | fvg standard manip bear NY trail | 1.30 | 1.71 | LONG |
| 26 | ifvg 1woche manip bear NY trail | 1.27 | 1.70 | LONG |
| 40 | ifvg sameday NY trail | 1.29 | 1.69 | LONG |
| 65 | 7d NY hrl lrl trail | 1.43 | 1.69 | LONG |
| 41 | fvg 2tage NY trail | 1.29 | 1.72 | LONG |
| 42 | fvg 1-2wochen NY trail | 1.30 | 1.62 | LONG |
| 78 | ob session NY manip trail | 1.38 | 1.59 | LONG |
| 84 | 7d NY manip long trail | 1.37 | 1.51 | LONG |
| 107 | 7d short NY hrl_lrl trail | 1.34 | 1.76 | SHORT |
| 32 | fvg standard NY trail | 1.31 | 1.63 | LONG |
| 43 | ifvg 1woche NY trail | 1.26 | 1.71 | LONG |
| 97 | fvg 2tage short NY trail | 1.27 | 1.35 | SHORT |

---

## 6. KONZEPT-MAP (Signal-Generatoren)

Die Algo-Bibliothek (`david_bibliothek/`) enthält ~41 Python-Dateien. Davon sind **27 gesperrt** (read-only, chmod 444) und 5 in `99_Alte_Algos_Noch_Nicht_Getestet/` + 3 in `04_Opening_Gaps/` noch ungetestet.

**Hauptkonzepte in aktiver Nutzung:**

| Kurzname | Vollname | Kategorie |
|----------|----------|-----------|
| 7d | Session Hoch-Tief Orderblock | OB-Typ |
| ob session | Session Orderblock | OB-Typ |
| ob chaos | Chaos Order Block | OB-Typ |
| ob tageshoch | Tageshoch-Tief Orderblock | OB-Typ |
| fvg standard | FVG Standard | FVG-Typ |
| fvg 2tage | FVG 2 Tage | FVG-Typ |
| fvg 1-2wochen | FVG 1-2 Wochen | FVG-Typ |
| ifvg sameday | iFVG SameDay | FVG-Typ |
| ifvg 1woche | iFVG 1 Woche | FVG-Typ |
| hrl lrl | HRL/LRL Filter (2 Varianten) | Kontext-Filter |
| macro short/long | Macro Time | Time-Filter |
| manip/manip bear | Liquidity Sweep | ICT-Konzept |
| trail | ATR-Trailing (exit_atr_trail) | Exit-Logik |
| breakeven | Breakeven (exit_breakeven) | Exit-Logik |
| next zone | Exit Next Zone | Exit-Logik |
| session level | Exit Session Level | Exit-Logik |
| first touch | Entry First Touch | Entry-Logik |
| second touch | Entry Second Touch 50% | Entry-Logik |
| displacement | Entry Displacement | Entry-Logik |

---

## 7. DATENBANKEN

| Datei | Zweck | Schema |
|-------|-------|--------|
| `output_david_1/builder.db` | Hauptergebnisse | Altes Schema (build_runs-Tabelle) |
| `output_worker_*/builder.db` | Worker-Ergebnisse | Neues Schema (strategies-Tabelle) |
| `builder.db` | Root-DB | Hauptdatenbank |
| `strategie_builder.db` | Zusätzlich | – |
| `output_worker_*/studies.db` | Optuna-Studies | Optuna-Intern |

---

## 8. CLI-BEFEHLE

| Befehl | Beschreibung |
|--------|-------------|
| `./sb.py "Idee"` | Einzelne Idee testen |
| `./sb.py batch <datei>` | Batch-Datei abarbeiten |
| `./sb.py lager` | Alle Ergebnisse anzeigen |
| `./sb.py show <id>` | Detailansicht eines Runs |
| `./sb.py inspect "Konzept"` | Signal-Rate + Heatmap |
| `./sb.py diagnose <id>` | Fehlerdiagnose |
| `./sb.py lock/unlock` | Algos sperren/entsperren |
| `./sb.py lock-status` | Sperrstatus anzeigen |
| `./sb.py maintain` | Wartungsarbeiten |

---

## 9. BETRIEB

### Starten
```bash
cd <PROJECT_ROOT>
./start_phase1.sh          # Phase 1 initial starten
# oder direkt:
nohup bash queue_runner.sh &  # Queue-Runner im Hintergrund
```

### Überwachung
- `temp_guard.sh` läuft automatisch als Temperaturwächter
- Reports in `output_worker_X/report_*.md`
- Logs in `/tmp/queue_runner.log`, `/tmp/temp_guard.log`

### Wichtige Dateien
- `CLAUDE.md` – Regeln für AI-Bearbeitung (niemals gesperrte Algos editieren!)
- `knowledge_sources/sources.yaml` – Datenpfade
- `ARCHITECTURE.md` – Detaillierte technische Dokumentation

---

## 10. ZUSAMMENFASSUNG AKTUELLER ZUSTAND

- **System:** Voll funktionsfähig, 3 parallele Worker aktiv
- **Getestet:** 105 Strategien (Stand Objektbaum 12.05.2026)
- **HO-validiert:** 37 Strategien
- **Robust gesamt:** 39 Strategien
- **In Queue:** 10 Batches (~50 Ideen), Schwerpunkt Exit-Varianten mit Filtern
- **Aktiv bearbeitet:** Batches 9-20 werden parallel von 3 Workern abgearbeitet
- **Letzte Aktivität:** 12.05.2026 ab 14:00 Uhr laufend (Worker 1-3 produzieren Reports)
- **Beste Strategien:** 7d-Konzepte und OB-Session mit Trail/Breakeven-Exits dominieren die Top-Ergebnisse (HO PF bis 4.06)
