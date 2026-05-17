"""
Hurst-Exponent Vergleich: Unsere Implementierung vs. 5 PyPI-Bibliotheken

Vergleicht H-Werte auf denselben NQ 1-Minuten-Daten (letzten 2000 Bars).

Ausgabe:
- Tabelle: Mittelwert, Std, Min, Max, % mean-reverting (H<0.45) pro Methode
- Korrelations-Matrix aller Methoden
- Scatter-Plot: unsere Implementierung vs. jede Bibliothek
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Daten laden ────────────────────────────────────────────────────────────────
DATA = Path(
    Path(__file__).resolve().parent.parent / "data"
    "/nq_backtest/data/nq_1m_databento_2024_2026.parquet"
)
if not DATA.exists():
    print("Datei nicht gefunden:", DATA)
    sys.exit(1)

df = pd.read_parquet(DATA)
df.columns = [c.capitalize() for c in df.columns]
close = df["Close"].ffill().dropna()

# Letzten 5000 Bars für Vergleich (= ~3-4 Handelstage auf 1min)
N_BARS = 5000
WINDOW = 100  # Rolling-Fenster (wie unsere Implementierung)

close_arr = close.values[-N_BARS:]
print(f"Bars: {len(close_arr)}, Fenster: {WINDOW}")
print(f"Preis-Bereich: {close_arr.min():.1f} – {close_arr.max():.1f}")
print()

# ── Hilfsfunktion: Rolling H für jede Bibliothek ──────────────────────────────


def rolling_hurst_mottl(prices: np.ndarray, window: int) -> np.ndarray:
    """hurst (Mottl): compute_Hc mit R/S-Methode"""
    from hurst import compute_Hc  # type: ignore

    result = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        try:
            H, _, _ = compute_Hc(prices[i - window : i], kind="price", simplified=True)
            result[i] = H
        except Exception:
            result[i] = 0.5
    return result


def rolling_hurst_nolds(prices: np.ndarray, window: int) -> np.ndarray:
    """nolds: DFA (Detrended Fluctuation Analysis)"""
    import nolds  # type: ignore

    result = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        try:
            result[i] = nolds.dfa(prices[i - window : i])
        except Exception:
            result[i] = 0.5
    return result


def rolling_hurst_exponent_pkg(prices: np.ndarray, window: int) -> np.ndarray:
    """hurst-exponent: generalized structure function"""
    try:
        from hurst_exponent import hurst_exponent_py  # type: ignore
    except ImportError:
        try:
            import hurst_exponent as he  # type: ignore

            getattr(he, "hurst_exponent", None) or getattr(he, "compute", None)
        except ImportError:
            return np.full(len(prices), np.nan)
    result = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        try:
            val = hurst_exponent_py(prices[i - window : i])
            result[i] = float(val) if val is not None else 0.5
        except Exception:
            result[i] = 0.5
    return result


def rolling_hurst_estimators(prices: np.ndarray, window: int) -> np.ndarray:
    """hurst-estimators: mehrere Methoden, wir nutzen RS"""
    try:
        import hurst_estimators as hest  # type: ignore

        fn = getattr(hest, "rs_hurst", None) or getattr(hest, "compute_hurst", None)
    except ImportError:
        return np.full(len(prices), np.nan)
    if fn is None:
        return np.full(len(prices), np.nan)
    result = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        try:
            result[i] = float(fn(prices[i - window : i]))
        except Exception:
            result[i] = 0.5
    return result


def rolling_hurst_exp_hurst(prices: np.ndarray, window: int) -> np.ndarray:
    """exp-hurst"""
    try:
        import exp_hurst as eh  # type: ignore

        fn = getattr(eh, "compute", None) or getattr(eh, "hurst", None)
    except ImportError:
        return np.full(len(prices), np.nan)
    if fn is None:
        return np.full(len(prices), np.nan)
    result = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        try:
            result[i] = float(fn(prices[i - window : i]))
        except Exception:
            result[i] = 0.5
    return result


def rolling_hurst_unsere(prices: np.ndarray, window: int) -> np.ndarray:
    """Unsere Implementierung (Log-Return Varianz Methode, numba)"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from david_bibliothek._00_nicht_funktionierend import hurst_exponent as he_module  # type: ignore

    # Direkt über run() laufen lassen
    df_tmp = pd.DataFrame({"Close": prices})
    out = he_module.run(df_tmp)
    return out["hurst_exp"].values


# ── Alle Methoden berechnen (mit Fortschrittsanzeige) ─────────────────────────

print("Berechne rolling H für alle Methoden...")
print("(Das dauert ein paar Minuten wegen Python-Loops)")
print()

# Unsere eigene (numba, schnell)
print("1/6  Unsere Implementierung (numba)...", end=" ", flush=True)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hurst_ours",
        Path(__file__).parent.parent
        / "david_bibliothek/00_Nicht_Funktionierend/hurst_exponent.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["hurst_ours"] = mod
    spec.loader.exec_module(mod)  # type: ignore
    df_tmp = pd.DataFrame({"Close": close_arr})
    h_ours = mod.run(df_tmp)["hurst_exp"].values
    print("OK")
except Exception as e:
    print(f"FEHLER: {e}")
    h_ours = np.full(len(close_arr), np.nan)

# Mottl (langsam – nur 500 Stichproben)
print("2/6  hurst (Mottl / R-S-Analyse)...", end=" ", flush=True)
try:
    # Nur jeden 5. Bar berechnen für Speed, dann interpolieren
    STEP = 5
    indices = list(range(WINDOW, len(close_arr), STEP))
    from hurst import compute_Hc  # type: ignore

    h_mottl_sparse = []
    for i in indices:
        try:
            H, _, _ = compute_Hc(
                close_arr[i - WINDOW : i], kind="price", simplified=True
            )
            h_mottl_sparse.append(H)
        except Exception:
            h_mottl_sparse.append(0.5)
    # Auf volle Länge interpolieren
    h_mottl = np.full(len(close_arr), np.nan)
    for k, i in enumerate(indices):
        h_mottl[i] = h_mottl_sparse[k]
    h_mottl_series = pd.Series(h_mottl).interpolate().values
    print("OK")
except Exception as e:
    print(f"FEHLER: {e}")
    h_mottl_series = np.full(len(close_arr), np.nan)

# nolds DFA
print("3/6  nolds (DFA)...", end=" ", flush=True)
try:
    import nolds  # type: ignore

    STEP = 10
    indices = list(range(WINDOW, len(close_arr), STEP))
    h_nolds_sparse = []
    for i in indices:
        try:
            h_nolds_sparse.append(nolds.dfa(close_arr[i - WINDOW : i]))
        except Exception:
            h_nolds_sparse.append(0.5)
    h_nolds = np.full(len(close_arr), np.nan)
    for k, i in enumerate(indices):
        h_nolds[i] = h_nolds_sparse[k]
    h_nolds_series = pd.Series(h_nolds).interpolate().values
    print("OK")
except Exception as e:
    print(f"FEHLER: {e}")
    h_nolds_series = np.full(len(close_arr), np.nan)

# hurst-exponent
print("4/6  hurst-exponent...", end=" ", flush=True)
try:
    import hurst_exponent as hexp  # type: ignore

    fn_names = [n for n in dir(hexp) if not n.startswith("_")]
    fn = None
    for name in fn_names:
        candidate = getattr(hexp, name)
        if callable(candidate):
            fn = candidate
            break
    if fn:
        STEP = 5
        indices = list(range(WINDOW, len(close_arr), STEP))
        h_hexp_sparse = []
        for i in indices:
            try:
                result = fn(close_arr[i - WINDOW : i])
                h_hexp_sparse.append(float(result))
            except Exception:
                h_hexp_sparse.append(0.5)
        h_hexp = np.full(len(close_arr), np.nan)
        for k, i in enumerate(indices):
            h_hexp[i] = h_hexp_sparse[k]
        h_hexp_series = pd.Series(h_hexp).interpolate().values
        print(f"OK (Funktion: {fn.__name__})")
    else:
        print(f"keine Funktion gefunden. Verfügbar: {fn_names}")
        h_hexp_series = np.full(len(close_arr), np.nan)
except Exception as e:
    print(f"FEHLER: {e}")
    h_hexp_series = np.full(len(close_arr), np.nan)

# hurst-estimators
print("5/6  hurst-estimators...", end=" ", flush=True)
try:
    import hurst_estimators as hest  # type: ignore

    fn_names = [
        n for n in dir(hest) if not n.startswith("_") and callable(getattr(hest, n))
    ]
    fn = None
    for name in fn_names:
        fn = getattr(hest, name)
        break
    if fn:
        STEP = 5
        indices = list(range(WINDOW, len(close_arr), STEP))
        h_hest_sparse = []
        for i in indices:
            try:
                h_hest_sparse.append(float(fn(close_arr[i - WINDOW : i])))
            except Exception:
                h_hest_sparse.append(0.5)
        h_hest = np.full(len(close_arr), np.nan)
        for k, i in enumerate(indices):
            h_hest[i] = h_hest_sparse[k]
        h_hest_series = pd.Series(h_hest).interpolate().values
        print(f"OK (Funktion: {fn.__name__})")
    else:
        print(f"keine Funktion gefunden. Verfügbar: {fn_names}")
        h_hest_series = np.full(len(close_arr), np.nan)
except Exception as e:
    print(f"FEHLER: {e}")
    h_hest_series = np.full(len(close_arr), np.nan)

# exp-hurst
print("6/6  exp-hurst...", end=" ", flush=True)
try:
    import exp_hurst as eh  # type: ignore

    fn_names = [
        n for n in dir(eh) if not n.startswith("_") and callable(getattr(eh, n))
    ]
    fn = None
    for name in fn_names:
        fn = getattr(eh, name)
        break
    if fn:
        STEP = 5
        indices = list(range(WINDOW, len(close_arr), STEP))
        h_eh_sparse = []
        for i in indices:
            try:
                h_eh_sparse.append(float(fn(close_arr[i - WINDOW : i])))
            except Exception:
                h_eh_sparse.append(0.5)
        h_eh = np.full(len(close_arr), np.nan)
        for k, i in enumerate(indices):
            h_eh[i] = h_eh_sparse[k]
        h_eh_series = pd.Series(h_eh).interpolate().values
        print(f"OK (Funktion: {fn.__name__})")
    else:
        print(f"keine Funktion gefunden. Verfügbar: {fn_names}")
        h_eh_series = np.full(len(close_arr), np.nan)
except Exception as e:
    print(f"FEHLER: {e}")
    h_eh_series = np.full(len(close_arr), np.nan)

# ── Ergebnis-Tabelle ───────────────────────────────────────────────────────────

print()
print("=" * 70)
print("ERGEBNIS-VERGLEICH (letzten 5000 Bars NQ 1min)")
print("=" * 70)

methods = {
    "Unsere (Log-Return)": h_ours,
    "Mottl (R/S)": h_mottl_series,
    "nolds (DFA)": h_nolds_series,
    "hurst-exponent": h_hexp_series,
    "hurst-estimators": h_hest_series,
    "exp-hurst": h_eh_series,
}

header = f"{'Methode':<22} {'Mittel':>7} {'Std':>6} {'Min':>6} {'Max':>6} {'H<0.45':>8} {'H<0.50':>8}"
print(header)
print("-" * 70)

for name, arr in methods.items():
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"{name:<22}  {'FEHLER / keine Daten':>40}")
        continue
    mean = np.mean(valid)
    std = np.std(valid)
    mn = np.min(valid)
    mx = np.max(valid)
    pct_revert = np.mean(valid < 0.45) * 100
    pct_50 = np.mean(valid < 0.50) * 100
    print(
        f"{name:<22} {mean:>7.4f} {std:>6.4f} {mn:>6.4f} {mx:>6.4f} {pct_revert:>7.1f}% {pct_50:>7.1f}%"
    )

# ── Korrelations-Matrix ────────────────────────────────────────────────────────
print()
print("KORRELATION (unsere vs. andere):")
print("-" * 40)
ref = pd.Series(h_ours)
for name, arr in list(methods.items())[1:]:
    s = pd.Series(arr)
    corr = ref.corr(s)
    print(f"  {name:<22} r = {corr:+.4f}")

# ── Wichtigster Wert: Mittel-H auf NQ ────────────────────────────────────────
print()
print("FAZIT:")
print("-" * 40)
valid_ours = h_ours[~np.isnan(h_ours) & (h_ours != 0.5)]
if len(valid_ours) > 0:
    print(f"  Unsere H-Werte (ohne NaN/0.5): Ø = {np.mean(valid_ours):.4f}")
    print(
        f"  NQ ist {'mean-reverting' if np.mean(valid_ours) < 0.45 else 'neutral/trending'} "
        f"(H Ø {np.mean(valid_ours):.3f})"
    )

print()
print("Script fertig.")
