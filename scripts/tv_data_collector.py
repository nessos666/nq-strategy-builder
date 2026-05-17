#!/usr/bin/env python3
"""
TradingView Data Collector – Autonomes Script
Verbindet sich direkt via CDP (Port 9222) mit TradingView Desktop.
Sammelt OHLCV+Volume fuer beliebige Timeframes und speichert als Parquet.

Nutzung:
    python3 -u scripts/tv_data_collector.py 2>&1 | tee /tmp/tv_collector.log

Voraussetzung: TradingView Desktop laeuft mit --remote-debugging-port=9222
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import websockets
except ImportError:
    print("pip install websockets")
    sys.exit(1)

# ─── Config ──────────────────────────────────────────────────────────────────
CDP_PORT = 9222
OUTPUT_DIR = Path(
    Path(__file__).resolve().parent.parent / "data"
    "/TRADINGPROJEKT/Kerzendatenbank/NQ/tradingview"
)

TIMEFRAMES = [
    # (tv_code, ordner, dateiname, scroll_pause_sek)
    ("15S", "15sek", "nq_15s_tv.parquet", 3),
    ("1", "1min", "nq_1m_tv.parquet", 2),
    ("5", "5min", "nq_5m_tv.parquet", 2),
    ("15", "15min", "nq_15m_tv.parquet", 2),
    ("60", "1h", "nq_1h_tv.parquet", 2),
]

SCROLL_AMOUNT = 10000
MAX_STALE_SCROLLS = 10
CHUNK_SIZE = 5000


# ─── CDP Verbindung ─────────────────────────────────────────────────────────
def find_tv_target() -> str:
    url = f"http://localhost:{CDP_PORT}/json/list"
    resp = urllib.request.urlopen(url, timeout=5)
    targets = json.loads(resp.read())
    for t in targets:
        if "tradingview.com/chart" in t.get("url", "").lower():
            return t["webSocketDebuggerUrl"]
    raise RuntimeError("TradingView chart nicht gefunden!")


class CDPClient:
    def __init__(self, ws):
        self.ws = ws
        self._id = 0
        self._center = None  # cached chart center

    async def start(self):
        await self._send("Runtime.enable")

    async def _send(
        self, method: str, params: dict | None = None, timeout: float = 30
    ) -> dict:
        """CDP command senden und auf Antwort warten (Events ueberspringen)."""
        self._id += 1
        mid = self._id
        cmd: dict = {"id": mid, "method": method}
        if params:
            cmd["params"] = params
        await self.ws.send(json.dumps(cmd))
        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
            data = json.loads(raw)
            if data.get("id") == mid:
                return data
            # Events (kein "id") werden uebersprungen

    async def js(self, expression: str) -> str | None:
        """JavaScript ausfuehren, Ergebnis als String."""
        resp = await self._send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True},
        )
        res = resp.get("result", {})
        if res.get("exceptionDetails"):
            desc = (
                res["exceptionDetails"]
                .get("exception", {})
                .get("description", "unknown")
            )
            raise RuntimeError(f"JS: {desc[:200]}")
        return res.get("result", {}).get("value")

    async def scroll_left(self):
        """Chart nach links scrollen via internes Chart-Model."""
        await self.js("""
            (function() {
                try {
                    var w = window.TradingViewApi.activeChart();
                    var model = w._chartWidget.model();
                    model.scrollChart(-200);
                    return 'model';
                } catch(e) {
                    return 'err:' + e.message;
                }
            })()
        """)


# ─── Hilfsfunktionen ────────────────────────────────────────────────────────
async def get_bar_info(cdp: CDPClient) -> dict:
    raw = await cdp.js("""
        (function() {
            try {
                var b = window.TradingViewApi.activeChart().getSeries().data().bars();
                if (!b || b.size() === 0) return '{"total":0}';
                return JSON.stringify({
                    total: b.size(),
                    first_date: new Date(b.first().value[0]*1000).toISOString().slice(0,19),
                    last_date: new Date(b.last().value[0]*1000).toISOString().slice(0,19)
                });
            } catch(e) { return '{"total":0,"err":"'+e.message+'"}'; }
        })()
    """)
    return json.loads(raw) if raw else {"total": 0}


async def store_bars(cdp: CDPClient) -> int:
    raw = await cdp.js("""
        (function() {
            window._tvB = [];
            var b = window.TradingViewApi.activeChart().getSeries().data().bars();
            b.fold(function(a, v) { window._tvB.push(v); return a; }, 0);
            return String(window._tvB.length);
        })()
    """)
    return int(raw) if raw else 0


async def extract_chunk(cdp: CDPClient, start: int, end: int) -> list:
    raw = await cdp.js(f"JSON.stringify(window._tvB.slice({start},{end}))")
    return json.loads(raw) if raw else []


async def dismiss_dialog(cdp: CDPClient):
    try:
        await cdp.js("""
            (function() {
                var b = document.querySelectorAll('button');
                for (var i = 0; i < b.length; i++)
                    if (b[i].textContent.trim() === 'Verbinden') { b[i].click(); return 1; }
                return 0;
            })()
        """)
    except Exception:
        pass


# ─── Sammlung pro Timeframe ─────────────────────────────────────────────────
async def collect_timeframe(cdp: CDPClient, tf_code: str, pause: float) -> list:
    # 1. Timeframe umschalten
    print(f"  [1/4] Timeframe -> {tf_code}", flush=True)
    await cdp.js(
        f"window.TradingViewApi.activeChart().setResolution('{tf_code}'); 'ok'"
    )
    await asyncio.sleep(5)
    await dismiss_dialog(cdp)
    cdp._center = None  # Reset nach TF-Wechsel

    # 2. Zum neuesten Bar
    print("  [2/4] Zum neuesten Bar...", flush=True)
    try:
        await cdp.js(
            "window.TradingViewApi.activeChart().executeActionById('timeScaleReset'); 'ok'"
        )
    except Exception:
        pass
    await asyncio.sleep(3)

    # 3. Scroll-Loop
    print("  [3/4] Historie laden...", flush=True)
    prev_total = 0
    stale = 0

    for scroll_n in range(500):  # max 500 scrolls
        info = await get_bar_info(cdp)
        total = info.get("total", 0)
        first = info.get("first_date", "?")

        if total > prev_total:
            print(f"         #{scroll_n:3d}: {total:>8,} bars | ab {first}", flush=True)
            stale = 0
        else:
            stale += 1

        if stale >= MAX_STALE_SCROLLS:
            break

        prev_total = total
        await cdp.scroll_left()
        await asyncio.sleep(pause)

        if scroll_n % 15 == 14:
            await dismiss_dialog(cdp)

    print(f"         Fertig: {prev_total:,} bars", flush=True)

    # 4. Extrahieren
    print("  [4/4] Extrahieren...", flush=True)
    stored = await store_bars(cdp)
    print(f"         {stored:,} bars gespeichert", flush=True)

    all_bars: list = []
    for s in range(0, stored, CHUNK_SIZE):
        e = min(s + CHUNK_SIZE, stored)
        chunk = await extract_chunk(cdp, s, e)
        all_bars.extend(chunk)
        print(f"         {len(all_bars):>8,} / {stored:,}", flush=True)

    return all_bars


def save_parquet(bars: list, path: Path) -> pd.DataFrame:
    df = pd.DataFrame(bars, columns=["time", "Open", "High", "Low", "Close", "Volume"])
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)
    df.index.name = None
    df = df.drop(columns=["time"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_parquet(path)
    return df


# ─── Main ────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60, flush=True)
    print("  TradingView Data Collector", flush=True)
    print(f"  Timeframes: {', '.join(t[0] for t in TIMEFRAMES)}", flush=True)
    print("=" * 60, flush=True)

    ws_url = find_tv_target()
    print(f"  Target: {ws_url[:50]}...\n", flush=True)

    async with websockets.connect(ws_url, max_size=100_000_000) as ws:
        cdp = CDPClient(ws)
        await cdp.start()

        for tf_code, folder, filename, pause in TIMEFRAMES:
            print(f"\n{'─' * 50}", flush=True)
            print(f"  {tf_code} -> {folder}/{filename}", flush=True)
            print(f"{'─' * 50}", flush=True)

            try:
                bars = await collect_timeframe(cdp, tf_code, pause)
            except Exception as e:
                print(f"  FEHLER: {e}", flush=True)
                import traceback

                traceback.print_exc()
                continue

            if not bars:
                print("  Keine Bars!", flush=True)
                continue

            out_dir = OUTPUT_DIR / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename
            df = save_parquet(bars, out_path)

            span = (df.index[-1] - df.index[0]).days
            print(f"\n  GESPEICHERT: {out_path}", flush=True)
            print(f"    Bars:  {len(df):>10,}", flush=True)
            print(f"    Range: {df.index[0]} -> {df.index[-1]}", flush=True)
            print(f"    Tage:  {span}", flush=True)
            print(
                f"    Size:  {out_path.stat().st_size / 1024 / 1024:.1f} MB", flush=True
            )

        # Zurueck auf 1min
        print("\n  Chart -> 1min...", flush=True)
        try:
            await cdp.js("window.TradingViewApi.activeChart().setResolution('1'); 'ok'")
        except Exception:
            pass

    print("\n" + "=" * 60, flush=True)
    print("  FERTIG!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
