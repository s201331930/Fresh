"""Download Tadawul bars once, in the pipeline cache format, and keep raw closes.

The pipeline cache matches daily_vs_weekly.fetch_ohlcv: adjusted OHLC, volume > 0.
The raw file keeps Close and Adj Close so corporate-action jumps can be flagged
without a second vendor pull.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import daily_vs_weekly as dv

RAW = ROOT / "validation" / "prices"
RAW.mkdir(parents=True, exist_ok=True)


def _universe():
    payload = json.loads((ROOT / "tasi_universe.json").read_text())
    return payload["assets"]


def _write_one(yahoo, raw):
    if raw is None or len(raw) == 0:
        raise RuntimeError("empty")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame = frame.rename(columns={"Adj Close": "AdjClose"})
    keep = [c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in frame.columns]
    frame = frame[keep]
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.to_csv(RAW / f"{yahoo.replace('.', '_')}_raw.csv")

    adj = frame.copy()
    if "AdjClose" in adj.columns and "Close" in adj.columns:
        close = adj["Close"].replace(0, np.nan)
        a2p = adj["AdjClose"] / close
        for col in ["Open", "High", "Low"]:
            if col in adj.columns:
                adj[col] = adj[col] * a2p
        adj["Close"] = adj["AdjClose"]
    out = adj[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    out = out[out["Volume"] > 0]
    cache = dv.CACHE_DIR / f"{yahoo.replace('.', '_')}_ohlcv.csv"
    out.to_csv(cache)
    return len(out)


def prefetch(only_scored=True):
    dv.fetch_macro_daily()
    assets = _universe()
    if only_scored:
        scored = set()
        for path in (ROOT / "tasi_periods").glob("*.json"):
            if path.name == "summary.json":
                continue
            rec = json.loads(path.read_text())
            if rec.get("status") == "ok":
                scored.add(rec["ticker"])
        assets = [a for a in assets if a["ticker"] in scored]
    missing = []
    for asset in assets:
        cache = dv.CACHE_DIR / f"{asset['yahoo'].replace('.', '_')}_ohlcv.csv"
        raw = RAW / f"{asset['yahoo'].replace('.', '_')}_raw.csv"
        if not (cache.exists() and raw.exists()):
            missing.append(asset)
    print(f"Prefetch {len(missing)} of {len(assets)}", flush=True)
    for i in range(0, len(missing), 30):
        batch = missing[i:i + 30]
        symbols = [a["yahoo"] for a in batch]
        print(f"[prefetch] {symbols[0]} .. {symbols[-1]}", flush=True)
        try:
            raw = dv.yf.download(
                symbols, start="2000-01-01", auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as exc:
            print(f"  batch failed: {exc}", flush=True)
            raw = None
        for asset in batch:
            yahoo = asset["yahoo"]
            try:
                if raw is None or len(raw) == 0:
                    raise RuntimeError("batch empty")
                if isinstance(raw.columns, pd.MultiIndex) and yahoo in raw.columns.get_level_values(0):
                    one = raw[yahoo].dropna(how="all")
                elif len(batch) == 1:
                    one = raw
                else:
                    raise RuntimeError("symbol missing from batch")
                n = _write_one(yahoo, one)
                print(f"  {yahoo}: {n}", flush=True)
            except Exception as exc:
                print(f"  miss {yahoo}: {exc}", flush=True)
                try:
                    one = dv.yf.Ticker(yahoo).history(start="2000-01-01", auto_adjust=False)
                    n = _write_one(yahoo, one)
                    print(f"  {yahoo}: {n} (single)", flush=True)
                except Exception as exc2:
                    print(f"  fail {yahoo}: {exc2}", flush=True)


if __name__ == "__main__":
    scored = "--all" not in sys.argv
    prefetch(only_scored=scored)
