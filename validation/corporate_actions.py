"""Flag Tadawul sessions that cannot be ordinary ±10% limit moves.

Yahoo .SR is stored two ways:
  validation/prices/*_raw.csv   raw Close and AdjClose
  daily_study/cache/*_ohlcv.csv adjusted OHLC actually fed to the model

A day is suspect when the adjusted close-to-close move exceeds 15%, or when
the adjustment ratio AdjClose/Close jumps. The cash market limit is ±10%,
so a larger adjusted move was not a normal continuous-trading print.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "validation" / "prices"
OUT = ROOT / "validation" / "corporate_actions.csv"
SUMMARY = ROOT / "validation" / "corporate_actions_summary.csv"


def _load(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    if "AdjClose" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "AdjClose"})
    return df


def scan_pipeline():
    """Flags on the adjusted tape the model actually trains on."""
    cache = ROOT / "daily_study" / "cache"
    universe = json.loads((ROOT / "tasi_universe.json").read_text())
    names = {a["yahoo"].replace(".", "_"): a for a in universe["assets"]}
    rows = []
    summaries = []
    for path in sorted(cache.glob("*_SR_ohlcv.csv")):
        key = path.name[: -len("_ohlcv.csv")]
        asset = names.get(key, {})
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        ret = df["Close"].pct_change()
        vol = float(ret.std() * np.sqrt(250)) if ret.notna().sum() > 30 else np.nan
        vol_2016 = float(ret[ret.index >= "2016-01-01"].std() * np.sqrt(250)) if (ret.index >= "2016-01-01").sum() > 30 else np.nan
        hit = ret[ret.abs() > 0.15]
        summaries.append({
            "ticker": asset.get("ticker", key),
            "yahoo": asset.get("yahoo", key),
            "name": asset.get("name", ""),
            "n_sessions": int(len(df)),
            "ann_vol_250": None if not np.isfinite(vol) else round(vol, 4),
            "ann_vol_from_2016": None if not np.isfinite(vol_2016) else round(vol_2016, 4),
            "n_gt_15": int(len(hit)),
            "n_gt_15_from_2016": int((hit.index >= "2016-01-01").sum()) if len(hit) else 0,
            "max_abs_ret": None if not len(hit) else round(float(hit.abs().max()), 4),
        })
        for dt, value in hit.items():
            rows.append({
                "ticker": asset.get("ticker", key),
                "yahoo": asset.get("yahoo", key),
                "name": asset.get("name", ""),
                "date": str(pd.Timestamp(dt).date()),
                "adj_return": round(float(value), 6),
                "in_or_after_2016": bool(pd.Timestamp(dt) >= pd.Timestamp("2016-01-01")),
            })
    events = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries).sort_values(["n_gt_15_from_2016", "max_abs_ret"], ascending=False)
    events.to_csv(ROOT / "validation" / "pipeline_jumps.csv", index=False)
    summary.to_csv(ROOT / "validation" / "pipeline_jump_summary.csv", index=False)
    print(f"pipeline events={len(events)} names_with_any={int((summary.n_gt_15>0).sum())} "
          f"names_from_2016={int((summary.n_gt_15_from_2016>0).sum())}")
    print(summary.head(15).to_string(index=False))


def scan():
    universe = json.loads((ROOT / "tasi_universe.json").read_text())
    names = {a["yahoo"].replace(".", "_"): a for a in universe["assets"]}
    rows = []
    summaries = []
    for path in sorted(RAW.glob("*_raw.csv")):
        key = path.name[: -len("_raw.csv")]
        asset = names.get(key, {})
        df = _load(path)
        if "Close" not in df.columns:
            continue
        raw_ret = df["Close"].pct_change()
        if "AdjClose" in df.columns:
            adj = df["AdjClose"]
            ratio = adj / df["Close"].replace(0, np.nan)
        else:
            adj = df["Close"]
            ratio = pd.Series(1.0, index=df.index)
        adj_ret = adj.pct_change()
        ratio_jump = ratio.pct_change().abs()
        flagged = (adj_ret.abs() > 0.15) | (raw_ret.abs() > 0.15) | (ratio_jump > 0.02)
        vol = float(adj_ret.std() * np.sqrt(250)) if adj_ret.notna().sum() > 30 else np.nan
        summaries.append({
            "ticker": asset.get("ticker", key),
            "yahoo": asset.get("yahoo", key),
            "name": asset.get("name", ""),
            "n_sessions": int(len(df)),
            "ann_vol_adj_250": round(vol, 4) if np.isfinite(vol) else None,
            "n_adj_gt_15": int((adj_ret.abs() > 0.15).sum()),
            "n_raw_gt_15": int((raw_ret.abs() > 0.15).sum()),
            "n_ratio_jump": int((ratio_jump > 0.02).sum()),
            "max_abs_adj_ret": round(float(adj_ret.abs().max()), 4) if adj_ret.notna().any() else None,
            "max_abs_raw_ret": round(float(raw_ret.abs().max()), 4) if raw_ret.notna().any() else None,
        })
        hit = df.loc[flagged.fillna(False)]
        for dt, rec in hit.iterrows():
            rows.append({
                "ticker": asset.get("ticker", key),
                "yahoo": asset.get("yahoo", key),
                "name": asset.get("name", ""),
                "date": str(pd.Timestamp(dt).date()),
                "raw_close": rec.get("Close"),
                "adj_close": rec.get("AdjClose", rec.get("Close")),
                "raw_return": None if pd.isna(raw_ret.loc[dt]) else round(float(raw_ret.loc[dt]), 6),
                "adj_return": None if pd.isna(adj_ret.loc[dt]) else round(float(adj_ret.loc[dt]), 6),
                "adj_over_raw": None if pd.isna(ratio.loc[dt]) else round(float(ratio.loc[dt]), 6),
                "ratio_jump": None if pd.isna(ratio_jump.loc[dt]) else round(float(ratio_jump.loc[dt]), 6),
                "adj_gt_15": bool(abs(adj_ret.loc[dt]) > 0.15) if pd.notna(adj_ret.loc[dt]) else False,
                "raw_gt_15": bool(abs(raw_ret.loc[dt]) > 0.15) if pd.notna(raw_ret.loc[dt]) else False,
                "ratio_jump_gt_2pct": bool(ratio_jump.loc[dt] > 0.02) if pd.notna(ratio_jump.loc[dt]) else False,
            })
    events = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries).sort_values(
        ["n_adj_gt_15", "max_abs_adj_ret"], ascending=False
    )
    events.to_csv(OUT, index=False)
    summary.to_csv(SUMMARY, index=False)
    print(f"events={len(events)} names={len(summary)}")
    print(summary.head(20).to_string(index=False))
    print("names with any adjusted |return|>15%:", int((summary["n_adj_gt_15"] > 0).sum()))
    print("names with ann vol > 0.80:", int((summary["ann_vol_adj_250"] > 0.80).sum()))


if __name__ == "__main__":
    scan_pipeline()
    scan()
