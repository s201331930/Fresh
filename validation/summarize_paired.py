"""Paired Sharpe differences for two audit configurations. Same name, same period."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "validation" / "runs"


def _rows(tag):
    rows = []
    for path in sorted((RUNS / tag).glob("*.json")):
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        for period in rec.get("periods", []):
            metrics = period["strategies"]["Binary (t=0)"]
            rows.append({
                "ticker": rec["ticker"],
                "period": period["label"],
                "sharpe": metrics["sharpe"],
                "bnh_sharpe": metrics["bnh_sharpe"],
                "cagr": metrics["cagr"],
                "bnh_cagr": metrics["bnh_cagr"],
                "gap": metrics["sharpe"] - metrics["bnh_sharpe"],
                "cagr_gap": metrics["cagr"] - metrics["bnh_cagr"],
            })
        full = rec.get("full", {}).get("strategies", {}).get("Binary (t=0)")
        if full:
            rows.append({
                "ticker": rec["ticker"],
                "period": "full",
                "sharpe": full["sharpe"],
                "bnh_sharpe": full["bnh_sharpe"],
                "cagr": full["cagr"],
                "bnh_cagr": full["bnh_cagr"],
                "gap": full["sharpe"] - full["bnh_sharpe"],
                "cagr_gap": full["cagr"] - full["bnh_cagr"],
            })
    return pd.DataFrame(rows)


def paired(base_tag, alt_tag):
    base = _rows(base_tag)
    alt = _rows(alt_tag)
    merged = base.merge(alt, on=["ticker", "period"], suffixes=("_base", "_alt"))
    merged["d_sharpe"] = merged["sharpe_alt"] - merged["sharpe_base"]
    merged["d_gap"] = merged["gap_alt"] - merged["gap_base"]
    merged["d_cagr_gap"] = merged["cagr_gap_alt"] - merged["cagr_gap_base"]
    return merged


def _rate(frame, col):
    return float((frame[col] > 0).mean()) if len(frame) else float("nan")


def report(base_tag, alt_tag, out_name):
    merged = paired(base_tag, alt_tag)
    out = ROOT / "validation" / out_name
    merged.to_csv(out, index=False)
    periods = merged[merged["period"] != "full"]
    full = merged[merged["period"] == "full"]
    lines = [
        f"paired {alt_tag} minus {base_tag}",
        f"period rows {len(periods)} names {periods['ticker'].nunique() if len(periods) else 0}",
        f"median d_sharpe {periods['d_sharpe'].median():+.4f}" if len(periods) else "no periods",
        f"median d_gap {periods['d_gap'].median():+.4f}" if len(periods) else "",
        f"beat rate base { _rate(periods, 'gap_base'):.3%} alt {_rate(periods, 'gap_alt'):.3%}" if len(periods) else "",
        f"full names {len(full)} median d_sharpe {full['d_sharpe'].median():+.4f}" if len(full) else "",
        f"full beat base {_rate(full, 'gap_base'):.3%} alt {_rate(full, 'gap_alt'):.3%}" if len(full) else "",
        f"full median sharpe base {full['sharpe_base'].median():.3f} alt {full['sharpe_alt'].median():.3f}" if len(full) else "",
        f"full median gap base {full['gap_base'].median():+.3f} alt {full['gap_alt'].median():+.3f}" if len(full) else "",
        f"full median cagr gap base {full['cagr_gap_base'].median():+.2f} alt {full['cagr_gap_alt'].median():+.2f}" if len(full) else "",
    ]
    text = "\n".join(lines)
    print(text)
    (ROOT / "validation" / (out_name.replace(".csv", ".txt"))).write_text(text + "\n")
    return merged


if __name__ == "__main__":
    base = sys.argv[1]
    alt = sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else f"paired_{base}_vs_{alt}.csv"
    report(base, alt, name)
