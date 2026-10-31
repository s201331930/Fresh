"""Score one TASI name under an audit flag.

The original walk-forward, cost (2 bps), seed (42), and feature code are called
unchanged. Flags only change the inputs handed to that code:

  --macro-lag 0     published calendar join (default)
  --macro-lag 1     prints strictly before the Saudi session
  --macro-lag none  macro columns removed
  --fill close      close(t) -> close(t+1) label (default)
  --fill next_open  open(t+1) -> open(t+2)
  --fill delay1     close(t+1) -> close(t+2)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import daily_vs_weekly as dv
import tasi_periods as tp
from validation.macro_clock import prepare_macro

RUNS = ROOT / "validation" / "runs"
SEED = 42


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def frame_hash(daily, macro):
    payload = pd.util.hash_pandas_object(daily["Close"].astype(float), index=True).values.tobytes()
    payload += pd.util.hash_pandas_object(macro.astype(float), index=True).values.tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def neutralize_jumps(daily, limit=0.15):
    """Remove |close-to-close| moves above `limit` from the price path.

    Later returns are unchanged. The flagged session's OHLC is scaled so its
    close equals the previous close. Volume is left alone. Returns the cleaned
    frame and the session dates whose labels (the previous bar) should not be
    trained on.
    """
    px = daily.copy()
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in px.columns]
    rets = px["Close"].pct_change()
    flags = list(rets.index[rets.abs() > limit])
    label_dates = []
    for dt in reversed(flags):
        loc = px.index.get_loc(dt)
        if not isinstance(loc, (int, np.integer)) or loc == 0:
            continue
        prev = float(px["Close"].iloc[loc - 1])
        cur = float(px["Close"].iloc[loc])
        if not np.isfinite(prev) or not np.isfinite(cur) or cur == 0.0:
            continue
        factor = prev / cur
        px.iloc[loc:, px.columns.get_indexer(price_cols)] = (
            px.iloc[loc:][price_cols].to_numpy(dtype=float) * factor
        )
        label_dates.append(px.index[loc - 1])
    return px, label_dates


def walk_forward_drop(X, y, candidates, step, top_n, oos_start, drop_index):
    """Same fold logic as daily_vs_weekly.walk_forward, excluding drop_index from fits."""
    drop = pd.DatetimeIndex(drop_index)
    idx = X.index
    oos_start = pd.Timestamp(oos_start)
    start = int(idx.searchsorted(oos_start))
    n = len(X)
    if start < 80 or start >= n - 40:
        raise RuntimeError(f"Bad OOS start location {start} of {n}")
    parts = []
    cursor = start
    while cursor < n:
        end = min(cursor + step, n)
        n_val = max(40, int(cursor * 0.12))
        fit_end = cursor - n_val
        if fit_end < 60:
            raise RuntimeError("Training window too short for a validation tail")

        def _kept(a, b):
            part_x = X.iloc[a:b]
            part_y = y.iloc[a:b]
            mask = ~part_x.index.isin(drop)
            return part_x.loc[mask], part_y.loc[mask]

        x_fit, y_fit = _kept(0, fit_end)
        x_val, y_val = _kept(fit_end, cursor)
        chosen = dv.select_features(x_fit, y_fit, x_val, y_val, candidates, top_n)
        pred = dv._fit_ensemble(
            x_fit, y_fit, x_val, y_val, X.iloc[cursor:end], chosen,
        )
        parts.append(pd.Series(pred, index=idx[cursor:end], name="prediction"))
        cursor = end
    predictions = pd.concat(parts)
    return predictions[~predictions.index.duplicated(keep="last")].sort_index()


def retarget(daily, X, y, fill):
    if fill == "close":
        return X, y
    px = dv.normalize_index(daily)
    if fill == "next_open":
        new = (px["Open"].shift(-2) / px["Open"].shift(-1) - 1.0) * 100.0
    elif fill == "delay1":
        new = (px["Close"].shift(-2) / px["Close"].shift(-1) - 1.0) * 100.0
    else:
        raise ValueError(fill)
    new = new.reindex(X.index).replace([np.inf, -np.inf], np.nan)
    mask = new.notna()
    return X.loc[mask], new.loc[mask]


def _round_metrics(row):
    keep = [
        "sharpe", "cagr", "max_drawdown", "in_market_pct", "n_bars", "n_years",
        "bnh_sharpe", "bnh_cagr", "bnh_max_drawdown", "turnover_per_year",
    ]
    return {k: row[k] for k in keep}


def score_asset(asset, macro_lag="0", fill="close", neutralize=False):
    t0 = time.time()
    dv.N_JOBS = 1
    code = asset["ticker"]
    daily = dv.fetch_ohlcv(asset["yahoo"])
    macro = dv.fetch_macro_daily()
    digest = frame_hash(daily, macro)
    dropped = []
    if neutralize:
        daily, dropped = neutralize_jumps(daily)
    aligned = prepare_macro(macro, daily.index, macro_lag)
    X, y, cols = dv.build_daily_frame(daily, aligned)
    X, y = retarget(daily, X, y, fill)
    schedule = tp.choose_schedule(X.index)
    if schedule is None:
        return {
            "ticker": code,
            "status": "skipped",
            "reason": "schedule",
            "macro_lag": str(macro_lag),
            "fill": fill,
            "seed": SEED,
            "git_commit": git_commit(),
            "data_hash": digest,
        }
    oos, kind, grid, periods = schedule
    if neutralize and dropped:
        predictions = walk_forward_drop(
            X, y, cols, tp.STEP, tp.TOP_N, oos, dropped,
        )
    else:
        predictions, _picked = dv.walk_forward(
            X, y, cols, tp.STEP, tp.TOP_N, oos_start=oos, quiet=True,
        )
    frame = pd.DataFrame({"prediction": predictions, "actual": y}).dropna().sort_index()
    frame = frame[frame.index >= oos]
    binary0 = dv.positions_binary(frame["prediction"], 0.0)
    scored = []
    for label, start, end in periods:
        row = tp._period_payload(frame, binary0, start, end)
        if row is None:
            continue
        block = frame[(frame.index >= start) & (frame.index < end)]
        scored.append({
            "label": label,
            "start": str(block.index[0].date()),
            "end": str(block.index[-1].date()),
            "strategies": {"Binary (t=0)": _round_metrics(row)},
        })
    if len(scored) < 2:
        return {
            "ticker": code,
            "status": "skipped",
            "reason": "fewer than two periods",
            "macro_lag": str(macro_lag),
            "fill": fill,
            "seed": SEED,
            "git_commit": git_commit(),
            "data_hash": digest,
        }
    full_start = pd.Timestamp(scored[0]["start"])
    full_end = frame.index.max()
    full = tp._period_payload(frame, binary0, full_start, full_end + pd.Timedelta(days=1))
    pred_path = _pred_path(macro_lag, fill, code, neutralize)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    frame[["prediction", "actual"]].to_csv(pred_path, index_label="date")
    payload = {
        "ticker": code,
        "yahoo": asset["yahoo"],
        "name": asset["name"],
        "status": "ok",
        "macro_lag": str(macro_lag),
        "fill": fill,
        "seed": SEED,
        "git_commit": git_commit(),
        "data_hash": digest,
        "cost_bps": dv.COST_BPS,
        "bars_per_year": 252,
        "schedule": kind,
        "grid": grid,
        "oos_start": str(pd.Timestamp(oos).date()),
        "neutralize_jumps": bool(neutralize),
        "n_neutralized_labels": int(len(dropped)),
        "n_features": int(len(cols)),
        "n_predictions": int(len(frame)),
        "full": {
            "start": str(full_start.date()),
            "end": str(full_end.date()),
            "strategies": {"Binary (t=0)": _round_metrics(full)} if full else {},
        },
        "periods": scored,
        "seconds": round(time.time() - t0, 1),
    }
    return payload


def _tag(macro_lag, fill, neutralize=False):
    lag = {"0": "lag0", "1": "lag1", "none": "nomacro"}.get(str(macro_lag), str(macro_lag))
    if fill == "close":
        tag = lag
    else:
        tag = f"{lag}_{fill}"
    if neutralize:
        tag = f"{tag}_clean"
    return tag


def _out_path(macro_lag, fill, code, neutralize=False):
    return RUNS / _tag(macro_lag, fill, neutralize) / f"{code}.json"


def _pred_path(macro_lag, fill, code, neutralize=False):
    return RUNS / _tag(macro_lag, fill, neutralize) / "predictions" / f"{code}.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker")
    parser.add_argument("--macro-lag", default="0")
    parser.add_argument("--fill", default="close", choices=["close", "next_open", "delay1"])
    parser.add_argument("--neutralize-jumps", action="store_true")
    args = parser.parse_args()
    asset = next(a for a in tp.load_assets() if a["ticker"] == args.ticker)
    dest = _out_path(args.macro_lag, args.fill, asset["ticker"], args.neutralize_jumps)
    if dest.exists():
        print(f"CACHED {asset['ticker']}", flush=True)
        return
    payload = score_asset(asset, args.macro_lag, args.fill, neutralize=args.neutralize_jumps)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2))
    t0 = payload.get("full", {}).get("strategies", {}).get("Binary (t=0)", {})
    print(
        f"DONE {asset['ticker']} lag={args.macro_lag} fill={args.fill} "
        f"sharpe={t0.get('sharpe')} bnh={t0.get('bnh_sharpe')} "
        f"{payload.get('seconds', 0)/60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
