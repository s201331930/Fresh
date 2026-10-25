"""
TASI daily-strategy consistency across calendar periods.

Every Main Market name in the Tadawul All Share list is scored on the daily
feature frame. The weekly builder stays in daily_vs_weekly.py. This run still
forms the Thursday weekly bars for each name so that series is ready for a
later join. It does not fit the weekly model.

No look-ahead:
  - Walk-forward predictions at day t use only training bars before that fold.
  - A period's threshold and Scaled divisor are fit on predictions strictly
    before the period. The frozen pair is fit once, on the block before the
    first scored period, then applied unchanged.
  - Adaptive looks back at past bars only.
  - Weekly bars are aggregated from daily history through that week. They are
    not fed to this model.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import daily_vs_weekly as dv

ROOT = Path(__file__).parent
UNIVERSE = ROOT / "tasi_universe.json"
OUT = ROOT / "tasi_periods"
OUT.mkdir(exist_ok=True)

WORKERS = 3
MIN_PERIOD_BARS = 180
MIN_CALIB_BARS = 120
MIN_TRAIN = 400
LOOKBACK = 126
STEP = 126
TOP_N = 50

THREE_YEAR = [
    ("2013-2015", "2013-01-01", "2016-01-01"),
    ("2016-2018", "2016-01-01", "2019-01-01"),
    ("2019-2021", "2019-01-01", "2022-01-01"),
    ("2022-2026", "2022-01-01", "2027-01-01"),
]
TWO_YEAR = [
    ("2014-2015", "2014-01-01", "2016-01-01"),
    ("2016-2017", "2016-01-01", "2018-01-01"),
    ("2018-2019", "2018-01-01", "2020-01-01"),
    ("2020-2021", "2020-01-01", "2022-01-01"),
    ("2022-2023", "2022-01-01", "2024-01-01"),
    ("2024-2026", "2024-01-01", "2027-01-01"),
]

VARIANTS = [
    "Binary (t=0)",
    "Binary (t=opt)",
    "Binary (t=opt frozen)",
    "Scaled",
    "Scaled (frozen)",
    "Adaptive",
]


def load_assets():
    payload = json.loads(UNIVERSE.read_text())
    return payload["assets"]


def _cache(yahoo):
    return dv.CACHE_DIR / f"{yahoo.replace('.', '_')}_ohlcv.csv"


def _store_history(yahoo, raw):
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No daily data returned for {yahoo}")
    cols = {c: raw[c] for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns}
    df = pd.DataFrame(cols)
    if "Adj Close" in raw.columns and "Close" in df.columns:
        close = df["Close"].replace(0, np.nan)
        a2p = raw["Adj Close"] / close
        for col in ["Open", "High", "Low"]:
            if col in df.columns:
                df[col] = df[col] * a2p
        df["Close"] = raw["Adj Close"]
    df = dv.normalize_index(df)
    df = df.dropna(subset=["Close"])
    df = df[df["Volume"] > 0]
    if df.empty:
        raise RuntimeError(f"No positive-volume sessions for {yahoo}")
    df.to_csv(_cache(yahoo))
    return len(df)


def prefetch(assets):
    dv.fetch_macro_daily()
    missing = [a for a in assets if not _cache(a["yahoo"]).exists()]
    print(f"Prefetch {len(missing)} of {len(assets)} TASI names", flush=True)
    failed = []
    for i in range(0, len(missing), 30):
        batch = missing[i:i + 30]
        symbols = [a["yahoo"] for a in batch]
        print(f"[prefetch] {symbols[0]} .. {symbols[-1]} ({len(symbols)})", flush=True)
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
                if raw is None or not isinstance(raw.columns, pd.MultiIndex) or yahoo not in raw.columns.get_level_values(0):
                    dv.fetch_ohlcv(yahoo)
                else:
                    _store_history(yahoo, raw[yahoo].dropna(how="all"))
            except Exception as exc:
                failed.append((asset, f"{type(exc).__name__}: {exc}"))
                print(f"  miss {yahoo}: {exc}", flush=True)
    return failed


def _bars(index, start, end):
    return int(((index >= start) & (index < end)).sum())


def _valid_periods(index, grid):
    found = []
    for label, start, end in grid:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if _bars(index, start, end) < MIN_PERIOD_BARS:
            continue
        if int((index < start).sum()) < MIN_CALIB_BARS:
            continue
        found.append((label, start, end))
    return found


def choose_schedule(index):
    if len(index) < MIN_TRAIN + MIN_CALIB_BARS + MIN_PERIOD_BARS:
        return None
    pos_2010 = int(index.searchsorted(pd.Timestamp("2010-01-01")))
    if pos_2010 >= MIN_TRAIN and len(index) - pos_2010 >= MIN_CALIB_BARS + MIN_PERIOD_BARS:
        oos = pd.Timestamp("2010-01-01")
        kind = "fixed_2010"
    else:
        oos = pd.Timestamp(index[MIN_TRAIN])
        kind = "shifted"
    # Period rules see only the dates the walk-forward will actually predict.
    future = index[index >= oos]
    three = _valid_periods(future, THREE_YEAR)
    two = _valid_periods(future, TWO_YEAR)
    # Prefer the longer blocks whenever a name can support two of them.
    if len(three) >= 2:
        return oos, kind, "3y", three
    if len(two) >= 2:
        return oos, kind, "2y", two
    return None


def _calib_before(frame, start):
    left = pd.Timestamp(start) - pd.DateOffset(years=3)
    calib = frame[(frame.index >= left) & (frame.index < start)]
    if len(calib) < MIN_CALIB_BARS:
        calib = frame[frame.index < start]
    if len(calib) > 756:
        calib = calib.iloc[-756:]
    if len(calib) < MIN_CALIB_BARS:
        return None
    return calib


def _scale_from(calib):
    positive = calib["prediction"][calib["prediction"] > 0]
    if len(positive):
        return float(positive.quantile(0.90))
    return float(calib["prediction"].abs().quantile(0.90))


def _score(positions, actual, start, end):
    metrics, _, _ = dv.score_window(
        positions, actual, 252, dv.COST_BPS, pd.Timestamp(start), pd.Timestamp(end),
    )
    return metrics


def _period_payload(frame, positions, start, end):
    block = frame[(frame.index >= start) & (frame.index < end)]
    metrics = _score(positions, frame["actual"], block.index[0], block.index[-1])
    if metrics is None:
        return None
    keep = [
        "sharpe", "cagr", "max_drawdown", "in_market_pct", "n_bars", "n_years",
        "bnh_sharpe", "bnh_cagr", "bnh_max_drawdown", "turnover_per_year",
    ]
    return {k: metrics[k] for k in keep}


def run_one(asset):
    code = asset["ticker"]
    dest = OUT / f"{code}.json"
    if dest.exists():
        return code, "cached"
    t0 = time.time()
    dv.N_JOBS = 1
    try:
        print(f"[{code}] {asset['name']}", flush=True)
        daily = dv.fetch_ohlcv(asset["yahoo"])
        weekly = dv.to_weekly(daily, "W-THU")
        if len(daily) < 1000:
            payload = _skip(asset, len(weekly), "fewer than 1000 sessions")
            dest.write_text(json.dumps(payload, indent=2))
            return code, "skipped"
        macro = dv.fetch_macro_daily()
        X, y, cols = dv.build_daily_frame(daily, macro)
        schedule = choose_schedule(X.index)
        if schedule is None:
            payload = _skip(asset, len(weekly), "not enough history for training and two evaluation periods")
            dest.write_text(json.dumps(payload, indent=2))
            return code, "skipped"
        oos, kind, grid, periods = schedule
        print(f"[{code}] {kind} {grid} oos={oos.date()} periods={len(periods)}", flush=True)
        predictions, _picked = dv.walk_forward(
            X, y, cols, STEP, TOP_N, oos_start=oos, quiet=True,
        )
        frame = pd.DataFrame({"prediction": predictions, "actual": y}).dropna().sort_index()
        frame = frame[frame.index >= oos]
        adaptive = dv.positions_adaptive(frame["prediction"], frame["actual"], LOOKBACK)
        binary0 = dv.positions_binary(frame["prediction"], 0.0)

        first_start = periods[0][1]
        frozen_calib = _calib_before(frame, first_start)
        if frozen_calib is None:
            raise RuntimeError("frozen calibration window is too short")
        frozen_threshold = dv.fit_threshold(frozen_calib["prediction"], frozen_calib["actual"])
        frozen_scale = _scale_from(frozen_calib)
        frozen_opt = dv.positions_binary(frame["prediction"], frozen_threshold)
        frozen_scaled = dv.positions_scaled(frame["prediction"], frozen_scale)

        scored_periods = []
        for label, start, end in periods:
            calib = _calib_before(frame, start)
            if calib is None:
                continue
            threshold = dv.fit_threshold(calib["prediction"], calib["actual"])
            scale = _scale_from(calib)
            positions = {
                "Binary (t=0)": binary0,
                "Binary (t=opt)": dv.positions_binary(frame["prediction"], threshold),
                "Binary (t=opt frozen)": frozen_opt,
                "Scaled": dv.positions_scaled(frame["prediction"], scale),
                "Scaled (frozen)": frozen_scaled,
                "Adaptive": adaptive,
            }
            strategies = {}
            for name, pos in positions.items():
                row = _period_payload(frame, pos, start, end)
                if row is not None:
                    strategies[name] = row
            if len(strategies) < len(VARIANTS):
                continue
            block = frame[(frame.index >= start) & (frame.index < end)]
            scored_periods.append({
                "label": label,
                "start": str(block.index[0].date()),
                "end": str(block.index[-1].date()),
                "refit_threshold": round(float(threshold), 4),
                "refit_scale": round(float(scale), 4),
                "strategies": strategies,
            })
        if len(scored_periods) < 2:
            payload = _skip(asset, len(weekly), "fewer than two usable periods after the walk-forward")
            dest.write_text(json.dumps(payload, indent=2))
            return code, "skipped"

        full_start = pd.Timestamp(scored_periods[0]["start"])
        full_end = frame.index.max()
        full_positions = {
            "Binary (t=0)": binary0,
            "Binary (t=opt)": frozen_opt,
            "Binary (t=opt frozen)": frozen_opt,
            "Scaled": frozen_scaled,
            "Scaled (frozen)": frozen_scaled,
            "Adaptive": adaptive,
        }
        full_strats = {}
        for name, pos in full_positions.items():
            row = _period_payload(frame, pos, full_start, full_end + pd.Timedelta(days=1))
            if row is not None:
                full_strats[name] = row
        eval_frame = frame[frame.index >= full_start]
        model = dv.eval_metrics(eval_frame["actual"], eval_frame["prediction"])
        payload = {
            "ticker": code,
            "yahoo": asset["yahoo"],
            "name": asset["name"],
            "status": "ok",
            "in_original_12": bool(asset.get("in_original_12")),
            "schedule": kind,
            "grid": grid,
            "oos_start": str(pd.Timestamp(oos).date()),
            "n_daily_bars": int(len(daily)),
            "n_weekly_bars": int(len(weekly)),
            "week_rule": "W-THU",
            "frozen_threshold": round(float(frozen_threshold), 4),
            "frozen_scale": round(float(frozen_scale), 4),
            "model": model,
            "full": {
                "start": str(full_start.date()),
                "end": str(full_end.date()),
                "strategies": full_strats,
            },
            "periods": scored_periods,
            "seconds": round(time.time() - t0, 1),
        }
        t0s = full_strats.get("Binary (t=0)", {})
        print(f"[{code}] full t=0 {t0s.get('sharpe')} vs bnh {t0s.get('bnh_sharpe')}  "
              f"periods={len(scored_periods)}  {payload['seconds']/60:.1f} min", flush=True)
        dest.write_text(json.dumps(payload, indent=2))
        return code, "ok"
    except Exception as exc:
        traceback.print_exc()
        payload = _skip(asset, None, f"{type(exc).__name__}: {exc}")
        payload["status"] = "error"
        dest.write_text(json.dumps(payload, indent=2))
        return code, payload["reason"]


def _skip(asset, n_weekly, reason):
    return {
        "ticker": asset["ticker"],
        "yahoo": asset.get("yahoo"),
        "name": asset.get("name"),
        "status": "skipped" if not str(reason).startswith(("Runtime", "Value", "Key", "Assertion")) else "error",
        "in_original_12": bool(asset.get("in_original_12")),
        "n_weekly_bars": n_weekly,
        "week_rule": "W-THU",
        "reason": reason,
    }


def _variant_stats(rows):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    gap = frame["sharpe"] - frame["bnh_sharpe"]
    by_period = {}
    for label, sub in frame.groupby("period"):
        g = sub["sharpe"] - sub["bnh_sharpe"]
        by_period[label] = {
            "n": int(len(sub)),
            "beat_bnh": int((g > 0).sum()),
            "median_sharpe": round(float(sub["sharpe"].median()), 3),
            "median_bnh": round(float(sub["bnh_sharpe"].median()), 3),
            "median_gap": round(float(g.median()), 3),
            "median_in_market": round(float(sub["in_market"].median()), 1),
        }
    majority = 0
    all_periods = 0
    names = 0
    for _, sub in frame.groupby("ticker"):
        names += 1
        beats = int(((sub["sharpe"] - sub["bnh_sharpe"]) > 0).sum())
        if beats > len(sub) / 2:
            majority += 1
        if beats == len(sub):
            all_periods += 1
    return {
        "ticker_periods": int(len(frame)),
        "beat_bnh": int((gap > 0).sum()),
        "median_sharpe": round(float(frame["sharpe"].median()), 3),
        "median_bnh": round(float(frame["bnh_sharpe"].median()), 3),
        "median_gap": round(float(gap.median()), 3),
        "median_cagr": round(float(frame["cagr"].median()), 2),
        "median_bnh_cagr": round(float(frame["bnh_cagr"].median()), 2),
        "median_dd": round(float(frame["dd"].median()), 2),
        "median_bnh_dd": round(float(frame["bnh_dd"].median()), 2),
        "median_in_market": round(float(frame["in_market"].median()), 1),
        "cash_lt_20": int((frame["in_market"] < 20).sum()),
        "names": names,
        "names_majority": majority,
        "names_all_periods": all_periods,
        "by_period": by_period,
    }


def summarize():
    records = []
    for path in sorted(OUT.glob("*.json")):
        if path.name == "summary.json":
            continue
        records.append(json.loads(path.read_text()))
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") == "error"]

    def collect(assets, section):
        bucket = {name: [] for name in VARIANTS}
        full_bucket = {name: [] for name in VARIANTS}
        for asset in assets:
            for period in asset["periods"]:
                for name, metrics in period["strategies"].items():
                    bucket[name].append({
                        "ticker": asset["ticker"],
                        "period": period["label"],
                        "sharpe": metrics["sharpe"],
                        "bnh_sharpe": metrics["bnh_sharpe"],
                        "cagr": metrics["cagr"],
                        "bnh_cagr": metrics["bnh_cagr"],
                        "dd": metrics["max_drawdown"],
                        "bnh_dd": metrics["bnh_max_drawdown"],
                        "in_market": metrics["in_market_pct"],
                    })
            for name, metrics in asset["full"]["strategies"].items():
                full_bucket[name].append({
                    "ticker": asset["ticker"],
                    "period": "full",
                    "sharpe": metrics["sharpe"],
                    "bnh_sharpe": metrics["bnh_sharpe"],
                    "cagr": metrics["cagr"],
                    "bnh_cagr": metrics["bnh_cagr"],
                    "dd": metrics["max_drawdown"],
                    "bnh_dd": metrics["bnh_max_drawdown"],
                    "in_market": metrics["in_market_pct"],
                })
        return {
            "section": section,
            "n": len(assets),
            "periods": {name: _variant_stats(rows) for name, rows in bucket.items()},
            "full": {name: _variant_stats(rows) for name, rows in full_bucket.items()},
        }

    summary = {
        "n_universe": len(load_assets()),
        "n_ok": len(ok),
        "n_skipped": len(skipped),
        "n_error": len(errors),
        "skipped": [{"ticker": r["ticker"], "name": r.get("name"), "reason": r.get("reason")} for r in skipped],
        "errors": [{"ticker": r["ticker"], "name": r.get("name"), "reason": r.get("reason")} for r in errors],
        "grid": {
            "3y": int(sum(r.get("grid") == "3y" for r in ok)),
            "2y": int(sum(r.get("grid") == "2y" for r in ok)),
            "fixed_2010": int(sum(r.get("schedule") == "fixed_2010" for r in ok)),
            "shifted": int(sum(r.get("schedule") == "shifted" for r in ok)),
        },
        "all": collect(ok, "all"),
        "original_12": collect([r for r in ok if r.get("in_original_12")], "original_12"),
        "rest": collect([r for r in ok if not r.get("in_original_12")], "rest"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(summary)
    t0 = summary["all"]["periods"]["Binary (t=0)"]
    print(f"\nCompleted {len(ok)}  skipped {len(skipped)}  errors {len(errors)}")
    if t0:
        print(f"Daily t=0 beats buy&hold in {t0['beat_bnh']}/{t0['ticker_periods']} ticker-periods")
        print(f"Names beating in a majority of periods: {t0['names_majority']}/{t0['names']}")
        print(f"Median gap {t0['median_gap']:+.3f}  median Sharpe {t0['median_sharpe']} vs {t0['median_bnh']}")
    return summary


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = summary["all"]["periods"].get("Binary (t=0)") or {}
    periods = t0.get("by_period") or {}
    if not periods:
        return
    labels = list(periods.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [periods[k]["median_sharpe"] for k in labels], width, color="#ffa726", label="Daily t=0")
    ax.bar(x + width / 2, [periods[k]["median_bnh"] for k in labels], width, color="#888888", label="Buy & hold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#ccc")
    ax.set_ylabel("Median Sharpe", color="#ccc")
    ax.set_title("TASI daily Binary (t=0) by period", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white")
    ax.tick_params(colors="#ccc")
    ax.grid(axis="y", color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")
    plt.tight_layout()
    fig.savefig(OUT / "sharpe_by_period.png", dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def _write_crash(asset, reason):
    payload = _skip(asset, None, reason)
    payload["status"] = "error"
    (OUT / f"{asset['ticker']}.json").write_text(json.dumps(payload, indent=2))


def main():
    assets = load_assets()
    pending = [a for a in assets if not (OUT / f"{a['ticker']}.json").exists()]
    print(f"TASI: {len(assets)} names, {len(pending)} still to run", flush=True)
    if pending:
        failed = prefetch(pending)
        for asset, reason in failed:
            if not (OUT / f"{asset['ticker']}.json").exists():
                _write_crash(asset, reason)
        pending = [a for a in assets if not (OUT / f"{a['ticker']}.json").exists()]
        logdir = Path("/tmp/tasi_period_logs")
        logdir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        })
        running = {}
        attempts = {}
        total = len(pending)
        finished = 0
        while pending or running:
            while pending and len(running) < WORKERS:
                asset = pending.pop(0)
                logf = open(logdir / f"{asset['ticker']}.log", "a")
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()), "one", asset["ticker"]],
                    stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
                )
                running[proc.pid] = (proc, asset, logf, time.time())
                print(f"START {asset['ticker']}", flush=True)
            time.sleep(2)
            done = []
            for pid, (proc, asset, logf, started) in running.items():
                rc = proc.poll()
                if rc is None:
                    continue
                logf.close()
                done.append(pid)
                finished += 1
                dest = OUT / f"{asset['ticker']}.json"
                if dest.exists():
                    status = json.loads(dest.read_text()).get("status", "written")
                    print(f"FINISHED {finished}/{total} {asset['ticker']}: {status} "
                          f"({(time.time()-started)/60:.1f} min)", flush=True)
                    continue
                attempts[asset["ticker"]] = attempts.get(asset["ticker"], 0) + 1
                if attempts[asset["ticker"]] < 2:
                    print(f"RETRY {asset['ticker']} exit={rc}", flush=True)
                    pending.append(asset)
                    finished -= 1
                else:
                    _write_crash(asset, f"worker exited {rc} without a result file")
                    print(f"FINISHED {finished}/{total} {asset['ticker']}: crash {rc}", flush=True)
            for pid in done:
                del running[pid]
    summarize()
    from generate_tasi_report import main as write_report
    write_report()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "summarize":
        summarize()
    elif cmd == "one":
        wanted = sys.argv[2]
        asset = next(a for a in load_assets() if a["ticker"] == wanted)
        code, status = run_one(asset)
        print(f"DONE {code}: {status}", flush=True)
    else:
        main()
