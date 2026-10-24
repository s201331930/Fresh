"""
Run the SPY daily / weekly / mixed study on all 102 assets.

Each asset uses the same ensemble, the same four strategy variants, and the
same no-look-ahead rules. Assets with history back to 2002 keep the SPY
calendar: predictions from 2010, threshold fit through 2012, score from 2013.
Shorter assets shift that calendar forward so the training, calibration, and
evaluation blocks never overlap and never use future bars.

Results are written one file per asset under daily_universe/ so a stopped run
can resume.
"""

import json
import os
import subprocess
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import daily_vs_weekly as dv
from expanded_study import ALL_ASSETS

OUT = Path(__file__).parent / "daily_universe"
OUT.mkdir(exist_ok=True)
STUDY_ASSETS = Path(__file__).parent / "study" / "expanded_study_summary.json"

STRATS = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
WORKERS = 3


def load_assets():
    done = set(json.loads(STUDY_ASSETS.read_text())["assets"])
    assets = [a for a in ALL_ASSETS if a["name"] in done]
    missing = done - {a["name"] for a in assets}
    if missing:
        raise RuntimeError(f"Asset metadata missing for {sorted(missing)}")
    return assets


def _supports(index, oos, calib_end, eval_start, min_train, min_calib, min_eval):
    oos = pd.Timestamp(oos)
    calib_end = pd.Timestamp(calib_end)
    eval_start = pd.Timestamp(eval_start)
    if len(index) < min_train + min_eval:
        return False
    start = int(index.searchsorted(oos))
    if start < min_train or start >= len(index) - min_eval:
        return False
    calib_n = int(((index >= oos) & (index < calib_end)).sum())
    eval_n = int((index >= eval_start).sum())
    return calib_n >= min_calib and eval_n >= min_eval


def choose_schedule(daily_index, weekly_index):
    """Return (oos_start, calib_end, eval_start, label) or None."""
    fixed = (pd.Timestamp("2010-01-01"), pd.Timestamp("2013-01-01"), pd.Timestamp("2013-01-01"))
    if (_supports(daily_index, *fixed, 400, 100, 252)
            and _supports(weekly_index, *fixed, 100, 20, 52)):
        return (*fixed, "fixed_2013")

    d_pos = min(504, len(daily_index) - 1)
    w_pos = min(104, len(weekly_index) - 1)
    oos = max(pd.Timestamp(daily_index[d_pos]), pd.Timestamp(weekly_index[w_pos]))
    for years in (3, 2):
        calib_end = pd.Timestamp(oos) + pd.DateOffset(years=years)
        eval_start = pd.Timestamp(calib_end)
        if (_supports(daily_index, oos, calib_end, eval_start, 400, 80, 252)
                and _supports(weekly_index, oos, calib_end, eval_start, 100, 20, 52)):
            return pd.Timestamp(oos), calib_end, eval_start, f"shifted_{years}y"
    return None


def run_one(asset):
    name = asset["name"]
    dest = OUT / f"{name}.json"
    if dest.exists():
        return name, "cached"
    t0 = time.time()
    dv.N_JOBS = 1
    try:
        rule = "W-THU" if asset["market"] == "TASI" else "W-FRI"
        print(f"[{name}] fetch {asset['ticker']}", flush=True)
        daily = dv.fetch_ohlcv(asset["ticker"])
        macro = dv.fetch_macro_daily()
        weekly, daily_f, mixed = dv.build_study_frames(daily, macro, rule)
        schedule = choose_schedule(daily_f[0].index, weekly[0].index)
        if schedule is None:
            payload = {
                "name": name, "ticker": asset["ticker"], "market": asset["market"],
                "label": asset["label"], "asset_class": asset["class"],
                "status": "skipped", "reason": "not enough history for train, calibration, and evaluation",
            }
        else:
            oos, calib_end, eval_start, kind = schedule
            print(f"[{name}] {kind} oos={pd.Timestamp(oos).date()} eval={pd.Timestamp(eval_start).date()}",
                  flush=True)
            specs = [
                ("weekly", weekly, 26, 40, 52, 26),
                ("daily", daily_f, 126, 50, 252, 126),
                ("mixed", mixed, 126, 60, 252, 126),
            ]
            modes = {}
            for mode, frame, step, top_n, bars_per_year, lookback in specs:
                X, y, cols = frame
                modes[mode] = dv.run_mode(
                    mode, X, y, cols, step, top_n, bars_per_year, lookback,
                    oos_start=oos, calib_end=calib_end, eval_start=eval_start,
                    persist=False, quiet=True,
                )
            payload = {
                "name": name, "ticker": asset["ticker"], "market": asset["market"],
                "label": asset["label"], "asset_class": asset["class"],
                "status": "ok", "schedule": kind,
                "oos_start": str(pd.Timestamp(oos).date()),
                "calib_end": str(pd.Timestamp(calib_end).date()),
                "eval_start": str(pd.Timestamp(eval_start).date()),
                "modes": modes,
                "seconds": round(time.time() - t0, 1),
            }
            best_d = max(STRATS, key=lambda s: modes["daily"]["strategies"][s]["sharpe"])
            best_w = max(STRATS, key=lambda s: modes["weekly"]["strategies"][s]["sharpe"])
            print(f"[{name}] daily {best_d} {modes['daily']['strategies'][best_d]['sharpe']:.3f}  "
                  f"weekly {best_w} {modes['weekly']['strategies'][best_w]['sharpe']:.3f}  "
                  f"{payload['seconds'] / 60:.1f} min", flush=True)
        dest.write_text(json.dumps(payload, indent=2, default=str))
        return name, payload["status"]
    except Exception as exc:
        traceback.print_exc()
        payload = {
            "name": name, "ticker": asset.get("ticker"), "market": asset.get("market"),
            "label": asset.get("label"), "asset_class": asset.get("class"),
            "status": "error", "reason": f"{type(exc).__name__}: {exc}",
        }
        dest.write_text(json.dumps(payload, indent=2))
        return name, payload["reason"]


def _best(modes, mode):
    block = modes[mode]["strategies"]
    name = max(STRATS, key=lambda s: block[s]["sharpe"])
    return name, block[name]


def _row(asset):
    modes = asset["modes"]
    d_name, d_best = _best(modes, "daily")
    w_name, w_best = _best(modes, "weekly")
    m_name, m_best = _best(modes, "mixed")
    d0 = modes["daily"]["strategies"]["Binary (t=0)"]
    w0 = modes["weekly"]["strategies"]["Binary (t=0)"]
    m0 = modes["mixed"]["strategies"]["Binary (t=0)"]
    dopt = modes["daily"]["strategies"]["Binary (t=opt)"]
    return {
        "name": asset["name"],
        "asset_class": asset["asset_class"],
        "market": asset["market"],
        "schedule": asset["schedule"],
        "eval_start": asset["eval_start"],
        "daily_best": d_name,
        "weekly_best": w_name,
        "mixed_best": m_name,
        "daily_sharpe": d_best["sharpe"],
        "weekly_sharpe": w_best["sharpe"],
        "mixed_sharpe": m_best["sharpe"],
        "daily_cagr": d_best["cagr"],
        "weekly_cagr": w_best["cagr"],
        "mixed_cagr": m_best["cagr"],
        "daily_dd": d_best["max_drawdown"],
        "weekly_dd": w_best["max_drawdown"],
        "mixed_dd": m_best["max_drawdown"],
        "daily_bnh_sharpe": d_best["bnh_sharpe"],
        "weekly_bnh_sharpe": w_best["bnh_sharpe"],
        "daily_bnh_cagr": d_best["bnh_cagr"],
        "daily_t0_sharpe": d0["sharpe"],
        "weekly_t0_sharpe": w0["sharpe"],
        "mixed_t0_sharpe": m0["sharpe"],
        "daily_topt_sharpe": dopt["sharpe"],
        "daily_topt_in_market": dopt["in_market_pct"],
        "weekly_feature_share": modes["mixed"]["weekly_feature_share"],
        "daily_dir": modes["daily"]["model"]["dir_accuracy"],
        "weekly_dir": modes["weekly"]["model"]["dir_accuracy"],
    }


def summarize():
    files = sorted(OUT.glob("*.json"))
    records = [json.loads(p.read_text()) for p in files if p.name != "universe_summary.json"]
    ok = [r for r in records if r.get("status") == "ok"]
    skipped = [r for r in records if r.get("status") == "skipped"]
    errors = [r for r in records if r.get("status") == "error"]
    rows = [_row(r) for r in ok]
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No completed assets to summarize")
        return

    frame["daily_beats_weekly"] = frame["daily_sharpe"] > frame["weekly_sharpe"]
    frame["daily_beats_bnh"] = frame["daily_sharpe"] > frame["daily_bnh_sharpe"]
    frame["mixed_beats_daily"] = frame["mixed_sharpe"] > frame["daily_sharpe"]
    frame["daily_t0_beats_weekly_t0"] = frame["daily_t0_sharpe"] > frame["weekly_t0_sharpe"]
    frame["daily_t0_beats_bnh"] = frame["daily_t0_sharpe"] > frame["daily_bnh_sharpe"]
    frame["daily_topt_beats_bnh"] = frame["daily_topt_sharpe"] > frame["daily_bnh_sharpe"]
    frame["mixed_t0_beats_daily_t0"] = frame["mixed_t0_sharpe"] > frame["daily_t0_sharpe"]

    n = len(frame)
    summary = {
        "n_ok": n,
        "n_skipped": len(skipped),
        "n_error": len(errors),
        "skipped": [r["name"] for r in skipped],
        "errors": [{k: r.get(k) for k in ("name", "reason")} for r in errors],
        "daily_best_beats_weekly_best": int(frame["daily_beats_weekly"].sum()),
        "daily_best_beats_bnh": int(frame["daily_beats_bnh"].sum()),
        "mixed_best_beats_daily_best": int(frame["mixed_beats_daily"].sum()),
        "daily_t0_beats_weekly_t0": int(frame["daily_t0_beats_weekly_t0"].sum()),
        "daily_t0_beats_bnh": int(frame["daily_t0_beats_bnh"].sum()),
        "daily_topt_beats_bnh": int(frame["daily_topt_beats_bnh"].sum()),
        "mixed_t0_beats_daily_t0": int(frame["mixed_t0_beats_daily_t0"].sum()),
        "median_daily_sharpe": round(float(frame["daily_sharpe"].median()), 3),
        "median_weekly_sharpe": round(float(frame["weekly_sharpe"].median()), 3),
        "median_mixed_sharpe": round(float(frame["mixed_sharpe"].median()), 3),
        "median_bnh_sharpe": round(float(frame["daily_bnh_sharpe"].median()), 3),
        "median_daily_minus_weekly": round(float((frame["daily_sharpe"] - frame["weekly_sharpe"]).median()), 3),
        "median_mixed_minus_daily": round(float((frame["mixed_sharpe"] - frame["daily_sharpe"]).median()), 3),
        "by_class": {},
        "assets": frame.to_dict(orient="records"),
    }
    for cls, sub in frame.groupby("asset_class"):
        summary["by_class"][cls] = {
            "n": int(len(sub)),
            "daily_beats_weekly": int(sub["daily_beats_weekly"].sum()),
            "daily_beats_bnh": int(sub["daily_beats_bnh"].sum()),
            "mixed_beats_daily": int(sub["mixed_beats_daily"].sum()),
            "median_daily_sharpe": round(float(sub["daily_sharpe"].median()), 3),
            "median_weekly_sharpe": round(float(sub["weekly_sharpe"].median()), 3),
            "median_mixed_sharpe": round(float(sub["mixed_sharpe"].median()), 3),
        }
    (OUT / "universe_summary.json").write_text(json.dumps(summary, indent=2))
    frame.to_csv(OUT / "universe_comparison.csv", index=False)
    _plot(frame)
    print(f"\nCompleted {n}  skipped {len(skipped)}  errors {len(errors)}")
    print(f"Daily best > weekly best: {summary['daily_best_beats_weekly_best']}/{n}")
    print(f"Daily best > buy&hold:    {summary['daily_best_beats_bnh']}/{n}")
    print(f"Mixed best > daily best:  {summary['mixed_best_beats_daily_best']}/{n}")
    print(f"Median Sharpe daily/weekly/mixed/bnh: "
          f"{summary['median_daily_sharpe']} / {summary['median_weekly_sharpe']} / "
          f"{summary['median_mixed_sharpe']} / {summary['median_bnh_sharpe']}")


def _plot(frame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax = axes[0]
    ax.set_facecolor("#1e1e1e")
    ax.scatter(frame["weekly_sharpe"], frame["daily_sharpe"], c="#26a69a", s=28, alpha=0.85)
    lims = [
        min(frame["weekly_sharpe"].min(), frame["daily_sharpe"].min()) - 0.1,
        max(frame["weekly_sharpe"].max(), frame["daily_sharpe"].max()) + 0.1,
    ]
    ax.plot(lims, lims, color="#888", ls="--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Best weekly Sharpe", color="#ccc")
    ax.set_ylabel("Best daily Sharpe", color="#ccc")
    ax.set_title("Daily vs weekly, best variant", color="white")
    ax.tick_params(colors="#ccc")
    ax.grid(color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax = axes[1]
    ax.set_facecolor("#1e1e1e")
    ax.scatter(frame["daily_sharpe"], frame["mixed_sharpe"], c="#ab47bc", s=28, alpha=0.85)
    lims = [
        min(frame["daily_sharpe"].min(), frame["mixed_sharpe"].min()) - 0.1,
        max(frame["daily_sharpe"].max(), frame["mixed_sharpe"].max()) + 0.1,
    ]
    ax.plot(lims, lims, color="#888", ls="--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Best daily Sharpe", color="#ccc")
    ax.set_ylabel("Best mixed Sharpe", color="#ccc")
    ax.set_title("Mixed vs daily, best variant", color="white")
    ax.tick_params(colors="#ccc")
    ax.grid(color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")
    fig.suptitle(f"102-asset check of the SPY daily result ({len(frame)} completed)", color="white")
    plt.tight_layout()
    fig.savefig(OUT / "daily_vs_weekly_scatter.png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    classes = sorted(frame["asset_class"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    x = np.arange(len(classes))
    width = 0.25
    for i, (col, color, label) in enumerate([
        ("weekly_sharpe", "#42a5f5", "Weekly"),
        ("daily_sharpe", "#ffa726", "Daily"),
        ("mixed_sharpe", "#26a69a", "Mixed"),
    ]):
        vals = [frame.loc[frame["asset_class"] == c, col].median() for c in classes]
        ax.bar(x + i * width, vals, width, color=color, label=label, alpha=0.9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=30, ha="right", color="#ccc", fontsize=8)
    ax.set_ylabel("Median best Sharpe", color="#ccc")
    ax.set_title("Median best Sharpe by asset class", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white")
    ax.tick_params(colors="#ccc")
    ax.grid(axis="y", color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")
    plt.tight_layout()
    fig.savefig(OUT / "sharpe_by_class.png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


def _prefetch(assets):
    dv.fetch_macro_daily()
    for asset in assets:
        cache = dv.CACHE_DIR / f"{asset['ticker'].replace('=', '_').replace('.', '_')}_ohlcv.csv"
        if cache.exists():
            continue
        print(f"[prefetch] {asset['name']} {asset['ticker']}", flush=True)
        dv.fetch_ohlcv(asset["ticker"])


def _write_crash(asset, reason):
    payload = {
        "name": asset["name"], "ticker": asset.get("ticker"), "market": asset.get("market"),
        "label": asset.get("label"), "asset_class": asset.get("class"),
        "status": "error", "reason": reason,
    }
    (OUT / f"{asset['name']}.json").write_text(json.dumps(payload, indent=2))


def main():
    assets = load_assets()
    pending = [a for a in assets if not (OUT / f"{a['name']}.json").exists()]
    print(f"Universe: {len(assets)} assets, {len(pending)} still to run", flush=True)
    if not pending:
        summarize()
        return
    _prefetch(pending)
    logdir = Path("/tmp/daily_universe_logs")
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
            log_path = logdir / f"{asset['name']}.log"
            logf = open(log_path, "a")
            proc = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "one", asset["name"]],
                stdout=logf, stderr=subprocess.STDOUT, cwd=str(Path(__file__).parent), env=env,
            )
            running[proc.pid] = (proc, asset, logf, time.time())
            print(f"START {asset['name']}", flush=True)
        time.sleep(2)
        done_pids = []
        for pid, (proc, asset, logf, t0) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            done_pids.append(pid)
            finished += 1
            dest = OUT / f"{asset['name']}.json"
            elapsed = time.time() - t0
            if dest.exists():
                status = json.loads(dest.read_text()).get("status", "written")
                print(f"FINISHED {finished}/{total} {asset['name']}: {status} ({elapsed/60:.1f} min)", flush=True)
                continue
            attempts[asset["name"]] = attempts.get(asset["name"], 0) + 1
            if attempts[asset["name"]] < 2:
                print(f"RETRY {asset['name']} exit={rc}", flush=True)
                pending.append(asset)
                finished -= 1
            else:
                _write_crash(asset, f"worker exited {rc} without a result file")
                print(f"FINISHED {finished}/{total} {asset['name']}: crash exit {rc}", flush=True)
        for pid in done_pids:
            del running[pid]
    summarize()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "summarize":
        summarize()
    elif cmd == "one":
        wanted = sys.argv[2]
        asset = next(a for a in load_assets() if a["name"] == wanted)
        name, status = run_one(asset)
        print(f"DONE {name}: {status}", flush=True)
    else:
        main()
