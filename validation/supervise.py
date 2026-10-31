"""Subprocess supervisor. Does not import the model libraries (fork crashes)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIODS = ROOT / "tasi_periods"
RUNS = ROOT / "validation" / "runs"
WORKERS = 3


def scored_assets():
    universe = json.loads((ROOT / "tasi_universe.json").read_text())
    by_ticker = {a["ticker"]: a for a in universe["assets"]}
    wanted = []
    for path in sorted(PERIODS.glob("*.json")):
        if path.name == "summary.json":
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        asset = by_ticker.get(rec["ticker"])
        if asset is not None:
            wanted.append(asset)
    return wanted


def tag(macro_lag, fill, neutralize=False):
    lag = {"0": "lag0", "1": "lag1", "none": "nomacro"}.get(str(macro_lag), str(macro_lag))
    if fill == "close":
        out = lag
    else:
        out = f"{lag}_{fill}"
    if neutralize:
        out = f"{out}_clean"
    return out


def run(macro_lag, fill, limit=None, neutralize=False):
    assets = scored_assets()
    if limit:
        assets = assets[: int(limit)]
    dest_dir = RUNS / tag(macro_lag, fill, neutralize)
    dest_dir.mkdir(parents=True, exist_ok=True)
    pending = [a for a in assets if not (dest_dir / f"{a['ticker']}.json").exists()]
    print(f"{tag(macro_lag, fill, neutralize)}: {len(assets)} names, {len(pending)} to run", flush=True)
    logdir = Path("/tmp/audit_logs") / tag(macro_lag, fill, neutralize)
    logdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    worker = str(ROOT / "validation" / "tasi_worker.py")
    running = {}
    attempts = {}
    total = len(pending)
    finished = 0
    while pending or running:
        while pending and len(running) < WORKERS:
            asset = pending.pop(0)
            logf = open(logdir / f"{asset['ticker']}.log", "a")
            cmd = [sys.executable, "-u", worker, asset["ticker"],
                   "--macro-lag", str(macro_lag), "--fill", fill]
            if neutralize:
                cmd.append("--neutralize-jumps")
            proc = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env,
            )
            running[proc.pid] = (proc, asset, logf, time.time())
            print(f"START {asset['ticker']}", flush=True)
        time.sleep(5)
        done = []
        for pid, (proc, asset, logf, started) in running.items():
            if proc.poll() is None:
                continue
            logf.close()
            done.append(pid)
            finished += 1
            dest = dest_dir / f"{asset['ticker']}.json"
            if dest.exists():
                status = json.loads(dest.read_text()).get("status")
                print(f"FINISHED {finished}/{total} {asset['ticker']} {status} "
                      f"({(time.time()-started)/60:.1f} min)", flush=True)
                continue
            attempts[asset["ticker"]] = attempts.get(asset["ticker"], 0) + 1
            if attempts[asset["ticker"]] < 2:
                print(f"RETRY {asset['ticker']}", flush=True)
                pending.append(asset)
                finished -= 1
            else:
                dest.write_text(json.dumps({
                    "ticker": asset["ticker"], "status": "error",
                    "reason": f"worker exited {proc.returncode}",
                }, indent=2))
                print(f"CRASH {asset['ticker']} rc={proc.returncode}", flush=True)
        for pid in done:
            del running[pid]
    print(f"COMPLETE {tag(macro_lag, fill, neutralize)}", flush=True)


if __name__ == "__main__":
    lag = sys.argv[1] if len(sys.argv) > 1 else "0"
    fill = sys.argv[2] if len(sys.argv) > 2 else "close"
    limit = sys.argv[3] if len(sys.argv) > 3 else None
    neutralize = "--neutralize-jumps" in sys.argv
    run(lag, fill, limit, neutralize)
