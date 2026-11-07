"""Supply, demand, and mood. One integer fit on the training window.

The features are the ones already in technical_indicators: the 200-session
average, RSI(14), Bollinger %b, DI+ / DI-, and volume against its own
average. Each one is a reason to act, not a fitted weight.

Demand regime: the close is still above the 200-session average. The
business year of buyers is intact.

The dip, which is the crowd and the tape disagreeing with that regime:
  - RSI at 40 or below: fear
  - Bollinger %b at 0.35 or below: price pressed into the lower band
  - DI- above DI+: supply has the directional upper hand
  - volume at least 1.2 times its 20-session average on a down close: effort
  - the last five closes are down: the pullback is real

Enter, from cash, when the regime is still demand and at least K of those
five signs are present. Exit when the regime breaks, or when RSI reaches 70
(greed). The order is filled at the next session's open.

K is 2, 3, or 4. It is chosen on the training portfolio only, by the
Sharpe gap against buy and hold. A tie goes to the stricter K. Validation
(2017-2019) and the test (2020 onward) are not read by that choice.

Windows, fixed before the run:
  train      through 2016-12-31
  validate   2017-01-01 through 2019-12-31
  test       2020-01-01 onward
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from demand_line import (
    book,
    load_symbol,
    metrics,
    neutralize_jumps,
    roster,
    simulate,
)
from technical_indicators import adx_di, bollinger_bands, rsi, sma

OUT_DIR_NAME = "supply_demand"
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-01")
VAL_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")
K_GRID = (2, 3, 4)
FEAR_RSI = 40.0
GREED_RSI = 70.0
BAND = 0.35
EFFORT = 1.2


def feature_frame(clean):
    """Causal columns. Row t uses prices and volume through the close of t."""
    close = clean["Close"]
    line = sma(close, 200)
    mood = rsi(close, 14)
    _, _, _, _, pct_b = bollinger_bands(close)
    _, di_plus, di_minus = adx_di(clean["High"], clean["Low"], close)
    pct_b = pct_b.replace([np.inf, -np.inf], np.nan)
    di_plus = di_plus.replace([np.inf, -np.inf], np.nan)
    di_minus = di_minus.replace([np.inf, -np.inf], np.nan)
    vol_ma = sma(clean["Volume"], 20)
    ret = close.pct_change()
    week = ret.rolling(5, min_periods=5).mean()
    ready = line.notna() & mood.notna() & pct_b.notna() & di_plus.notna() & vol_ma.notna()
    return pd.DataFrame({
        "demand": close > line,
        "fear": mood <= FEAR_RSI,
        "washed": pct_b <= BAND,
        "supply": di_minus > di_plus,
        "effort": (clean["Volume"] > EFFORT * vol_ma) & (ret < 0),
        "pullback": week < 0,
        "greed": mood >= GREED_RSI,
        "ready": ready,
    }, index=clean.index)


def position_signal(feat, k):
    """Desired position at the close. The fill is the next open, applied later."""
    score = feat[["fear", "washed", "supply", "effort", "pullback"]].fillna(False).sum(axis=1)
    holding = 0
    out = np.full(len(feat), np.nan)
    demand = feat["demand"].to_numpy(dtype=bool)
    greed = feat["greed"].fillna(False).to_numpy(dtype=bool)
    ready = feat["ready"].to_numpy(dtype=bool)
    score_v = score.to_numpy()
    for i in range(len(feat)):
        if not ready[i]:
            holding = 0
            continue
        elif holding == 0:
            if demand[i] and score_v[i] >= k:
                holding = 1
        elif (not demand[i]) or greed[i]:
            holding = 0
        out[i] = holding
    return pd.Series(out, index=feat.index)


def pick_k(gaps):
    """Highest train Sharpe gap. A tie keeps the stricter rule."""
    chosen = None
    for k in K_GRID:
        gap = gaps[k]
        if gap is None:
            continue
        if chosen is None or gap > chosen[0] or (gap == chosen[0] and k > chosen[1]):
            chosen = (gap, k)
    if chosen is None:
        raise RuntimeError("no training window could be scored")
    return chosen[1]


def _window(frame, start, end):
    out = frame
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def window_report(portfolio, start, end):
    part = _window(portfolio, start, end)
    if part.empty:
        return None
    strategy = metrics(part["strat"], part["in_market"], part["turnover"])
    buy_hold = metrics(part["bnh"])
    if strategy is None or buy_hold is None:
        return None
    worst = part["strat"].groupby(pd.Grouper(freq="ME")).apply(lambda s: (1 + s).prod() - 1)
    return {
        "strategy": strategy,
        "bnh": buy_hold,
        "gap_sharpe": round(strategy["sharpe"] - buy_hold["sharpe"], 3),
        "worst_month": None if worst.empty else round(float(worst.min()) * 100, 2),
        "median_names": int(part["n"].median()) if "n" in part else None,
    }


def name_gap(sleeves, start, end):
    gaps = []
    for sim in sleeves.values():
        part = _window(sim, start, end)
        strategy = metrics(part["strat"], part["position"], part["turnover"])
        buy_hold = metrics(part["bnh"])
        if strategy is None or buy_hold is None:
            continue
        gaps.append(strategy["sharpe"] - buy_hold["sharpe"])
    if not gaps:
        return None
    series = pd.Series(gaps)
    return {
        "n": int(len(series)),
        "beat": int((series > 0).sum()),
        "median_gap": round(float(series.median()), 3),
    }


def prepare(catalog):
    ready = {}
    for symbol, meta in catalog.items():
        raw = load_symbol(symbol)
        if raw is None or len(raw) < 260:
            continue
        clean, n_jumps = neutralize_jumps(raw)
        ready[symbol] = {
            **meta,
            "frame": clean[["Open", "Close"]],
            "feat": feature_frame(clean),
            "n_jumps": n_jumps,
        }
    return ready


def sleeves(prepared, kind, k=None):
    out = {}
    for symbol, payload in prepared.items():
        if kind == "dip":
            signal = position_signal(payload["feat"], k)
        else:
            signal = payload["feat"]["demand"].astype(float).where(payload["feat"]["ready"])
        sim = simulate(payload["frame"], signal)
        if not sim.empty:
            out[symbol] = sim
    return out


def evaluate_book(prepared, label):
    print(f"\n=== {label} ===", flush=True)
    by_k = {}
    train_gaps = {}
    for k in K_GRID:
        portfolio = book(sleeves(prepared, "dip", k))
        train = window_report(portfolio, None, TRAIN_END)
        gap = None if train is None else train["gap_sharpe"]
        train_gaps[k] = gap
        by_k[k] = portfolio
        print(f"  train K={k} gap {gap}", flush=True)
    chosen = pick_k(train_gaps)
    print(f"  chosen K={chosen} on train only", flush=True)
    chosen_book = by_k[chosen]
    baseline = book(sleeves(prepared, "baseline"))
    chosen_sleeves = sleeves(prepared, "dip", chosen)
    windows = {
        "train": (None, TRAIN_END),
        "validate": (VAL_START, VAL_END),
        "test": (TEST_START, None),
    }
    report = {"chosen_k": chosen, "train_gaps": train_gaps, "dip": {}, "baseline": {}, "test_names": {}}
    for name, (start, end) in windows.items():
        report["dip"][name] = window_report(chosen_book, start, end)
        report["baseline"][name] = window_report(baseline, start, end)
        report["test_names"][name] = name_gap(chosen_sleeves, start, end)
        dip = report["dip"][name]
        base = report["baseline"][name]
        if dip:
            print(
                f"  {name:8} dip Sharpe {dip['strategy']['sharpe']} vs BnH {dip['bnh']['sharpe']} "
                f"CAGR {dip['strategy']['cagr']} vs {dip['bnh']['cagr']} "
                f"in {dip['strategy']['in_market_pct']}",
                flush=True,
            )
        if base:
            print(
                f"  {name:8} sma Sharpe {base['strategy']['sharpe']} vs BnH {base['bnh']['sharpe']}",
                flush=True,
            )
    return report


def self_check():
    idx = pd.bdate_range("2020-01-06", periods=6)
    feat = pd.DataFrame({
        "demand": [True, True, True, True, False, True],
        "fear": [False, True, False, False, False, True],
        "washed": [False, True, False, False, False, True],
        "supply": [False, False, False, False, False, False],
        "effort": [False, False, False, False, False, False],
        "pullback": [False, False, False, False, False, False],
        "greed": [False, False, True, False, False, False],
        "ready": [True] * 6,
    }, index=idx)
    signal = position_signal(feat, k=2)
    # Two fear signs on the second close open a long. Greed on the next close
    # closes it. That buy is filled one open later.
    if list(signal.astype(int)) != [0, 1, 0, 0, 0, 1]:
        raise AssertionError(f"signal {list(signal.astype(int))}")
    if pick_k({2: 0.10, 3: 0.10, 4: 0.05}) != 3:
        raise AssertionError("tie did not keep the stricter K")
    if pick_k({2: 0.20, 3: -0.10, 4: 0.05}) != 2:
        raise AssertionError("train gap did not win")
    frame = pd.DataFrame({
        "Open": [10.0, 10.0, 20.0, 21.0, 22.0, 23.0],
        "Close": [10.0, 10.0, 20.0, 21.0, 22.0, 23.0],
    }, index=idx)
    out = simulate(frame, signal, cost_bps=2.0)
    # The buy signal is the second close (price 10). The fill is the third
    # open (price 20). The strategy must not earn that 10-to-20 gap.
    held = out.loc[out["position"] > 0, "strat"]
    expected = (21.0 / 20.0 - 1.0) - 0.0002
    if abs(float(held.iloc[0]) - expected) > 1e-9:
        raise AssertionError(f"fill {float(held.iloc[0])} != {expected}")
    print("self-check passed", flush=True)


def main():
    self_check()
    tasi, global_book = roster()
    from pathlib import Path
    out = Path(__file__).resolve().parent / OUT_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "windows": {
            "train": f"first bar through {TRAIN_END.date()}",
            "validate": f"{VAL_START.date()} through {VAL_END.date()}",
            "test": f"{TEST_START.date()} onward",
        },
        "choice": "K on the training portfolio Sharpe gap only; tie prefers stricter K",
        "fixed": {
            "fear_rsi": FEAR_RSI,
            "greed_rsi": GREED_RSI,
            "bollinger_pctb": BAND,
            "volume_multiple": EFFORT,
            "k_grid": list(K_GRID),
        },
    }
    report["tasi"] = evaluate_book(prepare(tasi), "TASI")
    report["global"] = evaluate_book(prepare(global_book), "global")
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
