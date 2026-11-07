"""Monthly cross-section, fit only on the past.

People underreact to news for weeks and overreact for a few days. That is
the behavior this book trades. It does not predict tomorrow's close. By the
next open that move is gone, which is why the daily models failed.

Once a month, at the close, a shallow boosted tree ranks every name by the
open-to-open return it expects until the next monthly entry. The book buys
the top 30 percent in equal weight at the next open and holds that book
until the next monthly open. The bottom 70 percent stays in cash.

The label of a month is the return from that next open to the following
entry open. A month enters the training set only after that exit open has
already happened. The tree is refit each month on those finished months
alone. Nothing in 2017-2019 or in 2020 onward is used to choose a setting.
There is nothing to choose: depth, learning rate, leaf size, the 30 percent,
and the feature list are fixed below.

The comparison that the tree has to beat is the same book ranked by the
classic 12-month return, skipping the last month. That sort is the same
underreaction, with no model. Equal weight of every investable name is the
market.

The Saudi book is the 155 Saudi names. The global book is the other names
in the 102-asset universe. Saudi names keep their own Sunday-Thursday
clock instead of being mixed onto a US Friday.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from demand_line import load_symbol, metrics, neutralize_jumps, roster
from technical_indicators import adx_di, bollinger_bands, rsi, sma

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "behavior_ml"
TOP_FRACTION = 0.30
MIN_NAMES = 5
MIN_TRAIN_MONTHS = 24
COST_BPS = 2.0
JUMP = 0.15
SEED = 42
TRAIN_END = pd.Timestamp("2016-12-31")
VAL_START = pd.Timestamp("2017-01-01")
VAL_END = pd.Timestamp("2019-12-31")
TEST_START = pd.Timestamp("2020-01-01")

# Scale-free, known at the close. The longest lookback is 252 sessions.
FEATURES = [
    "mom_21",          # recent move; short-horizon overreaction
    "mom_63",          # one quarter of underreaction
    "mom_252_21",      # a year of underreaction, skipping the last month
    "dist_sma200",     # business-year demand
    "rsi_14",          # fear and greed
    "bb_pctb",         # where price sits in its recent range
    "di_spread",       # demand pressure minus supply pressure
    "attention",       # recent volume against the year's volume
    "supply_share",    # share of volume that printed on down closes
    "dd_252",          # pain versus the year's high
    "rv_21",           # current fear
    "rv_ratio",        # current fear against the year's fear
]


def model_template():
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )


def fit_predict(train_x, train_y, today_x):
    """Standardize with the training rows only, then predict."""
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    model = model_template()
    model.fit((train_x - mu) / sd, train_y)
    pred = model.predict((today_x - mu) / sd)
    return pred, mu, sd, model


def on_or_before(index, ts):
    i = index.searchsorted(pd.Timestamp(ts), side="right") - 1
    if i < 0:
        return None
    return index[i]


def first_after(index, ts):
    i = index.searchsorted(pd.Timestamp(ts), side="right")
    if i >= len(index):
        return None
    return index[i]


def name_path(index, opens, assignments):
    """Weights decided at a close become active on the next session.

    `assignments` maps the decision bar (last session at or before the
    signal) to the weight. The return from that bar's open to the next
    open is still earned by the previous weight.
    """
    weight = pd.Series(np.nan, index=index, dtype=float)
    for ts, value in assignments:
        weight.loc[ts] = value
    weight = weight.ffill()
    position = weight.shift(1)
    oo = opens.shift(-1) / opens - 1.0
    oo = oo.replace([np.inf, -np.inf], np.nan)
    turn = position.diff().abs()
    first = position.first_valid_index()
    if first is not None:
        turn.loc[first] = abs(float(position.loc[first]))
    return position, oo, turn


def feature_frame(clean):
    close = clean["Close"]
    volume = clean["Volume"].astype(float)
    line = sma(close, 200)
    mood = rsi(close, 14)
    _, _, _, _, pct_b = bollinger_bands(close)
    _, di_plus, di_minus = adx_di(clean["High"], clean["Low"], close)
    pct_b = pct_b.replace([np.inf, -np.inf], np.nan)
    denom = (di_plus + di_minus).replace(0, np.nan)
    spread = (di_plus - di_minus) / denom
    vol_21 = volume.rolling(21, min_periods=21).mean()
    vol_252 = volume.rolling(252, min_periods=252).mean()
    down = (close.diff() < 0).astype(float)
    supply = (volume * down).rolling(21, min_periods=21).sum() / volume.rolling(21, min_periods=21).sum()
    high_252 = close.rolling(252, min_periods=252).max()
    ret = close.pct_change()
    rv_21 = ret.rolling(21, min_periods=21).std()
    rv_252 = ret.rolling(252, min_periods=252).std()
    frame = pd.DataFrame({
        "Open": clean["Open"],
        "mom_21": close / close.shift(21) - 1.0,
        "mom_63": close / close.shift(63) - 1.0,
        "mom_252_21": close.shift(21) / close.shift(252) - 1.0,
        "dist_sma200": close / line - 1.0,
        "rsi_14": mood,
        "bb_pctb": pct_b,
        "di_spread": spread,
        "attention": vol_21 / vol_252 - 1.0,
        "supply_share": supply,
        "dd_252": close / high_252 - 1.0,
        "rv_21": rv_21,
        "rv_ratio": rv_21 / rv_252.replace(0, np.nan),
    }, index=clean.index)
    return frame.replace([np.inf, -np.inf], np.nan)


def prepare(catalog):
    ready = {}
    clipped = 0
    for symbol in catalog:
        raw = load_symbol(symbol)
        if raw is None or len(raw) < 260:
            continue
        clean, _ = neutralize_jumps(raw)
        frame = feature_frame(clean)
        oo = frame["Open"].shift(-1) / frame["Open"] - 1.0
        bad = oo.abs() > JUMP
        clipped += int(bad.fillna(False).sum())
        frame = frame.loc[frame["mom_252_21"].notna()].copy()
        if len(frame) < 40:
            continue
        ready[symbol] = frame
    return ready, clipped


def month_ends(symbols):
    dates = pd.DatetimeIndex(sorted(set().union(*[set(frame.index) for frame in symbols.values()])))
    stamps = pd.Series(dates, index=dates)
    ends = stamps.groupby([dates.year, dates.month]).max()
    return pd.DatetimeIndex(ends.to_list()).sort_values()


def build_samples(symbols, ends):
    """One row per name per month whose holding period has a known exit open."""
    rows = []
    for i in range(len(ends) - 1):
        signal = ends[i]
        nxt = ends[i + 1]
        for symbol, frame in symbols.items():
            bar = on_or_before(frame.index, signal)
            entry = first_after(frame.index, signal)
            exit_ = first_after(frame.index, nxt)
            if bar is None or entry is None or exit_ is None:
                continue
            values = frame.loc[bar, FEATURES]
            if values.isna().any():
                continue
            label = float(frame.loc[exit_, "Open"] / frame.loc[entry, "Open"] - 1.0)
            if not np.isfinite(label):
                continue
            row = {name: float(values[name]) for name in FEATURES}
            row.update({
                "symbol": symbol,
                "signal": signal,
                "label_end": exit_,
                "label": label,
                "mom_252_21": float(values["mom_252_21"]),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def today_panel(symbols, signal):
    rows = []
    bars = {}
    for symbol, frame in symbols.items():
        bar = on_or_before(frame.index, signal)
        if bar is None:
            continue
        values = frame.loc[bar, FEATURES]
        if values.isna().any():
            continue
        row = {name: float(values[name]) for name in FEATURES}
        row["symbol"] = symbol
        rows.append(row)
        bars[symbol] = bar
    if not rows:
        return pd.DataFrame(), bars
    return pd.DataFrame(rows).set_index("symbol"), bars


def weights_from_rank(score, fraction=TOP_FRACTION):
    """Equal weight in the top fraction. `score` is indexed by symbol."""
    ordered = score.sort_values(ascending=False)
    k = max(1, int(round(len(ordered) * fraction)))
    chosen = ordered.index[:k]
    weight = pd.Series(0.0, index=ordered.index)
    weight.loc[chosen] = 1.0 / k
    return weight


def walk_forward(symbols):
    ends = month_ends(symbols)
    samples = build_samples(symbols, ends)
    decisions = {"ml": {}, "mom": {}, "ew": {}}
    bars_used = {}
    diagnostics = []
    test_model = None
    for signal in ends:
        today, bars = today_panel(symbols, signal)
        if len(today) < MIN_NAMES:
            continue
        train = samples[(samples["label_end"] <= signal) & (samples["signal"] < signal)]
        if train["signal"].nunique() < MIN_TRAIN_MONTHS:
            continue
        # The month being predicted is not a training row: its exit open is later.
        if (train["label_end"] > signal).any() or (train["signal"] >= signal).any():
            raise RuntimeError("training rows leaked past the signal")
        pred, _, _, model = fit_predict(
            train[FEATURES].to_numpy(dtype=float),
            train["label"].to_numpy(dtype=float),
            today[FEATURES].to_numpy(dtype=float),
        )
        today = today.copy()
        today["pred"] = pred
        decisions["ml"][signal] = weights_from_rank(today["pred"])
        decisions["mom"][signal] = weights_from_rank(today["mom_252_21"])
        decisions["ew"][signal] = pd.Series(1.0 / len(today), index=today.index)
        bars_used[signal] = bars

        def _corr(left, right):
            if float(left.std()) < 1e-12 or float(right.std()) < 1e-12:
                return 0.0
            return round(float(left.corr(right)), 3)

        diagnostics.append({
            "signal": str(pd.Timestamp(signal).date()),
            "n": int(len(today)),
            "n_train": int(len(train)),
            "n_train_months": int(train["signal"].nunique()),
            "pred_mom_corr": _corr(today["pred"], today["mom_252_21"]),
            "pred_reversal_corr": _corr(today["pred"], today["mom_21"]),
        })
        if test_model is None and pd.Timestamp(signal) >= TEST_START:
            test_model = (model, train, today)
        if len(diagnostics) % 12 == 0:
            print(f"    {signal.date()} names {len(today)} train-months {train['signal'].nunique()}", flush=True)
    return decisions, bars_used, diagnostics, test_model


def simulate_book(symbols, decisions):
    """Hold shares between monthly opens. Trade only when the book changes.

    A weight decided at the signal close is applied at the first session
    after that close. From that open until the next rebalance the shares
    are held, so weights drift with price. Cash earns nothing.
    """
    oo = {}
    for symbol, frame in symbols.items():
        series = frame["Open"].shift(-1) / frame["Open"] - 1.0
        series = series.replace([np.inf, -np.inf], np.nan)
        oo[symbol] = series.mask(series.abs() > JUMP, 0.0)
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(series.index) for series in oo.values()])))
    rebalance_at = {}
    for signal, weights in decisions.items():
        i = all_dates.searchsorted(pd.Timestamp(signal), side="right")
        if i >= len(all_dates):
            continue
        rebalance_at[all_dates[i]] = weights
    names = list(symbols)
    drifted = {symbol: 0.0 for symbol in names}
    started = False
    rows = []
    for dt in all_dates:
        turn_today = 0.0
        cost = 0.0
        if dt in rebalance_at:
            target = rebalance_at[dt]
            new = {
                symbol: (float(target.loc[symbol]) if symbol in target.index else 0.0)
                for symbol in names
            }
            turn_today = float(sum(abs(new[symbol] - drifted[symbol]) for symbol in names))
            cost = turn_today * (COST_BPS / 10000.0)
            drifted = new
            started = True
        if not started:
            continue
        gross = 0.0
        any_ret = False
        grown = {}
        for symbol in names:
            series = oo[symbol]
            if dt in series.index and pd.notna(series.loc[dt]):
                ret = float(series.loc[dt])
                any_ret = True
            else:
                ret = 0.0
            gross += drifted[symbol] * ret
            grown[symbol] = drifted[symbol] * (1.0 + ret)
        if not any_ret:
            continue
        cash = 1.0 - sum(drifted.values())
        total = sum(grown.values()) + cash
        if total > 0:
            drifted = {symbol: grown[symbol] / total for symbol in names}
        held = sum(1 for symbol in names if drifted[symbol] > 1e-8)
        day = (1.0 - cost) * (1.0 + gross) - 1.0
        rows.append((dt, day, turn_today, held))
    frame = pd.DataFrame(rows, columns=["date", "ret", "turnover", "held"]).set_index("date")
    return frame


def window_slice(frame, start, end):
    part = frame
    if start is not None:
        part = part[part.index >= start]
    if end is not None:
        part = part[part.index <= end]
    return part


def window_stats(frame, start, end):
    part = window_slice(frame, start, end)
    if len(part) < 40:
        return None
    stat = metrics(part["ret"], positions=pd.Series(1.0, index=part.index), turnover=part["turnover"])
    if stat is None:
        return None
    worst = part["ret"].groupby(pd.Grouper(freq="ME")).apply(lambda s: (1 + s).prod() - 1)
    stat["worst_month"] = None if worst.empty else round(float(worst.min()) * 100, 2)
    stat["avg_names"] = round(float(part["held"].mean()), 1)
    return stat


def excess_boot(excess, block=21, reps=400):
    """95% interval for the annualized mean of strategy minus buy and hold."""
    values = excess.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(values) < block * 2:
        return None
    rng = np.random.default_rng(SEED)
    means = np.empty(reps)
    nobs = len(values)
    for i in range(reps):
        drawn = []
        while len(drawn) < nobs:
            start = int(rng.integers(0, nobs))
            drawn.extend(values[start:start + block].tolist())
        means[i] = np.mean(drawn[:nobs])
    annual = means * 252 * 100
    return {
        "ann_mean_excess_pct": round(float(values.mean() * 252 * 100), 2),
        "ci95_low": round(float(np.quantile(annual, 0.025)), 2),
        "ci95_high": round(float(np.quantile(annual, 0.975)), 2),
    }


def pack(book, bench):
    """Stats for one return stream against the equal-weight book."""
    windows = {
        "train": (None, TRAIN_END),
        "validate": (VAL_START, VAL_END),
        "test": (TEST_START, None),
        "full": (None, None),
    }
    out = {}
    for name, (start, end) in windows.items():
        stat = window_stats(book, start, end)
        base = window_stats(bench, start, end)
        if stat is None or base is None:
            out[name] = None
            continue
        left = window_slice(book, start, end)["ret"]
        right = window_slice(bench, start, end)["ret"]
        both = pd.concat([left.rename("s"), right.rename("b")], axis=1).dropna()
        excess = both["s"] - both["b"]
        vol = float(excess.std())
        ir = float(excess.mean() / vol * np.sqrt(252)) if vol > 1e-12 else 0.0
        row = {
            "strategy": stat,
            "bnh": base,
            "gap_sharpe": round(stat["sharpe"] - base["sharpe"], 3),
            "information_ratio": round(ir, 3),
        }
        if name == "test":
            row["excess_bootstrap"] = excess_boot(excess)
        out[name] = row
    return out


def self_check():
    idx = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-04", "2020-02-05"])
    opens = pd.Series([10.0, 20.0, 21.0, 22.0], index=idx)
    position, oo, _turn = name_path(idx, opens, [(idx[0], 1.0)])
    if not pd.isna(position.loc[idx[0]]):
        raise AssertionError("signal close already holds a position")
    if abs(float(position.loc[idx[1]]) - 1.0) > 1e-9:
        raise AssertionError("entry open did not receive the weight")
    # First earned return is 21/20, not the 10-to-20 gap into the entry.
    earned = float((position.loc[idx[1]] * oo.loc[idx[1]]))
    if abs(earned - (21.0 / 20.0 - 1.0)) > 1e-9:
        raise AssertionError(f"earned {earned}")
    if abs(float(oo.loc[idx[0]]) - 1.0) > 1e-9:
        raise AssertionError("test setup lost the entry gap")

    # January's holding ends the open after February's month-end, so it is
    # not yet known at the February month-end.
    ends = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    sessions = pd.to_datetime(["2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-03-31", "2020-04-01"])
    frame = pd.DataFrame({"Open": np.arange(len(sessions), dtype=float) + 10}, index=sessions)
    for col in FEATURES:
        frame[col] = 0.1
    symbols = {"AAA": frame}
    samples = build_samples(symbols, ends)
    feb = samples[samples["signal"] == ends[0]]
    if feb.empty or pd.Timestamp(feb.iloc[0]["label_end"]) != pd.Timestamp("2020-03-02"):
        raise AssertionError(f"label end {feb}")
    usable_in_february = samples[samples["label_end"] <= ends[1]]
    if not usable_in_february.empty:
        raise AssertionError("February training saw a label that ends in March")
    usable_in_march = samples[samples["label_end"] <= ends[2]]
    if usable_in_march.empty:
        raise AssertionError("March should already know the January holding")

    train_x = np.zeros((40, len(FEATURES)))
    today_x = np.full((1, len(FEATURES)), 1e6)
    _pred, mu, _sd, _model = fit_predict(train_x, np.zeros(40), today_x)
    if abs(float(mu[0])) > 1e-9:
        raise AssertionError("scaler used the row being predicted")
    ranks = weights_from_rank(pd.Series([0.1, 0.4, 0.2, 0.9, 0.0, 0.3, 0.15, 0.05, 0.25, 0.35]))
    if int((ranks > 0).sum()) != 3:
        raise AssertionError(f"top 30% of 10 names is {(ranks > 0).sum()}")

    book = simulate_book(
        {"AAA": pd.DataFrame({"Open": opens}, index=idx)},
        {idx[0]: pd.Series({"AAA": 1.0})},
    )
    if float(book["ret"].max()) > 0.1:
        raise AssertionError(f"book earned the entry gap: {book['ret'].tolist()}")
    first = (1.0 - 0.0002) * (21.0 / 20.0) - 1.0
    if abs(float(book["ret"].iloc[0]) - first) > 1e-9:
        raise AssertionError(f"first held return {float(book['ret'].iloc[0])} != {first}")
    print("self-check passed", flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_book(catalog, label):
    print(f"\n=== {label} ===", flush=True)
    symbols, clipped = prepare(catalog)
    print(f"  names {len(symbols)} leftover open jumps zeroed {clipped}", flush=True)
    decisions, _bars_used, diagnostics, _model = walk_forward(symbols)
    print(f"  rebalances {len(decisions['ml'])}", flush=True)
    books = {key: simulate_book(symbols, decisions[key]) for key in ("ml", "mom", "ew")}
    report = {
        "n_names": len(symbols),
        "open_jumps_zeroed": clipped,
        "rebalances": len(decisions["ml"]),
        "first_signal": diagnostics[0]["signal"] if diagnostics else None,
        "last_signal": diagnostics[-1]["signal"] if diagnostics else None,
        "mean_pred_corr_with_12m": None if not diagnostics else round(
            float(np.mean([d["pred_mom_corr"] for d in diagnostics])), 3
        ),
        "mean_pred_corr_with_1m": None if not diagnostics else round(
            float(np.mean([d["pred_reversal_corr"] for d in diagnostics])), 3
        ),
        "ml": pack(books["ml"], books["ew"]),
        "momentum_12_1": pack(books["mom"], books["ew"]),
    }
    for key, book in books.items():
        book.to_csv(OUT / f"{label}_{key}.csv")
    return report


def main():
    self_check()
    OUT.mkdir(parents=True, exist_ok=True)
    tasi, global_book = roster()
    report = {
        "seed": SEED,
        "git_commit": git_commit(),
        "model": "HistGradientBoostingRegressor depth 3, 200 trees, min leaf 40, l2 1, no early stopping",
        "label": "open at the next session after the signal, to the open of the next monthly entry",
        "purge": "a month is trainable only once its exit open date is on or before the signal being predicted",
        "top_fraction": TOP_FRACTION,
        "min_train_months": MIN_TRAIN_MONTHS,
        "cost_bps": COST_BPS,
        "features": FEATURES,
        "windows": {
            "train": "walk-forward months through 2016-12-31, each predicted without its own future",
            "validate": "2017-01-01 through 2019-12-31, not used to choose anything",
            "test": "2020-01-01 onward, not used to choose anything",
        },
    }
    report["tasi"] = run_book(tasi, "tasi")
    global_listed = {symbol: meta for symbol, meta in global_book.items() if not str(symbol).endswith(".SR")}
    report["global"] = run_book(global_listed, "global")
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
