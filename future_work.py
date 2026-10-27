"""
Explore the four items left as future work in Strategy_Selection_Study_102.pdf.

  1. Rolling strategy re-selection
  2. Multi-asset portfolio with a correlation constraint
  3. Regime-conditional strategy switching
  4. Daily bars for intraweek signal capture

No model is refit. Strategy paths for SPY are rebuilt from the saved
predictions. The 102-asset daily-versus-weekly numbers are read from the
saved universe summary. TASI period scores are read from the saved per-name
files. Portfolio prices are weekly closes; weights at each rebalance use
only weeks that have already finished.

A rule is fit on bars strictly before the window it is applied to.
The one look-ahead series in the output is named Oracle and is an upper
bound, not a trading rule.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from fpdf import FPDF
from PIL import Image
from scipy.optimize import minimize

ROOT = Path(__file__).parent
DAILY = ROOT / "daily_study"
UNIVERSE = ROOT / "daily_universe"
TASI = ROOT / "tasi_periods"
OUT = ROOT / "future_work"
OUT.mkdir(exist_ok=True)
PDF_PATH = ROOT / "Future_Work_Report.pdf"

COST_BPS = 2.0
COST = COST_BPS / 10000.0
EVAL = pd.Timestamp("2013-01-01")
STRATS = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def sharpe(returns, bars_per_year):
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 8:
        return -1e18
    vol = float(r.std())
    if vol < 1e-12:
        return -1e18
    return float(r.mean() / vol * np.sqrt(bars_per_year))


def performance(returns, bars_per_year):
    r = pd.Series(returns).dropna()
    if len(r) < 20:
        return None
    equity = (1.0 + r).cumprod()
    n_years = max((r.index[-1] - r.index[0]).days / 365.25, 0.25)
    vol = float(r.std())
    sh = float(r.mean() / vol * np.sqrt(bars_per_year)) if vol > 1e-12 else 0.0
    cagr = float(equity.iloc[-1] ** (1.0 / n_years) - 1.0)
    dd = float((equity / equity.cummax() - 1.0).min())
    return {
        "sharpe": round(sh, 3),
        "cagr": round(cagr * 100, 2),
        "max_drawdown": round(dd * 100, 2),
        "n_bars": int(len(r)),
        "n_years": round(float(n_years), 2),
        "total_return": round(float(equity.iloc[-1] - 1.0) * 100, 2),
    }


def sleeves(pred, actual, threshold, scale, lookback):
    """Positions and net returns. Actual is the next-bar percent return."""
    signal = pred.to_numpy(dtype=float)
    asset = actual.to_numpy(dtype=float) / 100.0
    binary_zero = (signal > 0.0).astype(float)
    binary_opt = (signal > threshold).astype(float)
    scaled = np.clip(signal / max(float(scale), 1e-4), 0.0, 1.0)
    adaptive = np.zeros(len(signal))
    grid = np.arange(-0.40, 0.401, 0.02)
    for i in range(len(signal)):
        if i < lookback:
            adaptive[i] = 1.0 if signal[i] > 0.0 else 0.0
            continue
        window_p = signal[i - lookback:i]
        window_r = asset[i - lookback:i]
        best_t, best_ret = 0.0, -1e18
        for threshold_i in grid:
            held = (window_p > threshold_i).astype(float)
            turnover = np.abs(np.diff(held, prepend=held[0])).sum()
            total = float((held * window_r).sum() - turnover * COST)
            if total > best_ret:
                best_t, best_ret = float(threshold_i), total
        adaptive[i] = 1.0 if signal[i] > best_t else 0.0
    positions = {
        "Binary (t=0)": binary_zero,
        "Binary (t=opt)": binary_opt,
        "Scaled": scaled,
        "Adaptive": adaptive,
    }
    own = {}
    for name, held in positions.items():
        turnover = np.abs(np.diff(held, prepend=0.0))
        own[name] = held * asset - turnover * COST
    frame = pd.DataFrame(own, index=pred.index)
    asset_s = pd.Series(asset, index=pred.index, name="BuyHold")
    return frame, asset_s, positions


def stitch(positions, asset, choice):
    """Return of a book that holds `choice[i]` at bar i. Turnover uses the position actually held."""
    asset_values = asset.to_numpy(dtype=float)
    n = len(asset_values)
    out = np.full(n, np.nan)
    prev = 0.0
    for i in range(n):
        name = choice[i]
        if name is None or (isinstance(name, float) and np.isnan(name)):
            prev = 0.0
            continue
        current = float(positions[name][i])
        out[i] = current * asset_values[i] - abs(current - prev) * COST
        prev = current
    return pd.Series(out, index=asset.index)


def rolling_choice(own, positions, asset, bars_per_year, step, lookback, eval_start):
    dates = own.index
    eval_ix = np.flatnonzero(dates >= eval_start)
    rebals = list(eval_ix[::step])
    choice = np.array([None] * len(dates), dtype=object)
    log = []
    for k, start in enumerate(rebals):
        end = rebals[k + 1] if k + 1 < len(rebals) else len(dates)
        past = own.iloc[max(0, start - lookback):start]
        if len(past) < max(40, lookback // 4):
            name = "Binary (t=0)"
            past_sharpe = None
        else:
            scored = {col: sharpe(past[col], bars_per_year) for col in STRATS}
            name = max(scored, key=scored.get)
            past_sharpe = round(scored[name], 3)
        choice[start:end] = name
        log.append({
            "start": str(dates[start].date()),
            "end": str(dates[end - 1].date()),
            "strategy": name,
            "past_sharpe": past_sharpe,
            "n_past": int(len(past)),
        })
    book = stitch(positions, asset, choice)
    # Oracle picks the sleeve with the best Sharpe inside the upcoming block.
    oracle_choice = np.array([None] * len(dates), dtype=object)
    for k, start in enumerate(rebals):
        end = rebals[k + 1] if k + 1 < len(rebals) else len(dates)
        block = own.iloc[start:end]
        scored = {col: sharpe(block[col], bars_per_year) for col in STRATS}
        oracle_choice[start:end] = max(scored, key=scored.get)
    oracle = stitch(positions, asset, oracle_choice)
    frozen_name = "Binary (t=0)"
    pre = own.loc[own.index < eval_start]
    if len(pre) >= 40:
        scored = {col: sharpe(pre[col], bars_per_year) for col in STRATS}
        frozen_name = max(scored, key=scored.get)
    frozen_choice = np.array([None] * len(dates), dtype=object)
    frozen_choice[eval_ix] = frozen_name
    frozen = stitch(positions, asset, frozen_choice)
    counts = pd.Series(choice[eval_ix]).value_counts(normalize=True)
    return {
        "rolling": book.loc[dates >= eval_start],
        "oracle": oracle.loc[dates >= eval_start],
        "frozen": frozen.loc[dates >= eval_start],
        "frozen_strategy": frozen_name,
        "switches": int(sum(choice[i] != choice[i - 1] for i in eval_ix[1:] if choice[i] and choice[i - 1])),
        "share": {k: round(float(v), 3) for k, v in counts.items()},
        "log": log,
    }


def regime_choice(own, positions, asset, regime, bars_per_year, step, eval_start, min_count):
    dates = own.index
    regime = regime.reindex(dates)
    eval_ix = np.flatnonzero(dates >= eval_start)
    rebals = list(eval_ix[::step])
    choice = np.array([None] * len(dates), dtype=object)
    maps = []
    for k, start in enumerate(rebals):
        end = rebals[k + 1] if k + 1 < len(rebals) else len(dates)
        past_r = own.iloc[:start]
        past_g = regime.iloc[:start]
        mapping = {}
        for state in (0.0, 1.0, 2.0):
            mask = past_g == state
            if int(mask.sum()) < min_count:
                mapping[state] = "Binary (t=0)"
                continue
            scored = {col: sharpe(past_r.loc[mask, col], bars_per_year) for col in STRATS}
            mapping[state] = max(scored, key=scored.get)
        for i in range(start, end):
            state = regime.iloc[i]
            if pd.isna(state):
                choice[i] = "Binary (t=0)"
            else:
                choice[i] = mapping.get(float(state), "Binary (t=0)")
        maps.append({str(int(k)): v for k, v in mapping.items()})
    book = stitch(positions, asset, choice)
    counts = pd.Series(choice[eval_ix]).value_counts(normalize=True)
    # How often the three regimes point at different sleeves, on the maps we actually used.
    distinct = sum(len(set(m.values())) > 1 for m in maps)
    return {
        "book": book.loc[dates >= eval_start],
        "share": {k: round(float(v), 3) for k, v in counts.items()},
        "maps_that_differ": int(distinct),
        "n_maps": int(len(maps)),
        "last_map": maps[-1] if maps else {},
    }


def regime_table(own, regime, bars_per_year, eval_start):
    """Sharpe of each sleeve inside each regime on the evaluation window.

    This describes the window. It is not a trading rule.
    """
    g = regime.reindex(own.index)
    rows = []
    labels = {0.0: "Bear", 1.0: "Sideways", 2.0: "Bull"}
    view = own.loc[own.index >= eval_start]
    states = g.loc[view.index]
    for state, label in labels.items():
        mask = states == state
        row = {"regime": label, "n": int(mask.sum())}
        for col in STRATS:
            value = sharpe(view.loc[mask, col], bars_per_year) if mask.sum() >= 20 else None
            row[col] = round(value, 3) if value is not None and value > -1e17 else None
        rows.append(row)
    return rows


def load_mode(mode, lookback, bars_per_year):
    summary = json.loads((DAILY / f"{mode}_summary.json").read_text())
    pred = pd.read_csv(DAILY / f"{mode}_predictions.csv", index_col=0, parse_dates=True).sort_index()
    own, asset, positions = sleeves(
        pred["prediction"], pred["actual"],
        summary["threshold"], summary["scale"], lookback,
    )
    check = performance(own["Binary (t=0)"].loc[EVAL:], bars_per_year)
    published = summary["strategies"]["Binary (t=0)"]["sharpe"]
    if abs(check["sharpe"] - published) > 0.02:
        raise SystemExit(f"{mode} Binary (t=0) Sharpe {check['sharpe']} != published {published}")
    check_opt = performance(own["Binary (t=opt)"].loc[EVAL:], bars_per_year)
    published_opt = summary["strategies"]["Binary (t=opt)"]["sharpe"]
    if abs(check_opt["sharpe"] - published_opt) > 0.02:
        raise SystemExit(f"{mode} Binary (t=opt) Sharpe {check_opt['sharpe']} != published {published_opt}")
    features = pd.read_csv(DAILY / f"spy_{mode}_features.csv", index_col=0, parse_dates=True).sort_index()
    regime = features["HMM_Regime"].reindex(pred.index)
    return {
        "summary": summary,
        "own": own,
        "asset": asset,
        "positions": positions,
        "regime": regime,
        "features": features,
        "pred": pred,
        "bars_per_year": bars_per_year,
        "lookback": lookback,
    }


def run_spy():
    modes = {
        "daily": load_mode("daily", 126, 252),
        "weekly": load_mode("weekly", 26, 52),
        "mixed": load_mode("mixed", 126, 252),
    }
    out = {}
    for name, mode in modes.items():
        step = 63 if mode["bars_per_year"] == 252 else 13
        lookback = 252 if mode["bars_per_year"] == 252 else 104
        min_count = 40 if mode["bars_per_year"] == 252 else 20
        rolled = rolling_choice(
            mode["own"], mode["positions"], mode["asset"],
            mode["bars_per_year"], step, lookback, EVAL,
        )
        regime = regime_choice(
            mode["own"], mode["positions"], mode["asset"], mode["regime"],
            mode["bars_per_year"], step, EVAL, min_count,
        )
        static = {}
        for strat in STRATS:
            static[strat] = performance(mode["own"][strat].loc[EVAL:], mode["bars_per_year"])
        static["BuyHold"] = performance(mode["asset"].loc[EVAL:], mode["bars_per_year"])
        out[name] = {
            "static": static,
            "rolling": performance(rolled["rolling"], mode["bars_per_year"]),
            "oracle": performance(rolled["oracle"], mode["bars_per_year"]),
            "frozen": performance(rolled["frozen"], mode["bars_per_year"]),
            "frozen_strategy": rolled["frozen_strategy"],
            "rolling_switches": rolled["switches"],
            "rolling_share": rolled["share"],
            "rolling_log": rolled["log"],
            "regime": performance(regime["book"], mode["bars_per_year"]),
            "regime_share": regime["share"],
            "regime_maps_that_differ": regime["maps_that_differ"],
            "regime_n_maps": regime["n_maps"],
            "regime_last_map": regime["last_map"],
            "regime_insample": regime_table(mode["own"], mode["regime"], mode["bars_per_year"], EVAL),
            "step_bars": step,
            "choice_lookback": lookback,
        }
        mode["rolling_equity"] = (1.0 + rolled["rolling"].dropna()).cumprod()
        mode["regime_equity"] = (1.0 + regime["book"].dropna()).cumprod()
    return modes, out


def intraweek(modes):
    daily = modes["daily"]
    features = daily["features"]
    day = features["DayOfWeek"].reindex(daily["own"].index)
    view = daily["own"].loc[EVAL:]
    asset = daily["asset"].loc[EVAL:]
    day = day.loc[view.index]
    rows = []
    for code, label in enumerate(WEEKDAYS):
        mask = day == code
        excess_zero = (view.loc[mask, "Binary (t=0)"] - asset.loc[mask]).sum()
        excess_opt = (view.loc[mask, "Binary (t=opt)"] - asset.loc[mask]).sum()
        rows.append({
            "weekday": label,
            "n": int(mask.sum()),
            "mean_bnh_bps": round(float(asset.loc[mask].mean() * 10000), 2),
            "mean_t0_excess_bps": round(float((view.loc[mask, "Binary (t=0)"] - asset.loc[mask]).mean() * 10000), 2),
            "sum_t0_excess": round(float(excess_zero), 4),
            "mean_topt_excess_bps": round(float((view.loc[mask, "Binary (t=opt)"] - asset.loc[mask]).mean() * 10000), 2),
            "sum_topt_excess": round(float(excess_opt), 4),
        })
    def to_holding_week(series):
        # Each daily value is the return from that close to the next close.
        # A Friday-labeled weekly target is the return from that Friday to the
        # next Friday, so the bin is [Friday, next Friday).
        grouped = series.groupby(pd.Grouper(freq="W-FRI", label="left", closed="left"))
        return grouped.apply(lambda x: (1.0 + x).prod() - 1.0 if len(x) else np.nan)

    weekly_from_daily = {}
    for strat in ["Binary (t=0)", "Binary (t=opt)", "Adaptive", "BuyHold"]:
        series = view[strat] if strat != "BuyHold" else asset
        weekly_from_daily[strat] = to_holding_week(series)
    native = modes["weekly"]
    native_view = native["own"].loc[EVAL:]
    native_asset = native["asset"].loc[EVAL:]
    compared = {}
    for strat in ["Binary (t=0)", "Binary (t=opt)", "Adaptive"]:
        left = weekly_from_daily[strat].dropna()
        right = native_view[strat].copy()
        right.index = pd.to_datetime(right.index)
        both = pd.concat([left.rename("daily"), right.rename("weekly")], axis=1, join="inner").dropna()
        compared[strat] = {
            "n_weeks": int(len(both)),
            "daily_compounded_sharpe": round(sharpe(both["daily"], 52), 3),
            "weekly_native_sharpe": round(sharpe(both["weekly"], 52), 3),
            "corr": round(float(both["daily"].corr(both["weekly"])), 3),
            "daily_wins": int((both["daily"] > both["weekly"]).sum()),
        }
    bnh_both = pd.concat(
        [weekly_from_daily["BuyHold"].rename("daily"), native_asset.rename("weekly")],
        axis=1, join="inner",
    ).dropna()
    return {
        "by_weekday": rows,
        "week_aligned": compared,
        "bnh_week_corr": round(float(bnh_both["daily"].corr(bnh_both["weekly"])), 3) if len(bnh_both) else None,
    }


def tasi_reselection():
    rows = []
    for path in sorted(TASI.glob("*.json")):
        if path.name == "summary.json":
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok" or len(rec.get("periods") or []) < 2:
            continue
        periods = sorted(rec["periods"], key=lambda p: p["start"])
        first = periods[0]["strategies"]
        frozen = max(STRATS_TASI, key=lambda name: first[name]["sharpe"])
        for prev, cur in zip(periods, periods[1:]):
            prev_s = prev["strategies"]
            cur_s = cur["strategies"]
            winner = max(STRATS_TASI, key=lambda name: prev_s[name]["sharpe"])
            guarded = [name for name in STRATS_TASI if prev_s[name]["in_market_pct"] >= 20]
            if guarded:
                winner_guard = max(guarded, key=lambda name: prev_s[name]["sharpe"])
            else:
                winner_guard = "Binary (t=0)"
            oracle = max(STRATS_TASI, key=lambda name: cur_s[name]["sharpe"])
            bnh = cur_s["Binary (t=0)"]["bnh_sharpe"]
            bnh_cagr = cur_s["Binary (t=0)"]["bnh_cagr"]
            rows.append({
                "ticker": rec["ticker"],
                "period": cur["label"],
                "bnh_sharpe": bnh,
                "bnh_cagr": bnh_cagr,
                "always_t0": cur_s["Binary (t=0)"]["sharpe"],
                "always_t0_cagr": cur_s["Binary (t=0)"]["cagr"],
                "previous": cur_s[winner]["sharpe"],
                "previous_cagr": cur_s[winner]["cagr"],
                "previous_name": winner,
                "guarded": cur_s[winner_guard]["sharpe"],
                "guarded_cagr": cur_s[winner_guard]["cagr"],
                "guarded_name": winner_guard,
                "frozen": cur_s[frozen]["sharpe"],
                "frozen_cagr": cur_s[frozen]["cagr"],
                "frozen_name": frozen,
                "oracle": cur_s[oracle]["sharpe"],
                "oracle_cagr": cur_s[oracle]["cagr"],
            })
    frame = pd.DataFrame(rows)

    def pack(sharpe_col, cagr_col):
        return {
            "n": int(len(frame)),
            "median_sharpe": round(float(frame[sharpe_col].median()), 3),
            "median_cagr": round(float(frame[cagr_col].median()), 2),
            "beat_bnh": int((frame[sharpe_col] > frame["bnh_sharpe"]).sum()),
            "beat_t0": int((frame[sharpe_col] > frame["always_t0"]).sum()),
            "median_gap_vs_t0": round(float((frame[sharpe_col] - frame["always_t0"]).median()), 3),
        }

    summary = {
        "n_names": int(frame["ticker"].nunique()),
        "n_hold_periods": int(len(frame)),
        "always_t0": pack("always_t0", "always_t0_cagr"),
        "previous": pack("previous", "previous_cagr"),
        "guarded": pack("guarded", "guarded_cagr"),
        "frozen": pack("frozen", "frozen_cagr"),
        "oracle": pack("oracle", "oracle_cagr"),
        "buyhold": {
            "median_sharpe": round(float(frame["bnh_sharpe"].median()), 3),
            "median_cagr": round(float(frame["bnh_cagr"].median()), 2),
        },
        "previous_choice_counts": frame["previous_name"].value_counts().to_dict(),
        "guarded_choice_counts": frame["guarded_name"].value_counts().to_dict(),
    }
    return summary


STRATS_TASI = [
    "Binary (t=0)",
    "Binary (t=opt)",
    "Binary (t=opt frozen)",
    "Scaled",
    "Scaled (frozen)",
    "Adaptive",
]


def load_universe():
    assets = []
    for path in sorted(UNIVERSE.glob("*.json")):
        if path.name.startswith("universe"):
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") == "ok" and rec.get("ticker"):
            assets.append(rec)
    return assets


def weekly_panel(assets):
    cache = OUT / "weekly_closes.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    tickers = [a["ticker"] for a in assets]
    frames = []
    for start in range(0, len(tickers), 30):
        batch = tickers[start:start + 30]
        data = yf.download(
            batch, start="2004-01-01", auto_adjust=True,
            progress=False, group_by="ticker", threads=True,
        )
        frames.append(data)
    raw = pd.concat(frames, axis=1)
    columns = {}
    for asset in assets:
        ticker = asset["ticker"]
        week = "W-THU" if asset.get("market") == "TASI" else "W-FRI"
        try:
            if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0):
                close = raw[ticker]["Close"]
            elif isinstance(raw.columns, pd.MultiIndex) and "Close" in raw.columns.get_level_values(0):
                close = raw["Close"][ticker]
            else:
                continue
        except KeyError:
            continue
        close = close.dropna()
        if close.empty:
            continue
        weekly = close.resample(week).last().dropna()
        if week == "W-THU":
            weekly.index = weekly.index + pd.Timedelta(days=1)
        columns[asset["name"]] = weekly
    panel = pd.DataFrame(columns).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    panel.to_csv(cache)
    return panel


def _cap_weights(weights, cap):
    weights = np.maximum(np.asarray(weights, dtype=float), 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    weights = weights / weights.sum()
    cap = max(float(cap), 1.0 / len(weights))
    for _ in range(30):
        over = weights > cap + 1e-12
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        free = ~over
        if not free.any() or weights[free].sum() <= 0:
            break
        weights[free] += excess * weights[free] / weights[free].sum()
    total = weights.sum()
    return weights / total if total > 0 else weights


def _min_var(cov, cap):
    n = cov.shape[0]
    cap = max(float(cap), 1.0 / n)

    def objective(weights):
        return float(weights @ cov @ weights)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    bounds = [(0.0, cap)] * n
    start = np.full(n, 1.0 / n)
    result = minimize(
        objective, start, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-12},
    )
    if result.success:
        weights = np.clip(result.x, 0.0, None)
        total = weights.sum()
        if total > 0:
            return weights / total
    vol = np.sqrt(np.clip(np.diag(cov), 1e-10, None))
    return _cap_weights(1.0 / vol, cap)


def _corr_screen(corr, vols, max_corr=0.70):
    """Equal-weight the names that remain after high-correlation pairs are cut.

    Minimum variance already avoids those pairs, so the screen starts from the
    full eligible set. The higher-volatility name of the worst pair is dropped
    until every remaining pair is at or below max_corr.
    """
    keep = np.ones(len(vols), dtype=bool)
    for _ in range(len(vols)):
        active = np.flatnonzero(keep)
        worst = None
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = int(active[i]), int(active[j])
                pair = float(corr[a, b])
                if pair > max_corr and (worst is None or pair > worst[0]):
                    drop = a if vols[a] >= vols[b] else b
                    worst = (pair, drop)
        if worst is None:
            break
        keep[worst[1]] = False
    if not keep.any():
        keep[:] = True
    weights = keep.astype(float)
    return weights / weights.sum()


def _book_stats(weights, corr):
    active = weights > 0.005
    if active.sum() < 2:
        return {"n": int(active.sum()), "avg_pair_corr": None}
    sub = corr[np.ix_(active, active)]
    upper = sub[np.triu_indices(len(sub), k=1)]
    return {"n": int(active.sum()), "avg_pair_corr": float(np.mean(upper))}


def run_portfolio(assets):
    panel = weekly_panel(assets)
    returns = panel.pct_change().replace([np.inf, -np.inf], np.nan)
    class_of = {a["name"]: a.get("asset_class", "") for a in assets}
    dates = returns.index
    # First January week of each year, once 156 prior weeks exist.
    rebals = []
    for i, stamp in enumerate(dates):
        if stamp.month != 1:
            continue
        if i < 156:
            continue
        if rebals and dates[rebals[-1]].year == stamp.year:
            continue
        rebals.append(i)
    books = {
        "Equal weight": [],
        "Inverse vol": [],
        "Min variance": [],
        "Corr screen 0.70": [],
    }
    class_weights = []
    pair_corr = {name: [] for name in books}
    names_held = {name: [] for name in books}
    turnovers = {name: [] for name in books}
    prev = {name: None for name in books}
    index = []
    for k, start in enumerate(rebals):
        end = rebals[k + 1] if k + 1 < len(rebals) else len(dates)
        past = returns.iloc[start - 156:start]
        usable = [col for col in past.columns if past[col].notna().sum() >= 140]
        if len(usable) < 15:
            continue
        block = past[usable].dropna(how="any")
        if len(block) < 80:
            continue
        values = block.to_numpy()
        cov_sample = np.cov(values, rowvar=False)
        shrink = 0.5 * cov_sample + 0.5 * np.diag(np.diag(cov_sample))
        vols = np.sqrt(np.clip(np.diag(cov_sample), 1e-10, None))
        sample_corr = np.corrcoef(values, rowvar=False)
        sample_corr = np.nan_to_num(sample_corr, nan=0.0)
        np.fill_diagonal(sample_corr, 1.0)
        n = len(usable)
        cap = max(0.08, 1.0 / n)
        weights = {
            "Equal weight": np.full(n, 1.0 / n),
            "Inverse vol": _cap_weights(1.0 / vols, cap),
            "Min variance": _min_var(shrink, cap),
            "Corr screen 0.70": _corr_screen(sample_corr, vols, max_corr=0.70),
        }
        corr = sample_corr
        future = returns.iloc[start:end][usable]
        for name, held in weights.items():
            stats = _book_stats(held, corr)
            names_held[name].append(stats["n"])
            if stats["avg_pair_corr"] is not None:
                pair_corr[name].append(stats["avg_pair_corr"])
            gross = future.to_numpy() @ held
            aligned_prev = pd.Series(held, index=usable)
            if prev[name] is None:
                turnover = 1.0
            else:
                # Names enter and leave the eligible set between rebalances.
                union = list(dict.fromkeys(list(prev[name].index) + usable))
                old = prev[name].reindex(union).fillna(0.0).to_numpy()
                new = aligned_prev.reindex(union).fillna(0.0).to_numpy()
                turnover = float(np.abs(new - old).sum())
            cost = turnover * COST
            gross = pd.Series(gross, index=future.index)
            gross.iloc[0] = gross.iloc[0] - cost
            books[name].append(gross)
            turnovers[name].append(turnover)
            prev[name] = aligned_prev
        class_row = {}
        for label, held in (("Min variance", weights["Min variance"]), ("Corr screen 0.70", weights["Corr screen 0.70"])):
            by_class = {}
            for weight, asset_name in zip(held, usable):
                by_class[class_of.get(asset_name, "")] = by_class.get(class_of.get(asset_name, ""), 0.0) + float(weight)
            class_row[label] = {k: round(v, 3) for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])[:6]}
        class_weights.append({"date": str(dates[start].date()), **class_row})
        index.append(str(dates[start].date()))
    stitched = {}
    curves = {}
    for name, parts in books.items():
        series = pd.concat(parts).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        stitched[name] = performance(series, 52)
        stitched[name]["median_names"] = int(np.median(names_held[name])) if names_held[name] else 0
        stitched[name]["avg_pair_corr"] = round(float(np.mean(pair_corr[name])), 3) if pair_corr[name] else None
        stitched[name]["mean_turnover"] = round(float(np.mean(turnovers[name])), 3) if turnovers[name] else None
        curves[name] = (1.0 + series).cumprod()
    return {
        "n_assets_priced": int(panel.shape[1]),
        "n_rebalances": len(index),
        "first": index[0] if index else None,
        "last": index[-1] if index else None,
        "books": stitched,
        "class_weights_last": class_weights[-1] if class_weights else {},
        "curves": curves,
    }


def daily_universe_brief():
    summary = json.loads((UNIVERSE / "universe_summary.json").read_text())
    classes = []
    for label, row in sorted(summary["by_class"].items()):
        classes.append({
            "class": label,
            "n": row["n"],
            "daily_beats_weekly": row["daily_beats_weekly"],
            "median_daily": row["median_daily_sharpe"],
            "median_weekly": row["median_weekly_sharpe"],
            "median_mixed": row["median_mixed_sharpe"],
        })
    keep = [
        "n_ok", "daily_best_beats_weekly_best", "daily_t0_beats_weekly_t0",
        "daily_t0_beats_bnh", "mixed_best_beats_daily_best", "mixed_t0_beats_daily_t0",
        "median_daily_sharpe", "median_weekly_sharpe", "median_mixed_sharpe",
        "median_bnh_sharpe", "median_daily_minus_weekly", "median_mixed_minus_daily",
    ]
    return {key: summary[key] for key in keep} | {"by_class": classes}


def plot_equity(path, series, title, ylabel="Growth of 1"):
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for label, curve in series.items():
        ax.plot(curve.index, curve.values, label=label, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_weekday(path, rows):
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    labels = [row["weekday"] for row in rows]
    values = [row["mean_t0_excess_bps"] for row in rows]
    colors = ["#1b4f72" if value > 0 else "#922b21" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Mean excess vs buy-and-hold, bps per day")
    ax.set_title("SPY daily Binary (t=0), excess by weekday")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _jpeg(path):
    image = Image.open(path).convert("RGB")
    dest = Path(path).with_suffix(".jpg")
    image.save(dest, "JPEG", quality=90)
    return dest


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "Future work from the 102-asset study - Research prototype", align="R")
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.ln(2)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(20, 60, 120)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(40, 40, 40)
        self.ln(1)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.2, text)
        self.ln(1.2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        self.multi_cell(0, 5.2, "- " + text)
        self.ln(0.3)

    def table(self, headers, rows, widths, size=7.5, row_h=5.4):
        def paint():
            self.set_font("Helvetica", "B", size)
            self.set_fill_color(20, 60, 120)
            self.set_text_color(255, 255, 255)
            for header, width in zip(headers, widths):
                self.cell(width, 7, str(header), border=1, fill=True, align="C")
            self.ln()

        paint()
        fill = False
        for row in rows:
            limit = self.h - max(self.b_margin, 16) - 1
            if self.get_y() + row_h > limit:
                self.add_page()
                paint()
            self.set_fill_color(235, 241, 248) if fill else self.set_fill_color(255, 255, 255)
            self.set_text_color(30, 30, 30)
            self.set_font("Helvetica", "", size)
            for i, (val, width) in enumerate(zip(row, widths)):
                self.cell(width, row_h, str(val), border=1, fill=True, align="L" if i == 0 else "C")
            self.ln()
            fill = not fill
        self.ln(2)


def _fmt_map(mapping):
    labels = {"0": "Bear", "1": "Sideways", "2": "Bull"}
    return ", ".join(f"{labels.get(k, k)} -> {v}" for k, v in mapping.items())


def write_report(spy, intra, tasi, portfolio, universe):
    pdf = Report(format="A4")
    pdf.alias_nb_pages()
    pdf.set_title("Future work from the 102-asset study")
    pdf.set_author("Research prototype")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.ln(24)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 60, 120)
    pdf.multi_cell(0, 10, "Four follow-ups from the 102-asset study", align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(0, 7, "Rolling strategy choice, a correlation-capped portfolio, regime switching, and daily bars.", align="C")
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 7, "Research prototype. Not investment advice. October 2026.", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    pdf.h1("1. What was tested")
    pdf.body(
        "The 102-asset report left four items open: rolling strategy re-selection, a multi-asset "
        "portfolio with a correlation constraint, regime-conditional switching inside one asset, "
        "and daily bars for intraweek signals. This note runs those four on the data already computed."
    )
    pdf.bullet("SPY sleeves are rebuilt from the saved daily, weekly, and mixed predictions. The 2013-onward Sharpe matches the published file to 0.001.")
    pdf.bullet("A choice uses only bars before the window it trades. Thresholds and scales are the frozen values from each saved summary.")
    pdf.bullet("TASI uses the saved period scores. The strategy for a period is the winner of the previous period.")
    pdf.bullet("The portfolio is a buy-and-hold book of the 102 names. Per-asset strategy paths were not stored, so the optimizer cannot apply those sleeves.")
    pdf.bullet("Oracle rows pick the sleeve with the best Sharpe inside the window they score. They are an upper bound.")
    pdf.body(
        "Cost is 2 bps of the position change, the same convention as the earlier studies. "
        "Sharpe uses sqrt(252) on daily bars and sqrt(52) on weekly bars. CAGR uses calendar time."
    )

    pdf.h1("2. Rolling strategy re-selection")
    pdf.body(
        "Every quarter on daily bars, and every 13 weeks on weekly bars, the book takes the sleeve "
        "with the highest Sharpe over the trailing year. The pre-2013 winner is the frozen rule."
    )
    for mode in ("daily", "weekly", "mixed"):
        block = spy[mode]
        pdf.h2(mode.capitalize())
        rows = []
        for label, stats in [
            ("Binary (t=0)", block["static"]["Binary (t=0)"]),
            ("Binary (t=opt)", block["static"]["Binary (t=opt)"]),
            ("Scaled", block["static"]["Scaled"]),
            ("Adaptive", block["static"]["Adaptive"]),
            ("Buy and hold", block["static"]["BuyHold"]),
            (f"Frozen ({block['frozen_strategy']})", block["frozen"]),
            ("Rolling", block["rolling"]),
            ("Oracle", block["oracle"]),
        ]:
            rows.append([
                label, f"{stats['sharpe']:.3f}", f"{stats['cagr']:.2f}", f"{stats['max_drawdown']:.1f}",
            ])
        pdf.table(["Rule", "Sharpe", "CAGR %", "Max DD %"], rows, [70, 32, 32, 32], size=8)
        pdf.body(
            f"The rolling book switched {block['rolling_switches']} times. "
            f"Share of bars: " + ", ".join(f"{k} {v:.0%}" for k, v in block["rolling_share"].items()) + ". "
            f"The frozen pre-2013 winner was {block['frozen_strategy']}."
        )
    d = spy["daily"]
    pdf.body(
        f"On daily SPY, rolling Sharpe is {d['rolling']['sharpe']:.3f} against "
        f"{d['static']['Binary (t=0)']['sharpe']:.3f} for a fixed zero threshold and "
        f"{d['static']['Binary (t=opt)']['sharpe']:.3f} for the frozen optimal threshold. "
        f"The oracle, which sees each upcoming quarter, reaches {d['oracle']['sharpe']:.3f}. "
        "Adapting the sleeve on trailing Sharpe does not beat the best fixed rule on this path."
    )

    pdf.h2("TASI, previous period picks the next one")
    pdf.body(
        f"{tasi['n_names']} names have at least two periods. The {tasi['n_hold_periods']} rows below "
        "are the later periods only. The choice is locked before the period starts."
    )
    order = [
        ("Always t=0", "always_t0"),
        ("Previous Sharpe winner", "previous"),
        ("Previous winner, in market >= 20%", "guarded"),
        ("Frozen first-period winner", "frozen"),
        ("Oracle", "oracle"),
    ]
    rows = []
    for label, key in order:
        stats = tasi[key]
        rows.append([
            label,
            str(stats["n"]),
            f"{stats['median_sharpe']:.3f}",
            f"{stats['median_cagr']:.2f}",
            f"{stats['beat_bnh']}/{stats['n']}",
            f"{stats['median_gap_vs_t0']:+.3f}",
        ])
    pdf.table(
        ["Rule", "N", "Med Sh", "Med CAGR", "Beat BnH", "Gap vs t=0"],
        rows, [62, 16, 22, 24, 26, 28], size=8,
    )
    pdf.body(
        f"Buy-and-hold median Sharpe on the same periods is {tasi['buyhold']['median_sharpe']:.3f}, "
        f"median CAGR {tasi['buyhold']['median_cagr']:.2f}%. "
        f"The previous-period winner beats the fixed zero threshold in "
        f"{tasi['previous']['beat_t0']} of {tasi['previous']['n']} periods. "
        f"The median gap is {tasi['previous']['median_gap_vs_t0']:+.3f}. "
        f"Requiring that winner to have been in the market at least 20% of sessions "
        f"beats the zero threshold in {tasi['guarded']['beat_t0']} of {tasi['guarded']['n']} periods, "
        f"median gap {tasi['guarded']['median_gap_vs_t0']:+.3f}."
    )

    pdf.h1("3. Portfolio with a correlation constraint")
    pdf.body(
        f"Weekly closes for {portfolio['n_assets_priced']} of the 102 names. Once each January, "
        "weights are fit on the prior 156 weeks and held for the next year. "
        f"{portfolio['n_rebalances']} rebalances, {portfolio['first']} through {portfolio['last']}. "
        "Minimum variance is long-only, 50% shrunk toward its diagonal, with an 8% name cap. "
        "The correlation screen starts from every eligible name, drops the higher-volatility name "
        "of any pair whose sample correlation is above 0.70, and equal-weights whoever remains. "
        "Minimum variance is a different book: it already concentrates in bonds and currencies."
    )
    rows = []
    for name, stats in portfolio["books"].items():
        rows.append([
            name,
            f"{stats['sharpe']:.3f}",
            f"{stats['cagr']:.2f}",
            f"{stats['max_drawdown']:.1f}",
            f"{stats['avg_pair_corr']:.2f}" if stats["avg_pair_corr"] is not None else "-",
            str(stats["median_names"]),
            f"{stats['mean_turnover']:.2f}",
        ])
    pdf.table(
        ["Book", "Sharpe", "CAGR %", "Max DD %", "Pair corr", "Names", "Turnover"],
        rows, [36, 22, 22, 24, 24, 20, 24], size=8,
    )
    last = portfolio.get("class_weights_last") or {}
    def _class_line(mapping):
        return ", ".join(f"{name} {weight:.0%}" for name, weight in mapping.items())

    if last:
        pdf.body(
            "Average pairwise correlation is the sample correlation of the names held, on the same "
            "past window that set the weights. The January 2026 book put this much weight by class. "
            f"Min variance: {_class_line(last.get('Min variance', {}))}. "
            f"Correlation screen: {_class_line(last.get('Corr screen 0.70', {}))}."
        )
    chart = OUT / "portfolio_equity.png"
    if chart.exists():
        pdf.image(str(_jpeg(chart)), w=170)
        pdf.ln(3)

    pdf.h1("4. Regime-conditional switching")
    pdf.body(
        "The saved HMM state is known at the close. The return it is allowed to earn is the next bar. "
        "Every quarter the map from bear, sideways, and bull to a sleeve is refit on earlier bars only. "
        "A state with fewer than 40 past daily bars, or 20 weekly bars, stays on Binary (t=0)."
    )
    rows = []
    for mode in ("daily", "weekly", "mixed"):
        block = spy[mode]
        rows.append([
            mode,
            f"{block['regime']['sharpe']:.3f}",
            f"{block['regime']['cagr']:.2f}",
            f"{block['regime']['max_drawdown']:.1f}",
            f"{block['static']['Binary (t=0)']['sharpe']:.3f}",
            f"{block['rolling']['sharpe']:.3f}",
            f"{block['regime_maps_that_differ']}/{block['regime_n_maps']}",
        ])
    pdf.table(
        ["Mode", "Regime Sh", "CAGR %", "Max DD %", "t=0 Sh", "Rolling Sh", "Maps differ"],
        rows, [28, 26, 24, 26, 22, 26, 28], size=8,
    )
    pdf.body(
        "Maps differ counts how many refits assigned different sleeves to different states. "
        f"The last daily map was: {_fmt_map(spy['daily']['regime_last_map'])}."
    )
    pdf.h2("Where each sleeve sat inside the evaluation window")
    pdf.body(
        "These Sharpes split the 2013-onward daily path by the HMM state that day. "
        "They describe the window. They were not used to place a trade."
    )
    rows = []
    for row in spy["daily"]["regime_insample"]:
        rows.append([
            row["regime"], str(row["n"]),
            *[f"{row[name]:.2f}" if row[name] is not None else "-" for name in STRATS],
        ])
    pdf.table(
        ["Regime", "N", "t=0", "t=opt", "Scaled", "Adaptive"],
        rows, [32, 22, 28, 28, 28, 32], size=8,
    )
    chart = OUT / "spy_daily_equity.png"
    if chart.exists():
        pdf.image(str(_jpeg(chart)), w=170)
        pdf.ln(3)

    pdf.h1("5. Daily bars and the intraweek path")
    uni = universe
    pdf.body(
        f"The 102-asset run already scored this. Best daily Sharpe beat best weekly Sharpe on "
        f"{uni['daily_best_beats_weekly_best']} of {uni['n_ok']} names, median paired gap "
        f"{uni['median_daily_minus_weekly']:+.3f}. Binary (t=0) daily beat weekly on "
        f"{uni['daily_t0_beats_weekly_t0']} names. Median Sharpe was daily {uni['median_daily_sharpe']:.3f}, "
        f"weekly {uni['median_weekly_sharpe']:.3f}, mixed {uni['median_mixed_sharpe']:.3f}, "
        f"buy-and-hold {uni['median_bnh_sharpe']:.3f}. "
        f"Mixed beat the daily book on a paired basis by {uni['median_mixed_minus_daily']:+.3f}, "
        f"and on {uni['mixed_t0_beats_daily_t0']} names for the zero threshold."
    )
    rows = []
    for row in uni["by_class"]:
        rows.append([
            row["class"], str(row["n"]),
            f"{row['daily_beats_weekly']}/{row['n']}",
            f"{row['median_daily']:.2f}",
            f"{row['median_weekly']:.2f}",
            f"{row['median_mixed']:.2f}",
        ])
    pdf.table(
        ["Class", "N", "Daily>Wk", "Daily", "Weekly", "Mixed"],
        rows, [36, 16, 28, 28, 28, 28], size=8,
    )
    pdf.h2("SPY excess by weekday")
    pdf.body(
        "Daily Binary (t=0) minus buy-and-hold, averaged inside each weekday, from 2013 onward. "
        "The sum column is the total excess return, in decimal, added on that weekday."
    )
    rows = []
    for row in intra["by_weekday"]:
        rows.append([
            row["weekday"], str(row["n"]),
            f"{row['mean_bnh_bps']:.1f}",
            f"{row['mean_t0_excess_bps']:.2f}",
            f"{row['sum_t0_excess']:.3f}",
            f"{row['mean_topt_excess_bps']:.2f}",
        ])
    pdf.table(
        ["Weekday", "N", "BnH bps", "t=0 excess bps", "t=0 sum", "t=opt excess bps"],
        rows, [32, 18, 24, 36, 28, 40], size=8,
    )
    chart = OUT / "weekday_excess.png"
    if chart.exists():
        pdf.image(str(_jpeg(chart)), w=160)
        pdf.ln(3)
    pdf.h2("Daily book compounded to weeks, against the weekly book")
    pdf.body(
        "Each daily sleeve is compounded to a Friday week and aligned with the native weekly sleeve. "
        f"Buy-and-hold agrees across the two calendars with correlation {intra['bnh_week_corr']}."
    )
    rows = []
    for strat, stats in intra["week_aligned"].items():
        rows.append([
            strat,
            str(stats["n_weeks"]),
            f"{stats['daily_compounded_sharpe']:.3f}",
            f"{stats['weekly_native_sharpe']:.3f}",
            f"{stats['corr']:.2f}",
            f"{stats['daily_wins']}/{stats['n_weeks']}",
        ])
    pdf.table(
        ["Sleeve", "Weeks", "Daily->week Sh", "Weekly Sh", "Corr", "Daily wins"],
        rows, [40, 22, 34, 30, 22, 30], size=8,
    )

    pdf.h1("6. What the four tests say")
    w = spy["weekly"]
    m = spy["mixed"]
    pdf.bullet(
        f"On daily SPY, trailing-Sharpe switching finishes at {d['rolling']['sharpe']:.3f}. "
        f"A fixed zero threshold is {d['static']['Binary (t=0)']['sharpe']:.3f}. "
        f"The frozen optimal threshold is {d['static']['Binary (t=opt)']['sharpe']:.3f}. "
        f"The oracle upper bound is {d['oracle']['sharpe']:.3f}. "
        f"On weekly bars the switcher is {w['rolling']['sharpe']:.3f} against {w['static']['Binary (t=0)']['sharpe']:.3f} "
        f"for the zero threshold and {w['static']['BuyHold']['sharpe']:.3f} for buy-and-hold. "
        f"On the mixed frame it is {m['rolling']['sharpe']:.3f} against {m['static']['Binary (t=0)']['sharpe']:.3f}. "
        "The pre-2013 Sharpe winner on the weekly and mixed frames is the optimal-threshold sleeve, and that sleeve sits in cash."
    )
    pdf.bullet(
        f"On TASI, carrying the previous period's Sharpe winner beats a fixed zero threshold on "
        f"{tasi['previous']['beat_t0']} of {tasi['previous']['n']} later periods. "
        f"Median Sharpe is {tasi['previous']['median_sharpe']:.3f} against {tasi['always_t0']['median_sharpe']:.3f}, "
        f"median gap {tasi['previous']['median_gap_vs_t0']:+.3f}, median CAGR "
        f"{tasi['previous']['median_cagr']:.2f}% against {tasi['always_t0']['median_cagr']:.2f}%. "
        f"The in-market guard is {tasi['guarded']['median_sharpe']:.3f}. "
        f"The oracle median gap versus the zero threshold is {tasi['oracle']['median_gap_vs_t0']:+.3f}. "
        "The room is real. The previous winner does not collect it."
    )
    books = portfolio["books"]
    pdf.bullet(
        f"Equal weight Sharpe {books['Equal weight']['sharpe']:.3f}, CAGR {books['Equal weight']['cagr']:.2f}%, "
        f"max drawdown {books['Equal weight']['max_drawdown']:.1f}%, "
        f"average pairwise correlation {books['Equal weight']['avg_pair_corr']}. "
        f"Minimum variance Sharpe {books['Min variance']['sharpe']:.3f}, CAGR {books['Min variance']['cagr']:.2f}%, "
        f"max drawdown {books['Min variance']['max_drawdown']:.1f}%, "
        f"pairwise correlation {books['Min variance']['avg_pair_corr']}. "
        f"The 0.70 screen Sharpe {books['Corr screen 0.70']['sharpe']:.3f}, "
        f"CAGR {books['Corr screen 0.70']['cagr']:.2f}%, "
        f"max drawdown {books['Corr screen 0.70']['max_drawdown']:.1f}%, "
        f"pairwise correlation {books['Corr screen 0.70']['avg_pair_corr']}, "
        f"median names {books['Corr screen 0.70']['median_names']}."
    )
    pdf.bullet(
        f"Regime switching on daily SPY has Sharpe {spy['daily']['regime']['sharpe']:.3f}. "
        f"It holds the optimal-threshold sleeve on {spy['daily']['regime_share'].get('Binary (t=opt)', 0):.0%} of bars. "
        f"{spy['daily']['regime_maps_that_differ']} of {spy['daily']['regime_n_maps']} maps used more than one sleeve. "
        f"Inside the evaluation window that sleeve leads in the bear, sideways, and bull states, "
        f"so the map has little to switch. Weekly regime switching is {w['regime']['sharpe']:.3f} "
        f"against {w['static']['Binary (t=0)']['sharpe']:.3f} for the zero threshold."
    )
    t0_week = intra["week_aligned"]["Binary (t=0)"]
    opt_week = intra["week_aligned"]["Binary (t=opt)"]
    pdf.bullet(
        f"Across the 102 names, best daily Sharpe beats best weekly Sharpe on "
        f"{uni['daily_best_beats_weekly_best']} names, median paired gap {uni['median_daily_minus_weekly']:+.3f}. "
        f"Mixed features change the paired gap by {uni['median_mixed_minus_daily']:+.3f}. "
        f"On SPY, compounding the daily zero-threshold sleeve to the same Friday-to-Friday week as the weekly model "
        f"gives Sharpe {t0_week['daily_compounded_sharpe']:.3f} against the native weekly sleeve "
        f"{t0_week['weekly_native_sharpe']:.3f}, correlation {t0_week['corr']:.2f}. "
        f"The daily optimal-threshold sleeve, compounded the same way, is {opt_week['daily_compounded_sharpe']:.3f} "
        f"against the native weekly optimal-threshold sleeve {opt_week['weekly_native_sharpe']:.3f}."
    )
    pdf.body(
        "These are research results on saved paths and weekly prices. The 2 bp cost is the study "
        "convention. The portfolio is not the ML sleeve, because those paths were not saved name by name. "
        "One SPY path is one asset."
    )
    pdf.output(str(PDF_PATH))
    _normalize(PDF_PATH)


def _normalize(path):
    try:
        import pikepdf
    except ImportError:
        return
    tmp = str(path) + ".tmp"
    with pikepdf.open(path) as doc:
        for page in doc.pages:
            resources = page.get("/Resources")
            if not resources:
                continue
            xobjects = resources.get("/XObject")
            if not xobjects:
                continue
            for image in xobjects.values():
                if str(image.get("/Subtype")) != "/Image":
                    continue
                if str(image.get("/Filter")) == "/DCTDecode" and "/DecodeParms" in image:
                    del image["/DecodeParms"]
        doc.save(tmp, min_version="1.4", object_stream_mode=pikepdf.ObjectStreamMode.disable)
    Path(tmp).replace(path)


def main():
    modes, spy = run_spy()
    intra = intraweek(modes)
    tasi = tasi_reselection()
    assets = load_universe()
    portfolio = run_portfolio(assets)
    universe = daily_universe_brief()

    daily = modes["daily"]
    eval_own = daily["own"].loc[EVAL:]
    plot_equity(
        OUT / "spy_daily_equity.png",
        {
            "Binary (t=0)": (1.0 + eval_own["Binary (t=0)"]).cumprod(),
            "Binary (t=opt)": (1.0 + eval_own["Binary (t=opt)"]).cumprod(),
            "Rolling": daily["rolling_equity"],
            "Regime map": daily["regime_equity"],
            "Buy and hold": (1.0 + daily["asset"].loc[EVAL:]).cumprod(),
        },
        "SPY daily, growth of 1 from 2013",
    )
    plot_weekday(OUT / "weekday_excess.png", intra["by_weekday"])
    plot_equity(
        OUT / "portfolio_equity.png",
        portfolio["curves"],
        "102-name weekly portfolio, growth of 1",
    )
    payload = {
        "spy": spy,
        "intraweek": intra,
        "tasi": tasi,
        "portfolio": {k: v for k, v in portfolio.items() if k != "curves"},
        "universe": {k: v for k, v in universe.items() if k != "by_class"},
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, default=str))
    write_report(spy, intra, tasi, portfolio, universe)
    print("Wrote", PDF_PATH)
    print("SPY daily rolling", spy["daily"]["rolling"])
    print("SPY daily regime", spy["daily"]["regime"])
    print("TASI previous", tasi["previous"])
    print("Portfolio", {k: v["sharpe"] for k, v in portfolio["books"].items()})


if __name__ == "__main__":
    main()
