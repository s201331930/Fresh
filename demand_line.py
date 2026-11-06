"""Demand-line strategy. One rule, live fills.

At the close, own the asset when buyers have kept the price above its
200-session average. That average is a business year of demand. A break
below it means supply has taken the market. The order goes in at the next
session's open, never at the close that produced the signal.

Nothing here is fitted. 200 sessions, a 15% corporate-action screen, and
2 bps per unit of turnover were chosen before the run. A volume-confirmation
variant is reported beside the rule and does not replace it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "demand_line" / "cache"
OUT = ROOT / "demand_line"
SMA_PERIOD = 200
FLOW_PERIOD = 20
COST_BPS = 2.0
JUMP = 0.15
BARS_PER_YEAR = 252

CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def neutralize_jumps(daily, limit=JUMP):
    """Remove |close-to-close| moves above `limit` from the price path.

    Later percentage changes are unchanged. A bonus issue or a bad print
    stops being a trade. Volume is not scaled.
    """
    px = daily.copy()
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in px.columns]
    rets = px["Close"].pct_change()
    flags = list(rets.index[rets.abs() > limit])
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
    return px, len(flags)


def demand_signal(close, period=SMA_PERIOD):
    """1 when the close is above the demand line. Known at that close."""
    line = close.rolling(period, min_periods=period).mean()
    return (close > line).astype(float).where(line.notna())


def volume_signal(close, volume, period=SMA_PERIOD, flow=FLOW_PERIOD):
    """Demand line, plus recent volume sitting on the up days.

    Signed volume over `flow` sessions is positive when more shares changed
    hands on rising closes than on falling ones. That is the supply/demand
    tell. It is a robustness check, not the traded rule.
    """
    base = demand_signal(close, period)
    signed = np.sign(close.diff()).fillna(0.0) * volume.fillna(0.0)
    pressure = signed.rolling(flow, min_periods=flow).sum()
    confirmed = (pressure > 0).astype(float).where(pressure.notna())
    out = ((base == 1.0) & (confirmed == 1.0)).astype(float)
    return out.where(base.notna() & confirmed.notna())


def simulate(frame, signal, cost_bps=COST_BPS):
    """Fill the close signal at the next open.

    open-to-open return at t is open[t+1] / open[t] - 1.
    The position that earns it is the signal from the previous close,
    because that order is filled at open[t].
    """
    oo = frame["Open"].shift(-1) / frame["Open"] - 1.0
    position = signal.reindex(frame.index).shift(1)
    valid = oo.replace([np.inf, -np.inf], np.nan).notna() & position.notna()
    oo = oo.loc[valid]
    position = position.loc[valid].astype(float)
    if oo.empty:
        return pd.DataFrame(columns=["strat", "bnh", "position", "turnover", "oo", "entry_gap"])
    turnover = position.diff().abs()
    turnover.iloc[0] = float(position.iloc[0])
    strat = position * oo - turnover * (cost_bps / 10000.0)
    # Gap from the prior close to this open. An entry at this open does not earn it.
    gap = frame["Open"] / frame["Close"].shift(1) - 1.0
    gap = gap.reindex(position.index)
    entered = (position > 0) & (position.shift(1).fillna(0.0) <= 0)
    return pd.DataFrame({
        "strat": strat,
        "bnh": oo,
        "position": position,
        "turnover": turnover,
        "oo": oo,
        "entry_gap": gap.where(entered),
    })


def metrics(returns, positions=None, turnover=None, bars_per_year=BARS_PER_YEAR):
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 40:
        return None
    equity = (1.0 + r).cumprod()
    years = max((r.index[-1] - r.index[0]).days / 365.25, 0.25)
    vol = float(r.std())
    sharpe = float(r.mean() / vol * np.sqrt(bars_per_year)) if vol > 1e-12 else 0.0
    peak = equity.cummax()
    dd = float((equity / peak - 1.0).min())
    out = {
        "sharpe": round(sharpe, 3),
        "cagr": round(float(equity.iloc[-1] ** (1.0 / years) - 1.0) * 100, 2),
        "max_drawdown": round(dd * 100, 2),
        "total_return": round(float(equity.iloc[-1] - 1.0) * 100, 2),
        "n_days": int(len(r)),
        "n_years": round(float(years), 2),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
    }
    if positions is not None:
        p = positions.reindex(r.index).fillna(0.0)
        # Mean of the weight. For one asset this is time in the market.
        # For a book it is the share of capital that is invested.
        out["in_market_pct"] = round(float(p.mean() * 100), 1)
    if turnover is not None:
        t = turnover.reindex(r.index).fillna(0.0)
        out["turnover_per_year"] = round(float(t.sum() / years), 2)
    return out


def book(results):
    """Equal weight across sleeves that have a bar that day. Flat sleeves stay in the average."""
    strat = pd.DataFrame({k: v["strat"] for k, v in results.items()})
    bnh = pd.DataFrame({k: v["bnh"] for k, v in results.items()})
    pos = pd.DataFrame({k: v["position"] for k, v in results.items()})
    turn = pd.DataFrame({k: v["turnover"] for k, v in results.items()})
    n = strat.notna().sum(axis=1)
    mask = n > 0
    return pd.DataFrame({
        "strat": strat.mean(axis=1, skipna=True)[mask],
        "bnh": bnh.mean(axis=1, skipna=True)[mask],
        "in_market": pos.mean(axis=1, skipna=True)[mask],
        "turnover": turn.mean(axis=1, skipna=True)[mask],
        "n": n[mask],
    })


def yearly(book_df):
    rows = []
    for year, sub in book_df.groupby(book_df.index.year):
        m_s = metrics(sub["strat"], sub["in_market"], sub["turnover"])
        m_b = metrics(sub["bnh"])
        if m_s is None or m_b is None:
            continue
        rows.append({
            "year": int(year),
            "sharpe": m_s["sharpe"],
            "bnh_sharpe": m_b["sharpe"],
            "cagr": m_s["cagr"],
            "bnh_cagr": m_b["cagr"],
            "max_drawdown": m_s["max_drawdown"],
            "bnh_max_drawdown": m_b["max_drawdown"],
            "in_market_pct": m_s.get("in_market_pct"),
            "n": int(sub["n"].median()) if "n" in sub else None,
        })
    return rows


def self_check():
    """The signal close does not earn the gap into the next open."""
    idx = pd.bdate_range("2020-01-06", periods=6)
    frame = pd.DataFrame({
        "Open": [10.0, 10.0, 20.0, 21.0, 22.0, 23.0],
        "Close": [10.0, 10.0, 20.0, 21.0, 22.0, 23.0],
    }, index=idx)
    # Signal turns on at the second close. Fill is the third open (the gap from 10 to 20).
    signal = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0, 0.0], index=idx)
    out = simulate(frame, signal, cost_bps=2.0)
    # First held open-to-open move is 21/20 - 1, minus 2 bps of entry cost.
    held = out.loc[out["position"] > 0, "strat"]
    if held.empty:
        raise AssertionError("no position")
    first = float(held.iloc[0])
    expected = (21.0 / 20.0 - 1.0) - 0.0002
    if abs(first - expected) > 1e-9:
        raise AssertionError(f"fill {first} != next-open return {expected}")
    if float(out["strat"].max()) > 0.1:
        raise AssertionError("strategy earned the 10-to-20 entry gap")
    # Sell signal on the last close. There is no open after it, so the last
    # held bar is still long: we only sell at an open we do not have here.
    if int((out["position"] > 0).sum()) != 3:
        raise AssertionError(f"expected 3 held opens, got {(out['position'] > 0).sum()}")

    close = pd.Series([1, 2, 3, 4, 5, 1], index=idx, dtype=float)
    sig = demand_signal(close, period=3)
    # Last close is below its 3-session average, so the signal is flat.
    if float(sig.iloc[-1]) != 0.0:
        raise AssertionError("demand line did not go flat on the break")
    if float(sig.iloc[-2]) != 1.0:
        raise AssertionError("demand line should have been long before the break")
    print("self-check passed", flush=True)


def _cache_path(symbol):
    return CACHE / f"{symbol.replace('.', '_').replace('=', '_')}.csv"


def _write_adjusted(symbol, raw):
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    frame = frame.rename(columns={"Adj Close": "AdjClose"})
    keep = [c for c in ["Open", "High", "Low", "Close", "AdjClose", "Volume"] if c in frame.columns]
    frame = frame[keep].copy()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    if "AdjClose" in frame.columns and "Close" in frame.columns:
        close = frame["Close"].replace(0, np.nan)
        factor = frame["AdjClose"] / close
        for col in ["Open", "High", "Low"]:
            if col in frame.columns:
                frame[col] = frame[col] * factor
        frame["Close"] = frame["AdjClose"]
    out = frame[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close", "Open"])
    out = out[(out["Volume"] > 0) & (out["Open"] > 0) & (out["Close"] > 0)]
    out.to_csv(_cache_path(symbol))
    return len(out)


def load_symbol(symbol):
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df


def _cache_usable(symbol):
    path = _cache_path(symbol)
    if not path.exists() or path.stat().st_size < 100:
        return False
    return True


def prefetch(symbols):
    import yfinance as yf
    missing = [s for s in symbols if not _cache_usable(s)]
    print(f"download {len(missing)} of {len(symbols)}", flush=True)
    for i in range(0, len(missing), 40):
        batch = missing[i:i + 40]
        print(f"  {batch[0]} .. {batch[-1]}", flush=True)
        try:
            raw = yf.download(
                batch, start="2000-01-01", auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
        except Exception as exc:
            print(f"  batch failed: {exc}", flush=True)
            raw = None
        for symbol in batch:
            try:
                if raw is None or len(raw) == 0:
                    raise RuntimeError("empty batch")
                if isinstance(raw.columns, pd.MultiIndex) and symbol in raw.columns.get_level_values(0):
                    one = raw[symbol].dropna(how="all")
                elif len(batch) == 1:
                    one = raw
                else:
                    raise RuntimeError("missing")
                n = _write_adjusted(symbol, one)
                print(f"    {symbol}: {n}", flush=True)
            except Exception as exc:
                try:
                    one = yf.Ticker(symbol).history(start="2000-01-01", auto_adjust=False)
                    n = _write_adjusted(symbol, one)
                    print(f"    {symbol}: {n} single ({exc})", flush=True)
                except Exception as exc2:
                    print(f"    fail {symbol}: {exc2}", flush=True)


def run_symbol(symbol, kind):
    raw = load_symbol(symbol)
    if raw is None or len(raw) < SMA_PERIOD + 60:
        return None
    clean, n_jumps = neutralize_jumps(raw)
    close_sig = demand_signal(clean["Close"])
    vol_sig = volume_signal(clean["Close"], clean["Volume"])
    signal = close_sig if kind == "line" else vol_sig
    out = simulate(clean, signal)
    if out.empty:
        return None
    return out, n_jumps


def roster():
    tasi_names = {}
    universe = json.loads((ROOT / "tasi_universe.json").read_text())
    by_code = {a["ticker"]: a for a in universe["assets"]}
    for path in sorted((ROOT / "tasi_periods").glob("*.json")):
        if path.name == "summary.json":
            continue
        rec = json.loads(path.read_text())
        if rec.get("status") != "ok":
            continue
        asset = by_code[rec["ticker"]]
        tasi_names[asset["yahoo"]] = {
            "symbol": asset["yahoo"],
            "label": asset["name"],
            "group": "TASI",
            "book": "tasi",
        }
    global_names = {}
    for path in sorted((ROOT / "daily_universe").glob("*.json")):
        if path.name == "universe_summary.json":
            continue
        rec = json.loads(path.read_text())
        global_names[rec["ticker"]] = {
            "symbol": rec["ticker"],
            "label": rec.get("label") or rec["name"],
            "group": rec.get("asset_class") or "Other",
            "book": "global",
        }
    return tasi_names, global_names


def _pair_table(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["gap"] = frame["sharpe"] - frame["bnh_sharpe"]
    return frame


def summarize_names(ran):
    rows = []
    for symbol, payload in ran.items():
        m = metrics(payload["result"]["strat"], payload["result"]["position"], payload["result"]["turnover"])
        b = metrics(payload["result"]["bnh"])
        if m is None or b is None:
            continue
        gaps = payload["result"]["entry_gap"].dropna()
        rows.append({
            "symbol": symbol,
            "label": payload["label"],
            "group": payload["group"],
            "sharpe": m["sharpe"],
            "bnh_sharpe": b["sharpe"],
            "cagr": m["cagr"],
            "bnh_cagr": b["cagr"],
            "max_drawdown": m["max_drawdown"],
            "bnh_max_drawdown": b["max_drawdown"],
            "in_market_pct": m["in_market_pct"],
            "turnover_per_year": m["turnover_per_year"],
            "n_jumps": payload["n_jumps"],
            "mean_entry_gap_bps": None if gaps.empty else round(float(gaps.mean()) * 1e4, 2),
            "n_entries": int(len(gaps)),
        })
    return _pair_table(rows)


def run_book(catalog, kind):
    ran = {}
    for symbol, meta in catalog.items():
        done = run_symbol(symbol, kind)
        if done is None:
            continue
        result, n_jumps = done
        ran[symbol] = {**meta, "result": result, "n_jumps": n_jumps}
    names = summarize_names(ran)
    portfolio = book({k: v["result"] for k, v in ran.items()}) if ran else pd.DataFrame()
    return ran, names, portfolio


def without_top_contributors(ran, n_drop=10):
    """Same book after removing the sleeves with the largest excess wealth.

    Excess is the compounded strategy return minus the compounded buy-and-hold
    return. If the book only works because of a handful of names, this shows it.
    """
    scores = []
    for symbol, payload in ran.items():
        r = payload["result"]
        wealth_s = float((1.0 + r["strat"]).prod() - 1.0)
        wealth_b = float((1.0 + r["bnh"]).prod() - 1.0)
        scores.append((wealth_s - wealth_b, symbol))
    scores.sort(reverse=True)
    drop = {symbol for _, symbol in scores[:n_drop]}
    kept = {k: v["result"] for k, v in ran.items() if k not in drop}
    return {
        "dropped": [symbol for _, symbol in scores[:n_drop]],
        "portfolio": portfolio_payload(book(kept)) if kept else None,
    }


def portfolio_payload(portfolio):
    if portfolio.empty:
        return None
    m = metrics(portfolio["strat"], portfolio["in_market"], portfolio["turnover"])
    b = metrics(portfolio["bnh"])
    worst = portfolio["strat"].groupby(pd.Grouper(freq="ME")).apply(lambda s: (1 + s).prod() - 1)
    return {
        "strategy": m,
        "bnh": b,
        "gap_sharpe": None if m is None or b is None else round(m["sharpe"] - b["sharpe"], 3),
        "worst_month": None if worst.empty else round(float(worst.min()) * 100, 2),
        "median_names": int(portfolio["n"].median()),
        "yearly": yearly(portfolio),
    }


def main():
    self_check()
    tasi, global_book = roster()
    symbols = sorted(set(tasi) | set(global_book))
    prefetch(symbols)
    report = {"rule": {
        "signal": "close > 200-session average",
        "fill": "next session open",
        "sell": "next session open after the close breaks the average",
        "cost_bps_per_turnover": COST_BPS,
        "jump_screen": JUMP,
        "cash_yield": 0.0,
        "sma_period_precommitted": SMA_PERIOD,
    }}
    for kind, label in (("line", "demand_line"), ("volume", "demand_line_plus_volume")):
        print(f"\n=== {label} ===", flush=True)
        tasi_ran, tasi_names, tasi_book = run_book(tasi, kind)
        glob_ran, glob_names, glob_book = run_book(global_book, kind)
        by_class = {}
        if kind == "line":
            groups = {}
            for symbol, payload in glob_ran.items():
                groups.setdefault(payload["group"], {})[symbol] = payload["result"]
            for group, sleeves in sorted(groups.items()):
                by_class[group] = portfolio_payload(book(sleeves))
        block = {
            "n_tasi": len(tasi_ran),
            "n_global": len(glob_ran),
            "tasi_portfolio": portfolio_payload(tasi_book),
            "global_portfolio": portfolio_payload(glob_book),
            "by_class": by_class,
        }
        if kind == "line":
            block["tasi_without_top10"] = without_top_contributors(tasi_ran)
            block["global_without_top10"] = without_top_contributors(glob_ran)
        if not tasi_names.empty:
            block["tasi_names"] = {
                "n": int(len(tasi_names)),
                "beat": int((tasi_names["gap"] > 0).sum()),
                "median_sharpe": round(float(tasi_names["sharpe"].median()), 3),
                "median_bnh": round(float(tasi_names["bnh_sharpe"].median()), 3),
                "median_gap": round(float(tasi_names["gap"].median()), 3),
                "median_in_market": round(float(tasi_names["in_market_pct"].median()), 1),
                "median_entry_gap_bps": round(float(tasi_names["mean_entry_gap_bps"].median()), 2),
            }
            tasi_names.sort_values("gap", ascending=False).to_csv(OUT / f"{label}_tasi_names.csv", index=False)
        if not glob_names.empty:
            block["global_names"] = {
                "n": int(len(glob_names)),
                "beat": int((glob_names["gap"] > 0).sum()),
                "median_sharpe": round(float(glob_names["sharpe"].median()), 3),
                "median_bnh": round(float(glob_names["bnh_sharpe"].median()), 3),
                "median_gap": round(float(glob_names["gap"].median()), 3),
                "median_in_market": round(float(glob_names["in_market_pct"].median()), 1),
            }
            glob_names.sort_values("gap", ascending=False).to_csv(OUT / f"{label}_global_names.csv", index=False)
        if not tasi_book.empty:
            tasi_book.to_csv(OUT / f"{label}_tasi_portfolio.csv")
        if not glob_book.empty:
            glob_book.to_csv(OUT / f"{label}_global_portfolio.csv")
        report[label] = block
        tp = block["tasi_portfolio"]
        if tp and tp["strategy"]:
            print(
                f"TASI book Sharpe {tp['strategy']['sharpe']} vs BnH {tp['bnh']['sharpe']} "
                f"CAGR {tp['strategy']['cagr']} vs {tp['bnh']['cagr']} "
                f"DD {tp['strategy']['max_drawdown']} in mkt {tp['strategy'].get('in_market_pct')}",
                flush=True,
            )
        gp = block["global_portfolio"]
        if gp and gp["strategy"]:
            print(
                f"Global book Sharpe {gp['strategy']['sharpe']} vs BnH {gp['bnh']['sharpe']} "
                f"CAGR {gp['strategy']['cagr']} vs {gp['bnh']['cagr']}",
                flush=True,
            )
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
