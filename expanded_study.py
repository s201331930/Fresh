"""
Expanded Strategy Selection Study: 100+ assets across all asset classes.

Covers: US stocks, sector ETFs, broad-market ETFs, international ETFs,
commodity ETFs, currency ETFs, forex pairs, bond ETFs, REITs,
TASI stocks, international stocks, and thematic ETFs.

Reuses the walk-forward ML pipeline from multi_asset_backtest.py.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import sys
import traceback
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.tree import DecisionTreeClassifier, export_text
from collections import Counter

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from multi_asset_backtest import run_asset

RESULTS_DIR = Path(__file__).parent / "results"
STUDY_DIR = Path(__file__).parent / "study"
STUDY_DIR.mkdir(exist_ok=True)

EXISTING_20 = [
    {"name": "SPY",      "ticker": "SPY",      "market": "US",   "label": "S&P 500 ETF",           "class": "Broad ETF"},
    {"name": "XLK",      "ticker": "XLK",      "market": "US",   "label": "Technology Sector",      "class": "Sector ETF"},
    {"name": "XLF",      "ticker": "XLF",      "market": "US",   "label": "Financial Sector",       "class": "Sector ETF"},
    {"name": "XLE",      "ticker": "XLE",      "market": "US",   "label": "Energy Sector",          "class": "Sector ETF"},
    {"name": "SADAFCO",  "ticker": "2270.SR",  "market": "TASI", "label": "SADAFCO (Dairy)",        "class": "TASI Stock"},
    {"name": "Al_Rajhi", "ticker": "1120.SR",  "market": "TASI", "label": "Al Rajhi Bank",          "class": "TASI Stock"},
    {"name": "STC",      "ticker": "7010.SR",  "market": "TASI", "label": "Saudi Telecom",          "class": "TASI Stock"},
    {"name": "Mobily",   "ticker": "7020.SR",  "market": "TASI", "label": "Mobily (Telecom)",       "class": "TASI Stock"},
    {"name": "QQQ",      "ticker": "QQQ",      "market": "US",   "label": "Nasdaq 100 ETF",         "class": "Broad ETF"},
    {"name": "IWM",      "ticker": "IWM",      "market": "US",   "label": "Russell 2000 SmallCap",  "class": "Broad ETF"},
    {"name": "XLP",      "ticker": "XLP",      "market": "US",   "label": "Consumer Staples",       "class": "Sector ETF"},
    {"name": "XLV",      "ticker": "XLV",      "market": "US",   "label": "Healthcare Sector",      "class": "Sector ETF"},
    {"name": "XLU",      "ticker": "XLU",      "market": "US",   "label": "Utilities Sector",       "class": "Sector ETF"},
    {"name": "GLD",      "ticker": "GLD",      "market": "US",   "label": "Gold ETF",               "class": "Commodity"},
    {"name": "TLT",      "ticker": "TLT",      "market": "US",   "label": "20+ Year Treasury",      "class": "Bond ETF"},
    {"name": "EEM",      "ticker": "EEM",      "market": "US",   "label": "Emerging Markets",       "class": "Intl ETF"},
    {"name": "EWJ",      "ticker": "EWJ",      "market": "US",   "label": "Japan ETF",              "class": "Intl ETF"},
    {"name": "EWZ",      "ticker": "EWZ",      "market": "US",   "label": "Brazil ETF",             "class": "Intl ETF"},
    {"name": "SABIC",    "ticker": "2010.SR",  "market": "TASI", "label": "SABIC (Chemicals)",      "class": "TASI Stock"},
    {"name": "Maaden",   "ticker": "1211.SR",  "market": "TASI", "label": "Saudi Mining",           "class": "TASI Stock"},
]

NEW_ASSETS = [
    # US Large Cap Stocks
    {"name": "AAPL",  "ticker": "AAPL",  "market": "US", "label": "Apple", "class": "US Stock"},
    {"name": "MSFT",  "ticker": "MSFT",  "market": "US", "label": "Microsoft", "class": "US Stock"},
    {"name": "AMZN",  "ticker": "AMZN",  "market": "US", "label": "Amazon", "class": "US Stock"},
    {"name": "GOOG",  "ticker": "GOOG",  "market": "US", "label": "Alphabet", "class": "US Stock"},
    {"name": "META",  "ticker": "META",  "market": "US", "label": "Meta Platforms", "class": "US Stock"},
    {"name": "NVDA",  "ticker": "NVDA",  "market": "US", "label": "NVIDIA", "class": "US Stock"},
    {"name": "TSLA",  "ticker": "TSLA",  "market": "US", "label": "Tesla", "class": "US Stock"},
    {"name": "JPM",   "ticker": "JPM",   "market": "US", "label": "JP Morgan", "class": "US Stock"},
    {"name": "JNJ",   "ticker": "JNJ",   "market": "US", "label": "Johnson & Johnson", "class": "US Stock"},
    {"name": "PG",    "ticker": "PG",    "market": "US", "label": "Procter & Gamble", "class": "US Stock"},
    {"name": "UNH",   "ticker": "UNH",   "market": "US", "label": "UnitedHealth", "class": "US Stock"},
    {"name": "V",     "ticker": "V",     "market": "US", "label": "Visa", "class": "US Stock"},
    {"name": "MA",    "ticker": "MA",    "market": "US", "label": "Mastercard", "class": "US Stock"},
    {"name": "HD",    "ticker": "HD",    "market": "US", "label": "Home Depot", "class": "US Stock"},
    {"name": "KO",    "ticker": "KO",    "market": "US", "label": "Coca-Cola", "class": "US Stock"},
    {"name": "PEP",   "ticker": "PEP",   "market": "US", "label": "PepsiCo", "class": "US Stock"},
    {"name": "WMT",   "ticker": "WMT",   "market": "US", "label": "Walmart", "class": "US Stock"},
    {"name": "DIS",   "ticker": "DIS",   "market": "US", "label": "Disney", "class": "US Stock"},
    {"name": "NFLX",  "ticker": "NFLX",  "market": "US", "label": "Netflix", "class": "US Stock"},
    {"name": "BA",    "ticker": "BA",    "market": "US", "label": "Boeing", "class": "US Stock"},
    # Sector ETFs
    {"name": "XLI",   "ticker": "XLI",   "market": "US", "label": "Industrials Sector", "class": "Sector ETF"},
    {"name": "XLB",   "ticker": "XLB",   "market": "US", "label": "Materials Sector", "class": "Sector ETF"},
    {"name": "XLRE",  "ticker": "XLRE",  "market": "US", "label": "Real Estate Sector", "class": "Sector ETF"},
    # Broad Market ETFs
    {"name": "DIA",   "ticker": "DIA",   "market": "US", "label": "Dow Jones ETF", "class": "Broad ETF"},
    {"name": "MDY",   "ticker": "MDY",   "market": "US", "label": "S&P MidCap 400", "class": "Broad ETF"},
    # International ETFs
    {"name": "EWG",   "ticker": "EWG",   "market": "US", "label": "Germany ETF", "class": "Intl ETF"},
    {"name": "EWU",   "ticker": "EWU",   "market": "US", "label": "UK ETF", "class": "Intl ETF"},
    {"name": "EWA",   "ticker": "EWA",   "market": "US", "label": "Australia ETF", "class": "Intl ETF"},
    {"name": "EWC",   "ticker": "EWC",   "market": "US", "label": "Canada ETF", "class": "Intl ETF"},
    {"name": "EWT",   "ticker": "EWT",   "market": "US", "label": "Taiwan ETF", "class": "Intl ETF"},
    {"name": "EWY",   "ticker": "EWY",   "market": "US", "label": "South Korea ETF", "class": "Intl ETF"},
    {"name": "EWS",   "ticker": "EWS",   "market": "US", "label": "Singapore ETF", "class": "Intl ETF"},
    {"name": "VWO",   "ticker": "VWO",   "market": "US", "label": "Vanguard EM ETF", "class": "Intl ETF"},
    {"name": "INDA",  "ticker": "INDA",  "market": "US", "label": "India ETF", "class": "Intl ETF"},
    {"name": "FXI",   "ticker": "FXI",   "market": "US", "label": "China Large-Cap ETF", "class": "Intl ETF"},
    # Commodity ETFs
    {"name": "SLV",   "ticker": "SLV",   "market": "US", "label": "Silver ETF", "class": "Commodity"},
    {"name": "USO",   "ticker": "USO",   "market": "US", "label": "US Oil Fund", "class": "Commodity"},
    {"name": "DBA",   "ticker": "DBA",   "market": "US", "label": "Agriculture ETF", "class": "Commodity"},
    {"name": "DBC",   "ticker": "DBC",   "market": "US", "label": "Commodity Index", "class": "Commodity"},
    {"name": "PPLT",  "ticker": "PPLT",  "market": "US", "label": "Platinum ETF", "class": "Commodity"},
    {"name": "PALL",  "ticker": "PALL",  "market": "US", "label": "Palladium ETF", "class": "Commodity"},
    {"name": "UNG",   "ticker": "UNG",   "market": "US", "label": "Natural Gas ETF", "class": "Commodity"},
    {"name": "CPER",  "ticker": "CPER",  "market": "US", "label": "Copper ETF", "class": "Commodity"},
    # Currency ETFs
    {"name": "FXE",   "ticker": "FXE",   "market": "US", "label": "Euro Currency ETF", "class": "Currency"},
    {"name": "FXY",   "ticker": "FXY",   "market": "US", "label": "Japanese Yen ETF", "class": "Currency"},
    {"name": "FXB",   "ticker": "FXB",   "market": "US", "label": "British Pound ETF", "class": "Currency"},
    {"name": "FXA",   "ticker": "FXA",   "market": "US", "label": "Australian Dollar ETF", "class": "Currency"},
    {"name": "FXC",   "ticker": "FXC",   "market": "US", "label": "Canadian Dollar ETF", "class": "Currency"},
    {"name": "UUP",   "ticker": "UUP",   "market": "US", "label": "US Dollar Bull ETF", "class": "Currency"},
    {"name": "FXF",   "ticker": "FXF",   "market": "US", "label": "Swiss Franc ETF", "class": "Currency"},
    {"name": "CEW",   "ticker": "CEW",   "market": "US", "label": "EM Currency ETF", "class": "Currency"},
    # Forex Pairs
    {"name": "EURUSD", "ticker": "EURUSD=X", "market": "US", "label": "EUR/USD", "class": "Forex"},
    {"name": "GBPUSD", "ticker": "GBPUSD=X", "market": "US", "label": "GBP/USD", "class": "Forex"},
    {"name": "USDJPY", "ticker": "JPY=X",    "market": "US", "label": "USD/JPY", "class": "Forex"},
    {"name": "AUDUSD", "ticker": "AUDUSD=X", "market": "US", "label": "AUD/USD", "class": "Forex"},
    {"name": "USDCHF", "ticker": "CHF=X",    "market": "US", "label": "USD/CHF", "class": "Forex"},
    {"name": "USDCAD", "ticker": "CAD=X",    "market": "US", "label": "USD/CAD", "class": "Forex"},
    # Bond ETFs
    {"name": "IEF",   "ticker": "IEF",   "market": "US", "label": "7-10 Year Treasury", "class": "Bond ETF"},
    {"name": "SHY",   "ticker": "SHY",   "market": "US", "label": "1-3 Year Treasury", "class": "Bond ETF"},
    {"name": "HYG",   "ticker": "HYG",   "market": "US", "label": "High Yield Corporate", "class": "Bond ETF"},
    {"name": "LQD",   "ticker": "LQD",   "market": "US", "label": "IG Corporate Bond", "class": "Bond ETF"},
    {"name": "AGG",   "ticker": "AGG",   "market": "US", "label": "US Aggregate Bond", "class": "Bond ETF"},
    {"name": "BND",   "ticker": "BND",   "market": "US", "label": "Total Bond Market", "class": "Bond ETF"},
    {"name": "MUB",   "ticker": "MUB",   "market": "US", "label": "Municipal Bond", "class": "Bond ETF"},
    {"name": "TIP",   "ticker": "TIP",   "market": "US", "label": "TIPS Bond ETF", "class": "Bond ETF"},
    # REITs
    {"name": "VNQ",   "ticker": "VNQ",   "market": "US", "label": "US REIT ETF", "class": "REIT"},
    {"name": "IYR",   "ticker": "IYR",   "market": "US", "label": "iShares US REIT", "class": "REIT"},
    {"name": "REM",   "ticker": "REM",   "market": "US", "label": "Mortgage REIT ETF", "class": "REIT"},
    {"name": "VNQI",  "ticker": "VNQI",  "market": "US", "label": "Intl REIT ETF", "class": "REIT"},
    # TASI Stocks
    {"name": "Riyad_Bank",   "ticker": "1010.SR", "market": "TASI", "label": "Riyad Bank", "class": "TASI Stock"},
    {"name": "Alinma",       "ticker": "1150.SR", "market": "TASI", "label": "Alinma Bank", "class": "TASI Stock"},
    {"name": "Saudi_Kayan",  "ticker": "2350.SR", "market": "TASI", "label": "Saudi Kayan", "class": "TASI Stock"},
    {"name": "Tawuniya",     "ticker": "8010.SR", "market": "TASI", "label": "Tawuniya Insurance", "class": "TASI Stock"},
    {"name": "Dar_Al_Arkan", "ticker": "4300.SR", "market": "TASI", "label": "Dar Al Arkan", "class": "TASI Stock"},
    {"name": "Jarir",        "ticker": "4190.SR", "market": "TASI", "label": "Jarir Marketing", "class": "TASI Stock"},
    # International Stocks
    {"name": "BABA",  "ticker": "BABA",  "market": "US", "label": "Alibaba", "class": "Intl Stock"},
    {"name": "TSM",   "ticker": "TSM",   "market": "US", "label": "TSMC", "class": "Intl Stock"},
    {"name": "NVO",   "ticker": "NVO",   "market": "US", "label": "Novo Nordisk", "class": "Intl Stock"},
    {"name": "TM",    "ticker": "TM",    "market": "US", "label": "Toyota Motor", "class": "Intl Stock"},
    {"name": "SHEL",  "ticker": "SHEL",  "market": "US", "label": "Shell plc", "class": "Intl Stock"},
    {"name": "ASML",  "ticker": "ASML",  "market": "US", "label": "ASML Holdings", "class": "Intl Stock"},
    {"name": "SAP",   "ticker": "SAP",   "market": "US", "label": "SAP SE", "class": "Intl Stock"},
    {"name": "RIO",   "ticker": "RIO",   "market": "US", "label": "Rio Tinto", "class": "Intl Stock"},
    # Thematic ETFs
    {"name": "ARKK",  "ticker": "ARKK",  "market": "US", "label": "ARK Innovation", "class": "Thematic ETF"},
    {"name": "XBI",   "ticker": "XBI",   "market": "US", "label": "Biotech ETF", "class": "Thematic ETF"},
    {"name": "HACK",  "ticker": "HACK",  "market": "US", "label": "Cybersecurity ETF", "class": "Thematic ETF"},
    {"name": "TAN",   "ticker": "TAN",   "market": "US", "label": "Solar ETF", "class": "Thematic ETF"},
    {"name": "KWEB",  "ticker": "KWEB",  "market": "US", "label": "China Internet ETF", "class": "Thematic ETF"},
]

ALL_ASSETS = EXISTING_20 + NEW_ASSETS


def fetch_weekly(ticker, market):
    tk = yf.Ticker(ticker)
    daily = tk.history(start="2000-01-01", auto_adjust=False)
    if daily.empty:
        return None
    rule = "W-THU" if market == "TASI" else "W-FRI"
    weekly = pd.DataFrame()
    weekly["Open"] = daily["Open"].resample(rule).first()
    weekly["High"] = daily["High"].resample(rule).max()
    weekly["Low"] = daily["Low"].resample(rule).min()
    weekly["Close"] = daily["Close"].resample(rule).last()
    weekly["Volume"] = daily["Volume"].resample(rule).sum()
    if "Adj Close" in daily.columns:
        adj = daily["Adj Close"].resample(rule).last()
        a2p = adj / weekly["Close"]
        weekly["Close"] = adj
        weekly["Open"] *= a2p
        weekly["High"] *= a2p
        weekly["Low"] *= a2p
    weekly.dropna(subset=["Close"], inplace=True)
    weekly = weekly[weekly.index.notna()]
    if "Volume" in weekly.columns:
        weekly = weekly[(weekly["Volume"] > 0) | (weekly["Volume"].isna())]
    weekly.index = weekly.index.tz_localize(None)
    return weekly


def hurst_exponent(ts, max_lag=100):
    ts = ts.dropna().values
    n = len(ts)
    if n < 50:
        return 0.5
    lags = range(10, min(max_lag, n // 4))
    rs_values = []
    for lag in lags:
        subseries = [ts[i:i + lag] for i in range(0, n - lag, lag)]
        if len(subseries) < 2:
            continue
        rs_list = []
        for s in subseries:
            m = s.mean()
            y = np.cumsum(s - m)
            r = y.max() - y.min()
            std = s.std()
            if std > 0:
                rs_list.append(r / std)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
    if len(rs_values) < 3:
        return 0.5
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _, _, _, _ = sp_stats.linregress(x, y)
    return np.clip(slope, 0.0, 1.0)


def compute_characteristics(df):
    close = df["Close"]
    returns = close.pct_change().dropna() * 100
    n_weeks = len(returns)
    ann_factor = np.sqrt(52)
    ch = {}
    ch["n_weeks"] = n_weeks
    ch["ann_volatility"] = round(float(returns.std() * ann_factor), 2)
    ch["mean_weekly_ret"] = round(float(returns.mean()), 4)
    total_ret = close.iloc[-1] / close.iloc[0]
    n_years = n_weeks / 52
    ch["ann_return"] = round(float((total_ret ** (1 / max(n_years, 0.01)) - 1) * 100), 2)
    ch["raw_sharpe"] = round(float(returns.mean() / max(returns.std(), 1e-10) * ann_factor), 3)
    ch["skewness"] = round(float(returns.skew()), 3)
    ch["kurtosis"] = round(float(returns.kurtosis()), 3)
    ch["autocorr_1"] = round(float(returns.autocorr(lag=1)), 4) if n_weeks > 10 else 0
    ch["autocorr_4"] = round(float(returns.autocorr(lag=4)), 4) if n_weeks > 10 else 0
    ch["hurst"] = round(hurst_exponent(returns), 3)
    cum = (1 + returns / 100).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    ch["max_drawdown"] = round(float(dd.min() * 100), 2)
    ch["up_week_ratio"] = round(float((returns > 0).mean()), 3)
    ch["mean_abs_return"] = round(float(returns.abs().mean()), 4)
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    ch["tail_ratio"] = round(float(p95 / max(p5, 1e-10)), 3)
    neg_ret = returns[returns < 0]
    ch["downside_vol"] = round(float(neg_ret.std() * ann_factor), 2) if len(neg_ret) > 5 else 0
    sma20 = close.rolling(20).mean()
    above = (close > sma20).astype(int)
    regime_changes = above.diff().abs().sum()
    ch["regime_changes_per_year"] = round(float(regime_changes / max(n_years, 0.01)), 2)
    sma50 = close.rolling(50).mean().dropna()
    close_aligned = close.loc[sma50.index]
    ch["pct_above_sma50"] = round(float((close_aligned > sma50).mean()), 3)
    tr = pd.DataFrame({
        "hl": df["High"] - df["Low"],
        "hc": abs(df["High"] - df["Close"].shift(1)),
        "lc": abs(df["Low"] - df["Close"].shift(1)),
    }).max(axis=1)
    atr_pct = (tr / close * 100).mean()
    ch["avg_true_range_pct"] = round(float(atr_pct), 3)
    vol_cv = df["Volume"].std() / max(df["Volume"].mean(), 1) if "Volume" in df.columns and df["Volume"].mean() > 0 else 0
    ch["volume_cv"] = round(float(vol_cv), 3)
    ch["mean_reversion_score"] = round(-ch["autocorr_1"], 4)
    ch["calmar_ratio"] = round(float(ch["ann_return"] / max(abs(ch["max_drawdown"]), 0.01)), 3)
    return ch


def load_existing_results():
    results = {}
    spy_strat = Path(__file__).parent / "strategy" / "backtest_summary.json"
    if spy_strat.exists():
        with open(spy_strat) as f:
            s = json.load(f)
        results["SPY"] = s["strategies"]
    for d in RESULTS_DIR.iterdir():
        if d.is_dir() and (d / "summary.json").exists():
            with open(d / "summary.json") as f:
                s = json.load(f)
            results[s["name"]] = s["strategies"]
    return results


def run_pipeline_for_asset(asset):
    try:
        result = run_asset(asset)
        if result:
            return result["strategies"]
    except Exception as e:
        print(f"  FAILED {asset['name']}: {e}")
        traceback.print_exc()
    return None


def main():
    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing results: {sorted(existing.keys())}")

    to_run = [a for a in ALL_ASSETS if a["name"] not in existing]
    print(f"Need to run: {len(to_run)} assets")
    print(f"Total target: {len(ALL_ASSETS)} assets")

    # Run pipeline for new assets
    completed = 0
    failed = []
    for i, asset in enumerate(to_run):
        print(f"\n[{i+1}/{len(to_run)}] Running: {asset['name']} ({asset['label']}) ...", flush=True)
        strats = run_pipeline_for_asset(asset)
        if strats:
            existing[asset["name"]] = strats
            completed += 1
            print(f"  Done ({completed} completed, {len(failed)} failed)")
        else:
            failed.append(asset["name"])
            print(f"  FAILED ({completed} completed, {len(failed)} failed)")

    print(f"\n{'='*70}")
    print(f"Pipeline complete: {completed} new + {len(existing)-completed} existing = {len(existing)} total")
    if failed:
        print(f"Failed: {failed}")

    # Compute characteristics for ALL assets
    print(f"\n{'='*70}\n  COMPUTING CHARACTERISTICS FOR {len(ALL_ASSETS)} ASSETS\n{'='*70}")
    all_chars = {}
    for asset in ALL_ASSETS:
        name = asset["name"]
        if name not in existing:
            continue
        df = fetch_weekly(asset["ticker"], asset["market"])
        if df is None or len(df) < 100:
            print(f"  {name}: insufficient data, skipping")
            continue

        ch = compute_characteristics(df)
        strats = existing[name]

        strat_list = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
        best_by_sharpe = max(strat_list, key=lambda s: strats.get(s, {}).get("sharpe", -999))
        ch["best_strategy_sharpe"] = best_by_sharpe
        ch["best_sharpe_value"] = strats[best_by_sharpe]["sharpe"]
        ch["bnh_sharpe"] = strats[best_by_sharpe].get("bnh_sharpe", 0)
        ch["sharpe_excess"] = round(ch["best_sharpe_value"] - ch["bnh_sharpe"], 3)
        ch["market"] = asset["market"]
        ch["label"] = asset["label"]
        ch["asset_class"] = asset.get("class", "Other")

        for sn in strat_list:
            m = strats.get(sn, {})
            if m:
                ch[f"sharpe_{sn}"] = float(m["sharpe"])
                ch[f"return_{sn}"] = float(m["total_return"])
                ch[f"maxdd_{sn}"] = float(m["max_drawdown"])
                ch[f"diraccy_{sn}"] = float(m.get("dir_accuracy", 0))

        all_chars[name] = ch
        bs_s = ch["best_sharpe_value"]
        print(f"  {name:18s} {asset.get('class',''):15s} vol={ch['ann_volatility']:6.1f} "
              f"hurst={ch['hurst']:.3f} ac1={ch['autocorr_1']:+.4f} "
              f"best={best_by_sharpe:18s} sharpe={bs_s:.3f}")

    n_assets = len(all_chars)
    print(f"\nTotal assets with characteristics: {n_assets}")

    # Build characteristics DataFrame
    chars_df = pd.DataFrame(all_chars).T
    chars_df.index.name = "asset"

    # Strategy groups
    strat_groups = {}
    for name, ch in all_chars.items():
        bs = ch["best_strategy_sharpe"]
        strat_groups.setdefault(bs, []).append(name)

    print(f"\n{'='*70}\n  STRATEGY DISTRIBUTION\n{'='*70}")
    for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
        assets_in = strat_groups.get(sn, [])
        print(f"  {sn:18s}: {len(assets_in):3d} assets")
        # Show by asset class
        by_class = Counter(all_chars[a]["asset_class"] for a in assets_in)
        for cls, cnt in sorted(by_class.items()):
            cls_assets = [a for a in assets_in if all_chars[a]["asset_class"] == cls]
            print(f"    {cls:18s}: {cnt:2d} ({', '.join(cls_assets[:6])}{'...' if cnt>6 else ''})")

    # Decision Tree
    print(f"\n{'='*70}\n  DECISION TREE ANALYSIS\n{'='*70}")
    feature_cols = [
        "ann_volatility", "hurst", "autocorr_1", "autocorr_4", "skewness", "kurtosis",
        "regime_changes_per_year", "pct_above_sma50", "avg_true_range_pct",
        "up_week_ratio", "max_drawdown", "mean_reversion_score",
        "downside_vol", "tail_ratio", "volume_cv", "calmar_ratio",
        "mean_abs_return", "mean_weekly_ret",
    ]

    X, y, names_list = [], [], []
    for name in sorted(all_chars):
        ch = all_chars[name]
        row = [float(ch.get(f, 0)) for f in feature_cols]
        if any(np.isnan(v) or np.isinf(v) for v in row):
            continue
        X.append(row)
        y.append(ch["best_strategy_sharpe"])
        names_list.append(name)

    X = np.array(X)
    y = np.array(y)

    # Try multiple depths and pick best with leave-one-out cross-validation
    best_depth = 3
    best_loo_acc = 0
    for depth in [3, 4, 5]:
        loo_correct = 0
        for i in range(len(X)):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2, random_state=42)
            dt.fit(X_train, y_train)
            pred = dt.predict(X[i:i+1])[0]
            if pred == y[i]:
                loo_correct += 1
        loo_acc = loo_correct / len(X) * 100
        print(f"  Depth={depth}: LOO accuracy = {loo_correct}/{len(X)} = {loo_acc:.1f}%")
        if loo_acc > best_loo_acc:
            best_loo_acc = loo_acc
            best_depth = depth

    print(f"\n  Best depth: {best_depth} (LOO accuracy: {best_loo_acc:.1f}%)")

    dt_final = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=2, random_state=42)
    dt_final.fit(X, y)
    train_acc = dt_final.score(X, y) * 100
    print(f"  Training accuracy: {train_acc:.1f}%")

    tree_text = export_text(dt_final, feature_names=feature_cols)
    print(f"\n  Decision Tree Rules:\n{tree_text}")

    preds = dt_final.predict(X)
    importances = dt_final.feature_importances_
    print("  Feature importances:")
    for idx in np.argsort(importances)[::-1]:
        if importances[idx] > 0.01:
            print(f"    {feature_cols[idx]:30s}: {importances[idx]:.3f}")

    # Confusion matrix
    print("\n  Predictions:")
    correct = 0
    for i, name in enumerate(names_list):
        mark = "OK" if preds[i] == y[i] else "MISS"
        if preds[i] == y[i]:
            correct += 1
        print(f"    {name:18s}: pred={preds[i]:18s}  actual={y[i]:18s}  [{mark}]")

    # Mean characteristics by strategy
    print(f"\n{'='*70}\n  MEAN CHARACTERISTICS BY BEST STRATEGY\n{'='*70}")
    for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
        assets_in = strat_groups.get(sn, [])
        if not assets_in:
            continue
        sub = chars_df.loc[[a for a in assets_in if a in chars_df.index], feature_cols].astype(float)
        print(f"\n  {sn} ({len(assets_in)} assets):")
        for col in feature_cols:
            print(f"    {col:30s}: {sub[col].mean():10.3f} (std={sub[col].std():8.3f})")

    # Discriminating features with effect sizes
    print(f"\n{'='*70}\n  TOP DISCRIMINATING FEATURES PER STRATEGY\n{'='*70}")
    for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
        assets_in = [a for a in strat_groups.get(sn, []) if a in chars_df.index]
        rest = [a for a in chars_df.index if a not in assets_in]
        if not assets_in or not rest:
            continue
        sub = chars_df.loc[assets_in, feature_cols].astype(float)
        rest_df = chars_df.loc[rest, feature_cols].astype(float)
        diffs = []
        for col in feature_cols:
            pooled_std = chars_df.loc[chars_df.index.isin(list(all_chars.keys())), col].astype(float).std()
            if pooled_std > 0:
                effect = abs(sub[col].mean() - rest_df[col].mean()) / pooled_std
                direction = "HIGHER" if sub[col].mean() > rest_df[col].mean() else "LOWER"
                diffs.append((col, effect, direction, sub[col].mean(), rest_df[col].mean()))
        diffs.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  {sn} ({len(assets_in)} assets):")
        for col, es, d, gm, rm in diffs[:6]:
            print(f"    {col:30s}: {d:6s} ({gm:+10.3f} vs {rm:+10.3f}, d={es:.2f})")

    # Save everything
    chars_df.to_csv(STUDY_DIR / "expanded_characteristics.csv", float_format="%.4f")

    study_summary = {
        "n_assets": n_assets,
        "assets": sorted(all_chars.keys()),
        "strategy_groups": {k: sorted(v) for k, v in strat_groups.items()},
        "decision_tree": {
            "depth": best_depth,
            "loo_accuracy_pct": round(best_loo_acc, 1),
            "train_accuracy_pct": round(train_acc, 1),
            "rules": tree_text,
            "feature_importances": {feature_cols[i]: round(float(importances[i]), 4) for i in range(len(feature_cols)) if importances[i] > 0.001},
        },
        "predictions": {names_list[i]: {"predicted": str(preds[i]), "actual": str(y[i]), "correct": bool(preds[i] == y[i])} for i in range(len(names_list))},
        "characteristics": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv) for kk, vv in v.items()} for k, v in all_chars.items()},
        "by_asset_class": {cls: sorted(names) for cls, names in Counter(ch["asset_class"] for ch in all_chars.values()).items() for names in [[n for n, c in all_chars.items() if c["asset_class"] == cls]]},
    }
    with open(STUDY_DIR / "expanded_study_summary.json", "w") as f:
        json.dump(study_summary, f, indent=2, default=str)

    # Generate charts
    _generate_charts(all_chars, strat_groups, chars_df, feature_cols)

    print(f"\n{'='*70}")
    print(f"EXPANDED STUDY COMPLETE: {n_assets} assets")
    print(f"  Decision Tree LOO accuracy: {best_loo_acc:.1f}%")
    print(f"  Training accuracy: {train_acc:.1f}%")
    print(f"  Results saved to {STUDY_DIR}/")


def _generate_charts(all_chars, strat_groups, chars_df, feature_cols):
    strat_colors = {
        "Binary (t=0)": "#26a69a", "Binary (t=opt)": "#ab47bc",
        "Scaled": "#42a5f5", "Adaptive": "#ffa726"
    }

    # 1. Sharpe heatmap
    strat_list = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
    asset_names = sorted(all_chars.keys())
    sharpe_matrix = []
    for asset in asset_names:
        row = [float(all_chars[asset].get(f"sharpe_{sn}", 0)) for sn in strat_list]
        sharpe_matrix.append(row)
    sharpe_arr = np.array(sharpe_matrix)

    fig, ax = plt.subplots(figsize=(10, max(16, len(asset_names) * 0.25)))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    im = ax.imshow(sharpe_arr, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=1.5)
    ax.set_xticks(range(len(strat_list)))
    ax.set_xticklabels(strat_list, color="#cccccc", fontsize=8)
    ax.set_yticks(range(len(asset_names)))
    ax.set_yticklabels(asset_names, color="#cccccc", fontsize=5)
    for i in range(len(asset_names)):
        best_idx = np.argmax(sharpe_arr[i])
        for j in range(len(strat_list)):
            weight = "bold" if j == best_idx else "normal"
            color = "black" if sharpe_arr[i, j] > 0.5 else "white"
            ax.text(j, i, f"{sharpe_arr[i, j]:.2f}", ha="center", va="center",
                   fontsize=4, fontweight=weight, color=color)
    ax.set_title(f"Sharpe Ratio Heatmap ({len(asset_names)} Assets)", color="white", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.ax.tick_params(colors="#cccccc")
    plt.tight_layout()
    fig.savefig(STUDY_DIR / "expanded_sharpe_heatmap.png", dpi=200,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # 2. Strategy selection scatter: vol vs regime changes
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    fig2.patch.set_facecolor("#1e1e1e")
    ax2.set_facecolor("#1e1e1e")
    for sn, color in strat_colors.items():
        assets_in = strat_groups.get(sn, [])
        if not assets_in:
            continue
        vols = [float(all_chars[a]["ann_volatility"]) for a in assets_in]
        rcs = [float(all_chars[a]["regime_changes_per_year"]) for a in assets_in]
        ax2.scatter(vols, rcs, color=color, s=60, label=f"{sn} ({len(assets_in)})",
                   alpha=0.8, edgecolors="white", linewidth=0.3)
        for i, a in enumerate(assets_in):
            ax2.annotate(a, (vols[i], rcs[i]), fontsize=5, color="#cccccc",
                        ha="center", va="bottom")
    ax2.set_xlabel("Annualized Volatility (%)", color="#cccccc", fontsize=11)
    ax2.set_ylabel("Regime Changes per Year", color="#cccccc", fontsize=11)
    ax2.set_title(f"Strategy Selection Map ({len(all_chars)} Assets)", color="white", fontsize=13)
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=9)
    ax2.tick_params(colors="#cccccc")
    ax2.grid(color="#333333", alpha=0.5)
    for s in ax2.spines.values():
        s.set_color("#444444")
    plt.tight_layout()
    fig2.savefig(STUDY_DIR / "expanded_strategy_map.png", dpi=150,
                 bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()

    # 3. Strategy distribution by asset class
    class_strat = {}
    for name, ch in all_chars.items():
        cls = ch["asset_class"]
        sn = ch["best_strategy_sharpe"]
        class_strat.setdefault(cls, Counter())[sn] += 1

    classes = sorted(class_strat.keys())
    fig3, ax3 = plt.subplots(figsize=(14, 7))
    fig3.patch.set_facecolor("#1e1e1e")
    ax3.set_facecolor("#1e1e1e")
    x = np.arange(len(classes))
    width = 0.2
    for i, sn in enumerate(strat_list):
        vals = [class_strat[cls].get(sn, 0) for cls in classes]
        ax3.bar(x + i * width, vals, width, label=sn, color=strat_colors[sn], alpha=0.85)
    ax3.set_xticks(x + 1.5 * width)
    ax3.set_xticklabels(classes, color="#cccccc", fontsize=8, rotation=30, ha="right")
    ax3.set_ylabel("Number of Assets", color="#cccccc")
    ax3.set_title("Strategy Distribution by Asset Class", color="white", fontsize=13)
    ax3.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=9)
    ax3.tick_params(colors="#cccccc")
    ax3.grid(axis="y", color="#333333", alpha=0.5)
    for s in ax3.spines.values():
        s.set_color("#444444")
    plt.tight_layout()
    fig3.savefig(STUDY_DIR / "expanded_class_distribution.png", dpi=150,
                 bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close()

    # 4. Key characteristic distributions by strategy
    key_feats = ["up_week_ratio", "pct_above_sma50", "mean_reversion_score",
                 "regime_changes_per_year", "ann_volatility", "hurst"]
    fig4, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig4.patch.set_facecolor("#1e1e1e")
    fig4.suptitle("Key Characteristic Distributions by Best Strategy", color="white", fontsize=14, y=0.98)
    for idx, feat in enumerate(key_feats):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#1e1e1e")
        data_by_strat = []
        labels = []
        colors = []
        for sn in strat_list:
            assets_in = strat_groups.get(sn, [])
            if assets_in:
                vals = [float(all_chars[a][feat]) for a in assets_in if a in all_chars]
                data_by_strat.append(vals)
                labels.append(f"{sn}\n(n={len(vals)})")
                colors.append(strat_colors[sn])
        try:
            bp = ax.boxplot(data_by_strat, tick_labels=labels, patch_artist=True, widths=0.6)
        except TypeError:
            bp = ax.boxplot(data_by_strat, labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for el in ["whiskers", "caps", "medians"]:
            plt.setp(bp[el], color="white")
        ax.set_title(feat.replace("_", " ").title(), color="white", fontsize=10)
        ax.tick_params(colors="#cccccc", labelsize=7)
        ax.grid(axis="y", color="#333333", alpha=0.5)
        for s in ax.spines.values():
            s.set_color("#444444")
    plt.tight_layout()
    fig4.savefig(STUDY_DIR / "expanded_characteristic_distributions.png", dpi=150,
                 bbox_inches="tight", facecolor=fig4.get_facecolor())
    plt.close()

    print(f"  Charts saved to {STUDY_DIR}/")


if __name__ == "__main__":
    main()
