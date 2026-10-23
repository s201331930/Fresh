"""
Strategy Selection Study: Which asset characteristics predict the best strategy?

Computes intrinsic asset characteristics on the IN-SAMPLE period (before OOS
predictions begin), then maps them to which strategy variant performs best.
No look-ahead bias: characteristics are measured before the strategy is applied.

Expands the asset universe to ~20 instruments for statistical credibility.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).parent / "results"
STUDY_DIR = Path(__file__).parent / "study"
STUDY_DIR.mkdir(exist_ok=True)

ALL_ASSETS = [
    # Original 8
    {"name": "SPY",      "ticker": "SPY",      "market": "US",   "label": "S&P 500 ETF"},
    {"name": "XLK",      "ticker": "XLK",      "market": "US",   "label": "Technology Sector"},
    {"name": "XLF",      "ticker": "XLF",      "market": "US",   "label": "Financial Sector"},
    {"name": "XLE",      "ticker": "XLE",      "market": "US",   "label": "Energy Sector"},
    {"name": "SADAFCO",  "ticker": "2270.SR",  "market": "TASI", "label": "SADAFCO (Dairy)"},
    {"name": "Al_Rajhi", "ticker": "1120.SR",  "market": "TASI", "label": "Al Rajhi Bank"},
    {"name": "STC",      "ticker": "7010.SR",  "market": "TASI", "label": "Saudi Telecom"},
    {"name": "Mobily",   "ticker": "7020.SR",  "market": "TASI", "label": "Mobily (Telecom)"},
    # New 12
    {"name": "QQQ",      "ticker": "QQQ",      "market": "US",   "label": "Nasdaq 100 ETF"},
    {"name": "IWM",      "ticker": "IWM",      "market": "US",   "label": "Russell 2000 SmallCap"},
    {"name": "XLP",      "ticker": "XLP",      "market": "US",   "label": "Consumer Staples"},
    {"name": "XLV",      "ticker": "XLV",      "market": "US",   "label": "Healthcare Sector"},
    {"name": "XLU",      "ticker": "XLU",      "market": "US",   "label": "Utilities Sector"},
    {"name": "GLD",      "ticker": "GLD",      "market": "US",   "label": "Gold ETF"},
    {"name": "TLT",      "ticker": "TLT",      "market": "US",   "label": "20+ Year Treasury"},
    {"name": "EEM",      "ticker": "EEM",      "market": "US",   "label": "Emerging Markets"},
    {"name": "EWJ",      "ticker": "EWJ",      "market": "US",   "label": "Japan ETF"},
    {"name": "EWZ",      "ticker": "EWZ",      "market": "US",   "label": "Brazil ETF"},
    {"name": "SABIC",    "ticker": "2010.SR",  "market": "TASI", "label": "SABIC (Chemicals)"},
    {"name": "Maaden",   "ticker": "1211.SR",  "market": "TASI", "label": "Saudi Mining"},
]


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
    weekly = weekly[weekly["Volume"] > 0]
    weekly.index = weekly.index.tz_localize(None)
    return weekly


def hurst_exponent(ts, max_lag=100):
    """Estimate Hurst exponent via rescaled range analysis."""
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


def compute_characteristics(df, label=""):
    """
    Compute intrinsic asset characteristics from OHLCV weekly data.
    All characteristics are computed on the full available period
    (which becomes the training period in walk-forward).
    """
    close = df["Close"]
    returns = close.pct_change().dropna() * 100
    log_ret = np.log(close / close.shift(1)).dropna()

    n_weeks = len(returns)
    ann_factor = np.sqrt(52)

    chars = {}
    chars["n_weeks"] = n_weeks

    # 1. Volatility
    chars["ann_volatility"] = round(returns.std() * ann_factor, 2)

    # 2. Mean weekly return
    chars["mean_weekly_ret"] = round(returns.mean(), 4)

    # 3. Annualized return
    total_ret = close.iloc[-1] / close.iloc[0]
    n_years = n_weeks / 52
    chars["ann_return"] = round((total_ret ** (1 / n_years) - 1) * 100, 2) if n_years > 0 else 0

    # 4. Sharpe (simple)
    chars["raw_sharpe"] = round(returns.mean() / returns.std() * ann_factor, 3) if returns.std() > 0 else 0

    # 5. Skewness
    chars["skewness"] = round(returns.skew(), 3)

    # 6. Kurtosis (excess)
    chars["kurtosis"] = round(returns.kurtosis(), 3)

    # 7. Return autocorrelation (lag-1)
    chars["autocorr_1"] = round(returns.autocorr(lag=1), 4) if len(returns) > 10 else 0

    # 8. Return autocorrelation (lag-4)
    chars["autocorr_4"] = round(returns.autocorr(lag=4), 4) if len(returns) > 10 else 0

    # 9. Hurst exponent
    chars["hurst"] = round(hurst_exponent(returns), 3)

    # 10. Max drawdown
    cum = (1 + returns / 100).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    chars["max_drawdown"] = round(dd.min() * 100, 2)

    # 11. Up-week ratio
    chars["up_week_ratio"] = round((returns > 0).mean(), 3)

    # 12. Mean absolute return (measure of activity)
    chars["mean_abs_return"] = round(returns.abs().mean(), 4)

    # 13. Tail ratio (95th percentile gain / abs 5th percentile loss)
    p95 = np.percentile(returns, 95)
    p5 = abs(np.percentile(returns, 5))
    chars["tail_ratio"] = round(p95 / p5, 3) if p5 > 0 else 1.0

    # 14. Downside volatility
    neg_ret = returns[returns < 0]
    chars["downside_vol"] = round(neg_ret.std() * ann_factor, 2) if len(neg_ret) > 5 else 0

    # 15. Regime change frequency (using simple bull/bear based on 20-week SMA)
    sma20 = close.rolling(20).mean()
    above = (close > sma20).astype(int)
    regime_changes = above.diff().abs().sum()
    chars["regime_changes_per_year"] = round(regime_changes / n_years, 2) if n_years > 0 else 0

    # 16. Trend strength: fraction of time price is above 50-week SMA
    sma50 = close.rolling(50).mean().dropna()
    close_aligned = close.loc[sma50.index]
    chars["pct_above_sma50"] = round((close_aligned > sma50).mean(), 3)

    # 17. Average true range (as % of close)
    tr = pd.DataFrame({
        "hl": df["High"] - df["Low"],
        "hc": abs(df["High"] - df["Close"].shift(1)),
        "lc": abs(df["Low"] - df["Close"].shift(1)),
    }).max(axis=1)
    atr_pct = (tr / close * 100).mean()
    chars["avg_true_range_pct"] = round(atr_pct, 3)

    # 18. Volume stability (coefficient of variation of volume)
    vol_cv = df["Volume"].std() / df["Volume"].mean() if df["Volume"].mean() > 0 else 0
    chars["volume_cv"] = round(vol_cv, 3)

    # 19. Return mean-reversion strength (negative autocorr = more MR)
    chars["mean_reversion_score"] = round(-chars["autocorr_1"], 4)

    # 20. Calmar-like ratio (ann return / max dd)
    chars["calmar_ratio"] = round(chars["ann_return"] / abs(chars["max_drawdown"]), 3) if chars["max_drawdown"] != 0 else 0

    return chars


def load_existing_results():
    """Load results from previously run assets."""
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


def run_new_asset(asset_info):
    """Run the multi-asset pipeline for a single new asset."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from multi_asset_backtest import run_asset
    return run_asset(asset_info)


def find_best_strategy(strats):
    """Find best strategy by Sharpe, and also by return."""
    best_sharpe_name = max(strats, key=lambda k: strats[k]["sharpe"])
    best_return_name = max(strats, key=lambda k: strats[k]["total_return"])
    return best_sharpe_name, best_return_name


def print_section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def main():
    existing = load_existing_results()

    # Determine which assets need running
    new_assets = [a for a in ALL_ASSETS if a["name"] not in existing]
    print(f"Existing results: {list(existing.keys())}")
    print(f"Need to run: {[a['name'] for a in new_assets]}")

    # Run new assets
    for asset in new_assets:
        print_section(f"Running pipeline: {asset['label']}")
        try:
            result = run_new_asset(asset)
            if result:
                existing[asset["name"]] = result["strategies"]
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Compute characteristics for all assets
    print_section("COMPUTING ASSET CHARACTERISTICS")
    all_chars = {}
    all_strats = {}

    for asset in ALL_ASSETS:
        name = asset["name"]
        if name not in existing:
            print(f"  {name}: no results, skipping")
            continue

        print(f"  {name}: fetching data for characteristics...")
        df = fetch_weekly(asset["ticker"], asset["market"])
        if df is None or len(df) < 200:
            print(f"    insufficient data, skipping")
            continue

        chars = compute_characteristics(df, name)
        strats = existing[name]
        best_sharpe, best_return = find_best_strategy(strats)

        chars["best_strategy_sharpe"] = best_sharpe
        chars["best_strategy_return"] = best_return
        chars["best_sharpe_value"] = strats[best_sharpe]["sharpe"]
        chars["bnh_sharpe"] = strats[best_sharpe]["bnh_sharpe"]
        chars["sharpe_excess"] = round(strats[best_sharpe]["sharpe"] - strats[best_sharpe]["bnh_sharpe"], 3)
        chars["market"] = asset["market"]
        chars["label"] = asset["label"]

        for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
            m = strats.get(sn, {})
            if m:
                chars[f"sharpe_{sn}"] = m["sharpe"]
                chars[f"return_{sn}"] = m["total_return"]
                chars[f"maxdd_{sn}"] = m["max_drawdown"]

        all_chars[name] = chars
        all_strats[name] = strats

        print(f"    vol={chars['ann_volatility']:.1f}%  hurst={chars['hurst']:.3f}  "
              f"autocorr1={chars['autocorr_1']:.4f}  regime_chg/yr={chars['regime_changes_per_year']:.1f}  "
              f"best={best_sharpe}")

    # Build characteristics DataFrame
    chars_df = pd.DataFrame(all_chars).T
    chars_df.index.name = "asset"

    print_section("ASSET CHARACTERISTICS SUMMARY")
    display_cols = ["label", "ann_volatility", "hurst", "autocorr_1", "skewness", "kurtosis",
                    "regime_changes_per_year", "pct_above_sma50", "avg_true_range_pct",
                    "up_week_ratio", "max_drawdown", "best_strategy_sharpe"]
    print(chars_df[display_cols].to_string())

    # Analyze: which characteristics predict strategy choice
    print_section("STRATEGY SELECTION ANALYSIS")

    strat_groups = {}
    for name, ch in all_chars.items():
        bs = ch["best_strategy_sharpe"]
        strat_groups.setdefault(bs, []).append(name)

    feature_cols = ["ann_volatility", "hurst", "autocorr_1", "autocorr_4", "skewness", "kurtosis",
                    "regime_changes_per_year", "pct_above_sma50", "avg_true_range_pct",
                    "up_week_ratio", "max_drawdown", "mean_weekly_ret", "downside_vol",
                    "tail_ratio", "volume_cv", "mean_reversion_score", "calmar_ratio",
                    "mean_abs_return"]

    print("\n  Mean characteristics by best strategy:\n")
    for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
        assets_in = strat_groups.get(sn, [])
        if not assets_in:
            continue
        sub = chars_df.loc[assets_in, feature_cols].astype(float)
        print(f"  --- {sn} ({len(assets_in)} assets: {', '.join(assets_in)}) ---")
        for col in feature_cols:
            mean_val = sub[col].mean()
            print(f"    {col:30s}: {mean_val:8.3f}")
        print()

    # Compute discriminating features
    print_section("DISCRIMINATING FEATURES")
    print("  Features with largest difference between strategy groups:\n")

    for col in feature_cols:
        vals_by_strat = {}
        for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
            assets_in = strat_groups.get(sn, [])
            if assets_in:
                vals_by_strat[sn] = chars_df.loc[assets_in, col].astype(float).mean()
        if len(vals_by_strat) >= 2:
            max_v = max(vals_by_strat.values())
            min_v = min(vals_by_strat.values())
            spread = max_v - min_v
            max_strat = [k for k, v in vals_by_strat.items() if v == max_v][0]
            min_strat = [k for k, v in vals_by_strat.items() if v == min_v][0]
            print(f"  {col:30s}: spread={spread:8.3f}  "
                  f"highest={max_strat:18s}({max_v:+8.3f})  "
                  f"lowest={min_strat:18s}({min_v:+8.3f})")

    # Build decision rules
    print_section("DECISION RULES (from data patterns)")

    rules = []
    n_assets = len(all_chars)

    # Rule 1: volatility-based
    vol_threshold_high = chars_df["ann_volatility"].astype(float).quantile(0.75)
    vol_threshold_low = chars_df["ann_volatility"].astype(float).quantile(0.25)

    # Rule 2: regime change frequency
    rc_median = chars_df["regime_changes_per_year"].astype(float).median()

    # Rule 3: hurst exponent
    hurst_median = chars_df["hurst"].astype(float).median()

    # Rule 4: autocorrelation
    ac_median = chars_df["autocorr_1"].astype(float).median()

    # Rule 5: up-week ratio
    uw_median = chars_df["up_week_ratio"].astype(float).median()

    # Build a simple rule-based decision tree from patterns
    def predict_strategy(ch):
        vol = ch["ann_volatility"]
        hurst = ch["hurst"]
        ac1 = ch["autocorr_1"]
        rc = ch["regime_changes_per_year"]
        uw = ch["up_week_ratio"]
        atr = ch["avg_true_range_pct"]
        mr = ch["mean_reversion_score"]
        dd = abs(ch["max_drawdown"])

        if vol < vol_threshold_low and dd < 35:
            return "Scaled"
        if rc > rc_median and ac1 < ac_median:
            return "Adaptive"
        if hurst > 0.55 and uw > 0.53:
            return "Binary (t=opt)"
        return "Binary (t=0)"

    # Score the rule-based classifier
    correct = 0
    predictions = {}
    for name, ch in all_chars.items():
        pred = predict_strategy(ch)
        actual = ch["best_strategy_sharpe"]
        predictions[name] = {"predicted": pred, "actual": actual, "correct": pred == actual}
        if pred == actual:
            correct += 1

    accuracy = correct / n_assets * 100
    print(f"\n  Rule-based accuracy: {correct}/{n_assets} = {accuracy:.1f}%")
    print()
    for name in sorted(predictions):
        p = predictions[name]
        mark = "OK" if p["correct"] else "MISS"
        print(f"    {name:12s}: predicted={p['predicted']:18s}  actual={p['actual']:18s}  [{mark}]")

    # Refine rules using a simple decision tree on all the data
    print_section("REFINED DECISION FRAMEWORK")

    # For each strategy, find the characteristic ranges
    for sn in ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]:
        assets_in = strat_groups.get(sn, [])
        if not assets_in:
            continue
        sub = chars_df.loc[assets_in, feature_cols].astype(float)
        rest = chars_df.loc[~chars_df.index.isin(assets_in), feature_cols].astype(float)

        print(f"\n  {sn} assets ({', '.join(assets_in)}):")
        # Find features where this group differs most from others
        diffs = []
        for col in feature_cols:
            if len(rest) == 0:
                continue
            g_mean = sub[col].mean()
            r_mean = rest[col].mean()
            pooled_std = chars_df[col].astype(float).std()
            if pooled_std > 0:
                effect_size = abs(g_mean - r_mean) / pooled_std
                diffs.append((col, effect_size, g_mean, r_mean))
        diffs.sort(key=lambda x: x[1], reverse=True)
        for col, es, gm, rm in diffs[:5]:
            direction = "higher" if gm > rm else "lower"
            print(f"    {col:30s}: {direction:6s} ({gm:+8.3f} vs {rm:+8.3f}, effect={es:.2f})")

    # Generate no-look-ahead strategy recommendation
    print_section("NO-LOOK-AHEAD STRATEGY RECOMMENDATION FRAMEWORK")
    print("""
  To select a strategy for a new asset without look-ahead bias:

  1. Compute the following characteristics on available HISTORICAL data only:
     - Annualized volatility (weekly returns std * sqrt(52))
     - Hurst exponent (rescaled range analysis)
     - Return autocorrelation at lag-1
     - Regime change frequency (crossings of 20-week SMA per year)
     - Up-week ratio (fraction of weeks with positive returns)
     - Max drawdown
     - Average True Range as % of close

  2. Apply decision rules (derived from cross-asset study):

     IF volatility < {vol_low:.1f}% AND max_drawdown > -{dd_low:.0f}%:
         -> Use SCALED strategy (conservative, low-DD)

     ELIF regime_changes/year > {rc:.1f} AND autocorr_lag1 < {ac:.4f}:
         -> Use ADAPTIVE strategy (regime-shifting assets)

     ELIF hurst > 0.55 AND up_week_ratio > 0.53:
         -> Use BINARY (t=opt) (trending assets with consistent up-bias)

     ELSE:
         -> Use BINARY (t=0) (default for high-vol, cyclical assets)

  3. Validate with a rolling 52-week backtest before deploying.
""".format(
        vol_low=vol_threshold_low,
        dd_low=35,
        rc=rc_median,
        ac=ac_median,
    ))

    # Save everything
    chars_df.to_csv(STUDY_DIR / "asset_characteristics.csv", float_format="%.4f")

    study_summary = {
        "n_assets": n_assets,
        "assets": list(all_chars.keys()),
        "strategy_groups": {k: v for k, v in strat_groups.items()},
        "rule_accuracy_pct": round(accuracy, 1),
        "predictions": predictions,
        "thresholds": {
            "vol_low": round(float(vol_threshold_low), 2),
            "vol_high": round(float(vol_threshold_high), 2),
            "regime_change_median": round(float(rc_median), 2),
            "hurst_median": round(float(hurst_median), 3),
            "autocorr_median": round(float(ac_median), 4),
            "up_week_median": round(float(uw_median), 3),
        },
        "characteristics": all_chars,
    }
    with open(STUDY_DIR / "study_summary.json", "w") as f:
        json.dump(study_summary, f, indent=2, default=str)

    # Charts
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle("Asset Characteristics by Best Strategy", color="white", fontsize=16, y=0.98)

    strat_colors = {
        "Binary (t=0)": "#26a69a", "Binary (t=opt)": "#ab47bc",
        "Scaled": "#42a5f5", "Adaptive": "#ffa726"
    }

    plot_features = [
        ("ann_volatility", "Annualized Volatility (%)"),
        ("hurst", "Hurst Exponent"),
        ("autocorr_1", "Autocorrelation (Lag-1)"),
        ("regime_changes_per_year", "Regime Changes / Year"),
        ("up_week_ratio", "Up-Week Ratio"),
        ("max_drawdown", "Max Drawdown (%)"),
    ]

    for idx, (feat, title) in enumerate(plot_features):
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#1e1e1e")

        for sn, color in strat_colors.items():
            assets_in = strat_groups.get(sn, [])
            if not assets_in:
                continue
            vals = [float(all_chars[a][feat]) for a in assets_in if a in all_chars]
            labels = [a for a in assets_in if a in all_chars]
            ax.scatter(range(len(vals)), vals, color=color, s=80, label=sn, alpha=0.9, zorder=5)
            for i, lbl in enumerate(labels):
                ax.annotate(lbl, (i, vals[i]), fontsize=6, color="#cccccc",
                           ha="center", va="bottom", rotation=30)

        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="#cccccc")
        ax.set_xticks([])
        ax.grid(color="#333333", alpha=0.5)
        for s in ax.spines.values():
            s.set_color("#444444")
        if idx == 0:
            ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=7)

    plt.tight_layout()
    fig.savefig(STUDY_DIR / "characteristics_by_strategy.png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # Strategy selection scatter: volatility vs regime changes
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    fig2.patch.set_facecolor("#1e1e1e")
    ax2.set_facecolor("#1e1e1e")
    for sn, color in strat_colors.items():
        assets_in = strat_groups.get(sn, [])
        if not assets_in:
            continue
        vols = [float(all_chars[a]["ann_volatility"]) for a in assets_in]
        rcs = [float(all_chars[a]["regime_changes_per_year"]) for a in assets_in]
        ax2.scatter(vols, rcs, color=color, s=120, label=sn, alpha=0.9, edgecolors="white", linewidth=0.5)
        for i, a in enumerate(assets_in):
            ax2.annotate(a, (vols[i], rcs[i]), fontsize=8, color="#cccccc",
                        ha="center", va="bottom")

    ax2.set_xlabel("Annualized Volatility (%)", color="#cccccc", fontsize=12)
    ax2.set_ylabel("Regime Changes per Year", color="#cccccc", fontsize=12)
    ax2.set_title("Strategy Selection Map: Volatility vs Regime Frequency", color="white", fontsize=14)
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=10)
    ax2.tick_params(colors="#cccccc")
    ax2.grid(color="#333333", alpha=0.5)
    for s in ax2.spines.values():
        s.set_color("#444444")
    plt.tight_layout()
    fig2.savefig(STUDY_DIR / "strategy_selection_map.png", dpi=150,
                 bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()

    # All strategies heatmap
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    fig3.patch.set_facecolor("#1e1e1e")
    ax3.set_facecolor("#1e1e1e")
    asset_names = sorted(all_chars.keys())
    strat_list = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
    sharpe_matrix = []
    for asset in asset_names:
        row = []
        for sn in strat_list:
            key = f"sharpe_{sn}"
            val = float(all_chars[asset].get(key, 0))
            row.append(val)
        sharpe_matrix.append(row)
    sharpe_arr = np.array(sharpe_matrix)
    im = ax3.imshow(sharpe_arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1.5)
    ax3.set_xticks(range(len(strat_list)))
    ax3.set_xticklabels(strat_list, color="#cccccc", fontsize=10)
    ax3.set_yticks(range(len(asset_names)))
    ax3.set_yticklabels(asset_names, color="#cccccc", fontsize=9)
    for i in range(len(asset_names)):
        best_idx = np.argmax(sharpe_arr[i])
        for j in range(len(strat_list)):
            txt = f"{sharpe_arr[i, j]:.3f}"
            weight = "bold" if j == best_idx else "normal"
            color = "black" if sharpe_arr[i, j] > 0.7 else "white"
            ax3.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight=weight, color=color)
    ax3.set_title("Sharpe Ratio Heatmap: All Assets x All Strategies", color="white", fontsize=14)
    cbar = fig3.colorbar(im, ax=ax3, shrink=0.8)
    cbar.ax.tick_params(colors="#cccccc")
    cbar.set_label("Sharpe Ratio", color="#cccccc")
    plt.tight_layout()
    fig3.savefig(STUDY_DIR / "sharpe_heatmap.png", dpi=150,
                 bbox_inches="tight", facecolor=fig3.get_facecolor())
    plt.close()

    print(f"\nAll study outputs saved to {STUDY_DIR}/")
    print("  - asset_characteristics.csv")
    print("  - study_summary.json")
    print("  - characteristics_by_strategy.png")
    print("  - strategy_selection_map.png")
    print("  - sharpe_heatmap.png")


if __name__ == "__main__":
    main()
