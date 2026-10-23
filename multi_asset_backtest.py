"""
Multi-asset backtest: apply the same model/strategy pipeline to multiple instruments.

Instruments:
  - US Sector ETFs: XLK (Tech), XLF (Financials), XLE (Energy)
  - TASI Stocks: SADAFCO (2270.SR), Al Rajhi (1120.SR), STC (7010.SR), Mobily (7020.SR)

Each instrument runs the full pipeline:
  Data fetch -> Technical indicators -> HMM regime -> Feature engineering ->
  Walk-forward ensemble -> Strategy backtest -> Metrics

Results are stored per-asset under results/<ticker_slug>/ and a cross-asset
comparison summary is produced.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
import time
warnings.filterwarnings("ignore")

from technical_indicators import add_indicators
from hmm_regime import fit_hmm_walk_forward

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COST_PER_TRADE = 0.0002  # 0.02% round-trip

ASSETS = [
    {"name": "XLK",     "ticker": "XLK",     "market": "US",   "label": "Technology Select Sector"},
    {"name": "XLF",     "ticker": "XLF",     "market": "US",   "label": "Financial Select Sector"},
    {"name": "XLE",     "ticker": "XLE",     "market": "US",   "label": "Energy Select Sector"},
    {"name": "SADAFCO", "ticker": "2270.SR", "market": "TASI", "label": "SADAFCO (2270)"},
    {"name": "Al_Rajhi","ticker": "1120.SR", "market": "TASI", "label": "Al Rajhi Bank (1120)"},
    {"name": "STC",     "ticker": "7010.SR", "market": "TASI", "label": "Saudi Telecom (7010)"},
    {"name": "Mobily",  "ticker": "7020.SR", "market": "TASI", "label": "Mobily (7020)"},
]

MA_PERIODS = [5, 10, 20, 50, 100, 200]

LAG_COLS = [
    "Close_PctChg", "Volume_Log_Chg", "RSI_14", "MACD_Hist", "BB_PctB",
    "ADX_14", "DI_Plus_14", "DI_Minus_14",
    "HMM_Regime", "HMM_Prob_Bear", "HMM_Prob_Bull",
]
LAG_PERIODS = [1, 2, 3, 4]

INTERACTIONS = [
    ("RSI_14", "MACD_Hist"),
    ("BB_PctB", "Volume_Log_Chg"),
    ("ADX_14", "DI_Plus_14"),
    ("ADX_14", "DI_Minus_14"),
    ("Close_PctChg", "Volume_Log_Chg"),
    ("RSI_14", "BB_PctB"),
    ("HMM_Prob_Bull", "Close_PctChg"),
    ("HMM_Prob_Bear", "Close_PctChg"),
    ("HMM_Regime", "RSI_14"),
    ("Close_PctChg", "Close_PctChg_Lag1"),
]


def print_section(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_acc = ((y_true > 0) == (y_pred > 0)).mean() * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4),
            "r2": round(r2, 4), "dir_accuracy": round(dir_acc, 2)}


# =========================================================================
# DATA PIPELINE
# =========================================================================
def fetch_weekly_data(ticker, market, start="2000-01-01"):
    """Fetch daily data from yfinance and resample to weekly."""
    tk = yf.Ticker(ticker)
    daily = tk.history(start=start, auto_adjust=False)
    if daily.empty:
        raise ValueError(f"No data for {ticker}")

    resample_rule = "W-THU" if market == "TASI" else "W-FRI"

    weekly = pd.DataFrame()
    weekly["Open"] = daily["Open"].resample(resample_rule).first()
    weekly["High"] = daily["High"].resample(resample_rule).max()
    weekly["Low"] = daily["Low"].resample(resample_rule).min()
    weekly["Close"] = daily["Close"].resample(resample_rule).last()
    weekly["Volume"] = daily["Volume"].resample(resample_rule).sum()

    if "Adj Close" in daily.columns:
        adj_close = daily["Adj Close"].resample(resample_rule).last()
        a2p = adj_close / weekly["Close"]
        weekly["Open"] = weekly["Open"] * a2p
        weekly["High"] = weekly["High"] * a2p
        weekly["Low"] = weekly["Low"] * a2p
        weekly["Close"] = adj_close

    weekly.dropna(subset=["Close"], inplace=True)
    weekly = weekly[weekly["Volume"] > 0]
    weekly.index = weekly.index.tz_localize(None)
    weekly.index.name = "Week_Ending"
    return weekly


def fetch_macro_for_asset(df, market):
    """Fetch VIX and global macro data, aligned to asset's weekly index."""
    macro = {}

    vix_tk = yf.Ticker("^VIX")
    vix_daily = vix_tk.history(start="2000-01-01")
    if not vix_daily.empty:
        resample_rule = "W-THU" if market == "TASI" else "W-FRI"
        vix_weekly = vix_daily["Close"].resample(resample_rule).last().dropna()
        vix_weekly.index = vix_weekly.index.tz_localize(None)
        macro["VIX_Close"] = vix_weekly

    extras = {"^TNX": "TNX", "DX-Y.NYB": "DXY", "GC=F": "Gold", "BZ=F": "Brent"}
    for tick, name in extras.items():
        try:
            tk = yf.Ticker(tick)
            d = tk.history(start="2000-01-01")
            if d.empty:
                continue
            resample_rule = "W-THU" if market == "TASI" else "W-FRI"
            w = d["Close"].resample(resample_rule).last().dropna()
            w.index = w.index.tz_localize(None)
            macro[f"{name}_Close"] = w
        except Exception:
            pass

    if macro:
        macro_df = pd.DataFrame(macro)
        macro_df.index = macro_df.index.normalize()
        df.index = pd.to_datetime(df.index).normalize()
        df = df[~df.index.duplicated(keep="first")]
        df = df.join(macro_df, how="left")
        df = df[~df.index.duplicated(keep="first")]
        for col in macro_df.columns:
            df[col] = df[col].ffill().fillna(0)

    return df


def build_features(df):
    """Build the full feature set for one asset."""

    df["Volume_Log"] = np.log1p(df["Volume"].clip(lower=0))
    base_price_cols = ["Open", "Low", "High", "Close"]
    for col in base_price_cols:
        df[f"{col}_PctChg"] = df[col].pct_change() * 100
    df["Volume_Log_Chg"] = df["Volume_Log"].diff()

    df = df.replace([np.inf, -np.inf], np.nan)

    df = add_indicators(df, prefix="", close_col="Close", high_col="High",
                        low_col="Low", volume_col="Volume")

    if "VIX_Close" in df.columns:
        df["VIX_PctChg"] = df["VIX_Close"].pct_change() * 100

    for mcol in ["TNX_Close", "DXY_Close", "Gold_Close", "Brent_Close"]:
        if mcol in df.columns:
            df[f"{mcol.replace('_Close','')}_PctChg"] = df[mcol].pct_change() * 100

    # Regime features (rule-based)
    if "SMA_200" in df.columns:
        df["Above_SMA_200"] = (df["Close"] > df["SMA_200"]).astype(int)
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    if "VIX_Close" in df.columns:
        df["VIX_High_Regime"] = (df["VIX_Close"] > 25).astype(int)
    if "DI_Plus_14" in df.columns and "DI_Minus_14" in df.columns:
        df["Trend_Bullish"] = (df["DI_Plus_14"] > df["DI_Minus_14"]).astype(int)

    # HMM regime
    if "Close_PctChg" in df.columns and "VIX_Close" in df.columns:
        hmm_feats = fit_hmm_walk_forward(df, return_col="Close_PctChg", vix_col="VIX_Close")
        df = pd.concat([df, hmm_feats], axis=1)
    elif "Close_PctChg" in df.columns:
        vol_proxy = df["Close_PctChg"].rolling(12).std()
        vol_proxy.name = "Vol_Proxy"
        temp = pd.concat([df, vol_proxy], axis=1)
        hmm_feats = fit_hmm_walk_forward(temp, return_col="Close_PctChg", vix_col="Vol_Proxy")
        df = pd.concat([df, hmm_feats], axis=1)

    # Time features
    idx = pd.to_datetime(df.index)
    df["Month"] = idx.month
    df["Quarter"] = idx.quarter
    df["WeekOfYear"] = idx.isocalendar().week.values.astype(int)
    df["SellInMay"] = ((idx.month >= 5) & (idx.month <= 10)).astype(int)

    # Target
    df["Next_Week_Return"] = df["Close"].pct_change(1).shift(-1) * 100
    df.dropna(subset=["Next_Week_Return"], inplace=True)

    # Lags
    for col in LAG_COLS:
        if col not in df.columns:
            continue
        for lag in LAG_PERIODS:
            df[f"{col}_Lag{lag}"] = df[col].shift(lag)

    # Rolling stats
    if "Close_PctChg" in df.columns:
        for window in [4, 8, 12]:
            df[f"Close_PctChg_RollMean_{window}"] = df["Close_PctChg"].rolling(window).mean()
            df[f"Close_PctChg_RollStd_{window}"] = df["Close_PctChg"].rolling(window).std()

    # Interactions
    for a, b in INTERACTIONS:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]

    return df


def impute_nans(X):
    """Fill NaNs and infinities in feature matrix."""
    X = X.replace([np.inf, -np.inf], np.nan)

    brent_cols = [c for c in X.columns if "Brent" in c]
    X[brent_cols] = X[brent_cols].fillna(0)

    indicator_cols = [c for c in X.columns if any(x in c for x in
                      ["SMA_", "EMA_", "BB_", "MACD", "RSI_", "ADX_", "DI_Plus",
                       "DI_Minus", "OBV", "HMM_"])]
    X[indicator_cols] = X[indicator_cols].ffill()

    lag_cols = [c for c in X.columns if "_Lag" in c]
    X[lag_cols] = X[lag_cols].ffill()

    roll_cols = [c for c in X.columns if "Roll" in c]
    X[roll_cols] = X[roll_cols].ffill()

    interact_cols = [c for c in X.columns if "_x_" in c]
    X[interact_cols] = X[interact_cols].fillna(0)

    X.ffill(inplace=True)
    X.bfill(inplace=True)

    X = X.replace([np.inf, -np.inf], 0)
    return X


def select_features(X_train, y_train, X_val, y_val, all_features, top_n=40):
    """Use XGBoost importance to select top features."""
    top_n = min(top_n, len(all_features))
    selector = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="rmse", early_stopping_rounds=20, random_state=42,
    )
    selector.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    importance = selector.get_booster().get_score(importance_type="gain")
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    selected = [f for f, _ in imp_sorted[:top_n]]
    available = [f for f in selected if f in all_features]
    if len(available) < 5:
        available = all_features[:min(top_n, len(all_features))]
    print(f"    Selected {len(available)} features (from {len(all_features)})")
    return available


# =========================================================================
# WALK-FORWARD PREDICTION
# =========================================================================
def walk_forward_predict(X, y, feature_cols, min_train_weeks=260, step_weeks=26):
    """Walk-forward OOS predictions using expanding window."""
    n = len(X)
    predictions = pd.Series(dtype=float, name="prediction")
    scaler = StandardScaler()

    fold = 0
    start = min_train_weeks

    while start < n:
        end = min(start + step_weeks, n)
        train_idx = X.index[:start]
        test_idx = X.index[start:end]

        X_tr = X.loc[train_idx, feature_cols]
        y_tr = y.loc[train_idx]
        X_te = X.loc[test_idx, feature_cols]

        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=feature_cols)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=feature_cols)

        xgb_m = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            gamma=1, reg_alpha=1.0, reg_lambda=5.0,
            eval_metric="rmse", early_stopping_rounds=30, random_state=42,
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y.loc[test_idx])], verbose=0)

        lgb_m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1,
        )
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y.loc[test_idx])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        cat_m = CatBoostRegressor(
            iterations=500, depth=4, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=42, verbose=0,
        )
        cat_m.fit(X_tr, y_tr, eval_set=(X_te, y.loc[test_idx]),
                  early_stopping_rounds=30)

        ridge_m = Ridge(alpha=10.0)
        ridge_m.fit(X_tr_s, y_tr)

        p = (0.35 * xgb_m.predict(X_te) +
             0.30 * lgb_m.predict(X_te) +
             0.25 * cat_m.predict(X_te) +
             0.10 * ridge_m.predict(X_te_s))

        fold_preds = pd.Series(p, index=test_idx, name="prediction")
        predictions = pd.concat([predictions, fold_preds])

        fold += 1
        start = end

    print(f"    Walk-forward: {fold} folds, {len(predictions)} OOS predictions")
    return predictions


# =========================================================================
# STRATEGIES
# =========================================================================
def strategy_binary(predictions, threshold=0.0):
    return (predictions > threshold).astype(float)


def strategy_scaled(predictions, max_pred=1.0):
    pos = predictions / max_pred
    return pos.clip(0.0, 1.0)


def strategy_adaptive(predictions, actuals, lookback=26):
    positions = pd.Series(0.0, index=predictions.index)
    for i in range(len(predictions)):
        if i < lookback:
            positions.iloc[i] = 1.0 if predictions.iloc[i] > 0.0 else 0.0
            continue
        recent_preds = predictions.iloc[max(0, i - lookback):i]
        recent_rets = actuals.iloc[max(0, i - lookback):i]
        best_t, best_ret = 0.0, -np.inf
        for t in np.arange(-0.5, 0.5, 0.05):
            pos = (recent_preds > t).astype(float)
            trades = (pos.diff().abs().fillna(0) > 0).sum()
            ret = (pos * recent_rets / 100).sum() - trades * COST_PER_TRADE
            if ret > best_ret:
                best_t, best_ret = t, ret
        positions.iloc[i] = 1.0 if predictions.iloc[i] > best_t else 0.0
    return positions


def backtest(positions, actual_returns, label="Strategy"):
    """Simulate returns and compute all metrics."""
    weekly_ret = actual_returns / 100.0
    trades = positions.diff().abs().fillna(0)
    costs = trades * COST_PER_TRADE
    strat_ret = positions * weekly_ret - costs

    equity = (1 + strat_ret).cumprod()
    bnh_equity = (1 + weekly_ret).cumprod()

    n_weeks = len(equity)
    n_years = n_weeks / 52

    total_return = equity.iloc[-1] - 1
    bnh_total = bnh_equity.iloc[-1] - 1

    cagr = (equity.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0
    bnh_cagr = (bnh_equity.iloc[-1]) ** (1 / n_years) - 1 if n_years > 0 else 0

    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(52) if strat_ret.std() > 0 else 0
    bnh_sharpe = weekly_ret.mean() / weekly_ret.std() * np.sqrt(52) if weekly_ret.std() > 0 else 0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    bnh_rolling_max = bnh_equity.cummax()
    bnh_dd = (bnh_equity - bnh_rolling_max) / bnh_rolling_max
    bnh_max_dd = bnh_dd.min()

    in_market = (positions > 0).mean() * 100
    n_trades = int(trades.gt(0).sum())
    win_weeks = ((strat_ret > 0) & (positions > 0)).sum()
    total_active_weeks = (positions > 0).sum()
    win_rate = win_weeks / total_active_weeks * 100 if total_active_weeks > 0 else 0

    return {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 2),
        "in_market_pct": round(in_market, 1),
        "n_trades": n_trades,
        "n_weeks": n_weeks,
        "bnh_total_return": round(bnh_total * 100, 2),
        "bnh_cagr": round(bnh_cagr * 100, 2),
        "bnh_sharpe": round(bnh_sharpe, 3),
        "bnh_max_drawdown": round(bnh_max_dd * 100, 2),
    }, equity, bnh_equity


# =========================================================================
# PER-ASSET PIPELINE
# =========================================================================
def run_asset(asset_info):
    """Full pipeline for a single asset. Returns metrics dict or None on failure."""
    name = asset_info["name"]
    ticker = asset_info["ticker"]
    market = asset_info["market"]
    label = asset_info["label"]

    asset_dir = RESULTS_DIR / name
    asset_dir.mkdir(exist_ok=True)

    print_section(f"{label} ({ticker})")

    # 1. Fetch data
    print(f"  [1/6] Fetching weekly data ...")
    try:
        df = fetch_weekly_data(ticker, market)
    except Exception as e:
        print(f"  FAILED to fetch data: {e}")
        return None
    print(f"    {len(df)} weeks, {df.index.min().date()} to {df.index.max().date()}")

    # 2. Macro data
    print(f"  [2/6] Fetching macro data ...")
    df = fetch_macro_for_asset(df, market)

    # 3. Features
    print(f"  [3/6] Building features ...")
    df = build_features(df)

    exclude = ["Next_Week_Return", "OB_Bull", "OB_Bear", "OB_Type",
               "OB_High", "OB_Low", "OB_Mid", "Target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df["Next_Week_Return"].copy()

    X = impute_nans(X)
    print(f"    {len(X)} rows, {len(feature_cols)} features")

    # Check minimum data
    min_train = max(260, int(len(X) * 0.5))
    if len(X) < min_train + 52:
        print(f"  SKIPPED: only {len(X)} rows, need at least {min_train + 52}")
        return None

    # 4. Feature selection
    print(f"  [4/6] Feature selection ...")
    n = len(X)
    sel_end = int(n * 0.70)
    top_n = min(40, len(feature_cols))
    selected = select_features(
        X.iloc[:sel_end][feature_cols], y.iloc[:sel_end],
        X.iloc[sel_end:sel_end + 200][feature_cols], y.iloc[sel_end:sel_end + 200],
        feature_cols, top_n=top_n,
    )

    # 5. Walk-forward predictions
    print(f"  [5/6] Walk-forward predictions ...")
    min_wf = max(260, int(len(X) * 0.45))
    predictions = walk_forward_predict(X, y, selected, min_train_weeks=min_wf, step_weeks=26)

    aligned = pd.DataFrame({"prediction": predictions, "actual": y}).dropna()
    preds = aligned["prediction"]
    actuals = aligned["actual"]
    print(f"    Aligned: {len(aligned)} OOS predictions")
    print(f"    Period: {aligned.index.min().date()} to {aligned.index.max().date()}")

    if len(aligned) < 52:
        print(f"  SKIPPED: only {len(aligned)} OOS predictions, need >= 52")
        return None

    # Model metrics
    model_metrics = eval_metrics(actuals, preds)
    print(f"    DirAcc={model_metrics['dir_accuracy']:.2f}%  R2={model_metrics['r2']:.4f}")

    # Optimize threshold
    split = len(preds) // 2
    best_t, best_sharpe = 0.0, -np.inf
    for t in np.arange(-0.5, 0.5, 0.01):
        pos = (preds.iloc[:split] > t).astype(float)
        ret = pos * actuals.iloc[:split] / 100
        sr = ret.mean() / ret.std() * np.sqrt(52) if ret.std() > 0 else 0
        if sr > best_sharpe:
            best_t, best_sharpe = t, sr

    # 6. Backtest strategies
    print(f"  [6/6] Backtesting strategies ...")
    strategies = {
        "Binary (t=0)": strategy_binary(preds, threshold=0.0),
        "Binary (t=opt)": strategy_binary(preds, threshold=best_t),
        "Scaled": strategy_scaled(preds, max_pred=max(0.5, preds.quantile(0.9))),
        "Adaptive": strategy_adaptive(preds, actuals, lookback=26),
    }

    all_metrics = {}
    all_equities = {}
    bnh_eq = None

    for sname, positions in strategies.items():
        metrics, eq, bnh = backtest(positions, actuals, sname)
        all_metrics[sname] = metrics
        all_equities[sname] = eq
        bnh_eq = bnh

    # Print results
    print(f"\n    {'Strategy':20s}  {'Return':>8s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>8s}  {'WinRate':>8s}  {'InMkt':>6s}")
    print(f"    {'-'*20}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}")
    for sn, m in all_metrics.items():
        print(f"    {sn:20s}  {m['total_return']:7.1f}%  {m['cagr']:6.2f}%  {m['sharpe']:7.3f}  "
              f"{m['max_drawdown']:7.2f}%  {m['win_rate']:7.2f}%  {m['in_market_pct']:5.1f}%")
    bm = list(all_metrics.values())[0]
    print(f"    {'Buy & Hold':20s}  {bm['bnh_total_return']:7.1f}%  {bm['bnh_cagr']:6.2f}%  "
          f"{bm['bnh_sharpe']:7.3f}  {bm['bnh_max_drawdown']:7.2f}%")

    # Best strategy
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["sharpe"])
    print(f"\n    Best by Sharpe: {best_name} ({all_metrics[best_name]['sharpe']:.3f})")

    # HMM info
    hmm_selected = [f for f in selected if "HMM" in f]

    # Equity curve chart
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    colors = {"Binary (t=0)": "#26a69a", "Scaled": "#42a5f5",
              "Adaptive": "#ffa726", "Binary (t=opt)": "#ab47bc"}
    ax.plot(bnh_eq.index, bnh_eq.values, color="#666666", linewidth=2,
            linestyle="--", label="Buy & Hold", alpha=0.8)
    for sn, eq in all_equities.items():
        c = colors.get(sn, "#26a69a")
        ax.plot(eq.index, eq.values, color=c, linewidth=1.5, label=sn, alpha=0.9)
    ax.set_ylabel("Equity ($1 start)", color="#cccccc")
    ax.set_title(f"{label} - Equity Curves", color="white", fontsize=13)
    ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#cccccc")
    ax.grid(color="#333333", alpha=0.5)
    for s in ax.spines.values():
        s.set_color("#444444")
    plt.tight_layout()
    fig.savefig(asset_dir / "equity_curves.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # Save summary
    result = {
        "name": name,
        "label": label,
        "ticker": ticker,
        "market": market,
        "data_period": f"{df.index.min().date()} to {df.index.max().date()}",
        "total_weeks": len(df),
        "oos_weeks": len(aligned),
        "oos_period": f"{aligned.index.min().date()} to {aligned.index.max().date()}",
        "total_features": len(feature_cols),
        "selected_features": len(selected),
        "hmm_features_selected": hmm_selected,
        "model_metrics": model_metrics,
        "optimized_threshold": round(best_t, 2),
        "best_strategy": best_name,
        "strategies": all_metrics,
    }
    with open(asset_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# =========================================================================
# CROSS-ASSET COMPARISON
# =========================================================================
def generate_comparison(all_results):
    """Generate cross-asset comparison summary."""
    print_section("CROSS-ASSET COMPARISON")

    # Include SPY from its own results
    spy_strat = Path(__file__).parent / "strategy" / "backtest_summary.json"
    spy_model = Path(__file__).parent / "model" / "model_summary.json"
    if spy_strat.exists() and spy_model.exists():
        with open(spy_strat) as f:
            ss = json.load(f)
        with open(spy_model) as f:
            ms = json.load(f)
        hmm_sel = [ft for ft in ms.get("selected_feature_list", []) if "HMM" in ft]
        spy_result = {
            "name": "SPY",
            "label": "SPDR S&P 500 ETF",
            "ticker": "SPY",
            "market": "US",
            "oos_weeks": ss["oos_predictions"],
            "oos_period": ss["backtest_period"],
            "model_metrics": ms["walk_forward"]["aggregate"],
            "best_strategy": ss["best_strategy"],
            "strategies": ss["strategies"],
            "hmm_features_selected": hmm_sel,
            "selected_features": ms["selected_features"],
        }
        all_results = [spy_result] + all_results

    # Print comparison table
    print(f"\n  {'Asset':12s}  {'Market':6s}  {'OOS Wk':>6s}  {'DirAcc':>7s}  "
          f"{'Best Strat':15s}  {'Sharpe':>7s}  {'BnH Sharpe':>10s}  "
          f"{'Return':>8s}  {'BnH Ret':>8s}  {'MaxDD':>7s}  {'BnH DD':>7s}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*15}  {'-'*7}  {'-'*10}  "
          f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}")

    comparison_rows = []
    for r in all_results:
        best = r["best_strategy"]
        bm = r["strategies"][best]
        da = r["model_metrics"].get("dir_accuracy", 0)
        row = {
            "asset": r["name"],
            "market": r["market"],
            "label": r.get("label", r["name"]),
            "oos_weeks": r.get("oos_weeks", bm["n_weeks"]),
            "dir_accuracy": da,
            "best_strategy": best,
            "sharpe": bm["sharpe"],
            "bnh_sharpe": bm["bnh_sharpe"],
            "total_return": bm["total_return"],
            "bnh_return": bm["bnh_total_return"],
            "max_dd": bm["max_drawdown"],
            "bnh_max_dd": bm["bnh_max_drawdown"],
            "cagr": bm["cagr"],
            "bnh_cagr": bm["bnh_cagr"],
            "win_rate": bm["win_rate"],
            "in_market": bm["in_market_pct"],
            "hmm_features": len(r.get("hmm_features_selected", [])),
            "beats_bnh_sharpe": bm["sharpe"] > bm["bnh_sharpe"],
            "beats_bnh_return": bm["total_return"] > bm["bnh_total_return"],
        }
        comparison_rows.append(row)

        print(f"  {row['asset']:12s}  {row['market']:6s}  {row['oos_weeks']:6d}  "
              f"{row['dir_accuracy']:6.1f}%  {row['best_strategy']:15s}  "
              f"{row['sharpe']:7.3f}  {row['bnh_sharpe']:10.3f}  "
              f"{row['total_return']:7.1f}%  {row['bnh_return']:7.1f}%  "
              f"{row['max_dd']:6.1f}%  {row['bnh_max_dd']:6.1f}%")

    beats_sharpe = sum(1 for r in comparison_rows if r["beats_bnh_sharpe"])
    beats_return = sum(1 for r in comparison_rows if r["beats_bnh_return"])
    total = len(comparison_rows)

    print(f"\n  Summary: {beats_sharpe}/{total} assets beat buy-and-hold on Sharpe ratio")
    print(f"           {beats_return}/{total} assets beat buy-and-hold on total return")

    # Save comparison
    comp_summary = {
        "assets": comparison_rows,
        "beats_bnh_sharpe": f"{beats_sharpe}/{total}",
        "beats_bnh_return": f"{beats_return}/{total}",
        "all_strategies": {},
    }

    for r in all_results:
        comp_summary["all_strategies"][r["name"]] = r["strategies"]

    with open(RESULTS_DIR / "comparison_summary.json", "w") as f:
        json.dump(comp_summary, f, indent=2, default=str)

    # Comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#1e1e1e")
    names = [r["asset"] for r in comparison_rows]
    x = np.arange(len(names))
    w = 0.35

    for idx, (metric, title) in enumerate([
        ("sharpe", "Sharpe Ratio"), ("total_return", "Total Return (%)"), ("max_dd", "Max Drawdown (%)")
    ]):
        ax = axes[idx]
        ax.set_facecolor("#1e1e1e")
        strat_vals = [r[metric] for r in comparison_rows]
        bnh_key = "bnh_sharpe" if metric == "sharpe" else ("bnh_return" if metric == "total_return" else "bnh_max_dd")
        bnh_vals = [r[bnh_key] for r in comparison_rows]
        ax.bar(x - w / 2, bnh_vals, w, color="#666666", label="Buy & Hold", alpha=0.7)
        ax.bar(x + w / 2, strat_vals, w, color="#26a69a", label="Best Strategy", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color="#cccccc")
        ax.set_title(title, color="white", fontsize=12)
        ax.tick_params(colors="#cccccc")
        ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=8)
        ax.grid(axis="y", color="#333333", alpha=0.5)
        for s in ax.spines.values():
            s.set_color("#444444")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "comparison_chart.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Comparison chart saved to {RESULTS_DIR / 'comparison_chart.png'}")

    return comparison_rows


# =========================================================================
# MAIN
# =========================================================================
def main():
    all_results = []

    for asset in ASSETS:
        try:
            result = run_asset(asset)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR processing {asset['name']}: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        comparison = generate_comparison(all_results)

    print(f"\n\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
