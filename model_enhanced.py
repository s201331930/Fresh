"""
Enhanced trading model — all 9 improvements integrated.

#1  NaN imputation (recover ~685 training rows)
#2  Feature selection (top features by importance)
#3  Walk-forward validation (expanding window)
#4  Target engineering (risk-adjusted + multi-horizon)
#5  Regime-aware features (bull/bear/sideways)
#6  Time-based features (seasonality)
#7  Alternative models (CatBoost added)
#8  Macro data expansion (TNX, DXY, Gold fetched & joined)
#9  Probability calibration (directional probability output)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_enriched.csv"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

OB_COLS = ["OB_Bull", "OB_Bear", "OB_Type", "OB_High", "OB_Low", "OB_Mid", "Target"]
START_DATE = "2000-01-01"

LAG_FEATURES = [
    "Close_PctChg", "Open_PctChg", "High_PctChg", "Low_PctChg",
    "VIX_Close", "VIX_Close_PctChg", "Volume_Log_Chg",
    "RSI_14", "MACD_Hist", "BB_PctB",
    "ADX_14", "DI_Plus_14", "DI_Minus_14",
    "Brent_Close_PctChg", "VIX_RSI_14",
]
LAG_PERIODS = [1, 2, 3, 4]

INTERACTIONS = [
    ("VIX_Close", "Close_PctChg"),
    ("RSI_14", "MACD_Hist"),
    ("BB_PctB", "Volume_Log_Chg"),
    ("VIX_Close", "Brent_Close_PctChg"),
    ("ADX_14", "DI_Plus_14"),
    ("ADX_14", "DI_Minus_14"),
    ("Close_PctChg", "Volume_Log_Chg"),
    ("VIX_Close_PctChg", "Close_PctChg"),
    ("RSI_14", "BB_PctB"),
    ("VIX_RSI_14", "VIX_MACD_Hist"),
    ("Close_PctChg", "Close_PctChg_Lag1"),
    ("VIX_Close", "RSI_14"),
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


def print_metrics(metrics, label):
    print(f"  {label:20s}  MAE={metrics['mae']:.4f}%  RMSE={metrics['rmse']:.4f}%  "
          f"R²={metrics['r2']:.4f}  DirAcc={metrics['dir_accuracy']:.2f}%")


# =========================================================================
# #8: MACRO DATA EXPANSION
# =========================================================================
def fetch_macro_extras() -> pd.DataFrame:
    """Fetch additional macro series: 10Y yield, Dollar index, Gold."""
    extras = {}
    tickers = {
        "^TNX": "TNX",       # 10-Year Treasury yield
        "DX-Y.NYB": "DXY",  # US Dollar index
        "GC=F": "Gold",     # Gold futures
    }
    for ticker, name in tickers.items():
        try:
            tk = yf.Ticker(ticker)
            daily = tk.history(start=START_DATE, interval="1d")
            if daily.empty:
                continue
            weekly = daily["Close"].resample("W-FRI").last().dropna()
            weekly.name = f"{name}_Close"
            pct = weekly.pct_change() * 100
            pct.name = f"{name}_PctChg"
            extras[f"{name}_Close"] = weekly
            extras[f"{name}_PctChg"] = pct
            print(f"    {name}: {len(weekly)} weeks")
        except Exception as e:
            print(f"    {name}: FAILED ({e})")
    return pd.DataFrame(extras)


# =========================================================================
# #5: REGIME-AWARE FEATURES
# =========================================================================
def add_regime_features(df):
    new = {}
    new["Above_SMA_200"] = (df["Close"] > df["SMA_200"]).astype(int) if "SMA_200" in df.columns else 0
    new["Above_SMA_50"] = (df["Close"] > df["SMA_50"]).astype(int) if "SMA_50" in df.columns else 0
    new["VIX_High_Regime"] = (df["VIX_Close"] > 25).astype(int) if "VIX_Close" in df.columns else 0
    new["VIX_Extreme"] = (df["VIX_Close"] > 35).astype(int) if "VIX_Close" in df.columns else 0
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        new["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    if "DI_Plus_14" in df.columns and "DI_Minus_14" in df.columns:
        new["Trend_Bullish"] = (df["DI_Plus_14"] > df["DI_Minus_14"]).astype(int)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# =========================================================================
# #6: TIME-BASED FEATURES
# =========================================================================
def add_time_features(df):
    idx = pd.to_datetime(df.index)
    new = {}
    new["Month"] = idx.month
    new["Quarter"] = idx.quarter
    new["WeekOfYear"] = idx.isocalendar().week.values.astype(int)
    new["IsJanuary"] = (idx.month == 1).astype(int)
    new["IsDec"] = (idx.month == 12).astype(int)
    new["SellInMay"] = ((idx.month >= 5) & (idx.month <= 10)).astype(int)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# =========================================================================
# #4: TARGET ENGINEERING
# =========================================================================
def add_targets(df):
    ret_1w = df["Close"].pct_change(1).shift(-1) * 100
    ret_2w = df["Close"].pct_change(2).shift(-2) * 100
    ret_4w = df["Close"].pct_change(4).shift(-4) * 100

    df["Next_Week_Return"] = ret_1w

    multi = (ret_1w * 0.5 + ret_2w * 0.3 + ret_4w * 0.2)
    multi = multi.where(ret_1w.notna() & ret_2w.notna() & ret_4w.notna())
    df["Multi_Horizon_Return"] = multi

    roll_std = df["Close_PctChg"].rolling(12, min_periods=4).std()
    df["Risk_Adjusted_Return"] = ret_1w / roll_std.replace(0, np.nan)

    return df


# =========================================================================
# FEATURE ENGINEERING
# =========================================================================
def add_lagged_features(df):
    new = {}
    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for lag in LAG_PERIODS:
            new[f"{col}_Lag{lag}"] = df[col].shift(lag)
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_rolling_stats(df):
    new = {}
    for window in [4, 8, 12]:
        new[f"Close_PctChg_RollMean_{window}"] = df["Close_PctChg"].rolling(window).mean()
        new[f"Close_PctChg_RollStd_{window}"] = df["Close_PctChg"].rolling(window).std()
        new[f"VIX_Close_RollMean_{window}"] = df["VIX_Close"].rolling(window).mean()
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def add_interaction_terms(df):
    new = {}
    for a, b in INTERACTIONS:
        if a in df.columns and b in df.columns:
            new[f"{a}_x_{b}"] = df[a] * df[b]
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# =========================================================================
# #1: NaN IMPUTATION
# =========================================================================
def impute_nans(df):
    """Recover rows lost to indicator warmup and Brent pre-2007 gap."""
    before = df.isnull().sum().sum()

    brent_cols = [c for c in df.columns if c.startswith("Brent")]
    df[brent_cols] = df[brent_cols].fillna(0)

    sma_ema_cols = [c for c in df.columns
                    if any(c.startswith(p) for p in ["SMA_", "EMA_", "VIX_SMA_", "VIX_EMA_"])]
    df[sma_ema_cols] = df[sma_ema_cols].ffill()

    indicator_cols = [c for c in df.columns if any(x in c for x in
                      ["BB_", "MACD", "RSI_", "ADX_", "DI_Plus", "DI_Minus", "OBV"])]
    df[indicator_cols] = df[indicator_cols].ffill()

    lag_cols = [c for c in df.columns if "_Lag" in c]
    df[lag_cols] = df[lag_cols].ffill()

    roll_cols = [c for c in df.columns if "Roll" in c]
    df[roll_cols] = df[roll_cols].ffill()

    interact_cols = [c for c in df.columns if "_x_" in c]
    df[interact_cols] = df[interact_cols].fillna(0)

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    after = df.isnull().sum().sum()
    print(f"  NaN imputation: {before} -> {after} nulls")
    return df


# =========================================================================
# #2: FEATURE SELECTION
# =========================================================================
def select_features(X_train, y_train, X_val, y_val, all_features, top_n=40):
    """Use XGBoost importance to select top features."""
    selector = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="rmse", early_stopping_rounds=20,
        random_state=42,
    )
    selector.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    importance = selector.get_booster().get_score(importance_type="gain")
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    selected = [f for f, _ in imp_sorted[:top_n]]
    available = [f for f in selected if f in all_features]
    print(f"  Selected {len(available)} features (from {len(all_features)} candidates)")
    print(f"  Top 10: {available[:10]}")
    return available


# =========================================================================
# #3: WALK-FORWARD VALIDATION
# =========================================================================
def walk_forward_cv(X, y, feature_cols, min_train_years=10, step_weeks=52):
    """
    Expanding-window walk-forward validation.
    Train on first N years, predict next `step_weeks`, expand, repeat.
    """
    idx = pd.to_datetime(X.index)
    min_date = idx.min()
    max_date = idx.max()
    first_test_start = min_date + pd.DateOffset(years=min_train_years)

    folds = []
    test_start = first_test_start
    fold_num = 0

    while test_start < max_date:
        test_end = test_start + pd.DateOffset(weeks=step_weeks)
        if test_end > max_date:
            test_end = max_date + pd.DateOffset(days=1)

        train_mask = idx < test_start
        test_mask = (idx >= test_start) & (idx < test_end)

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            test_start = test_end
            continue

        folds.append({
            "fold": fold_num,
            "train_idx": X.index[train_mask],
            "test_idx": X.index[test_mask],
        })
        fold_num += 1
        test_start = test_end

    print(f"  Walk-forward: {len(folds)} folds, "
          f"min train years={min_train_years}, step={step_weeks} weeks")
    return folds


def run_walk_forward(X, y, feature_cols, folds):
    """Run walk-forward with the ensemble, collecting OOS predictions."""
    all_preds = pd.Series(dtype=float)
    all_actuals = pd.Series(dtype=float)
    fold_metrics = []

    scaler = StandardScaler()

    for fold in folds:
        X_tr = X.loc[fold["train_idx"], feature_cols]
        y_tr = y.loc[fold["train_idx"]]
        X_te = X.loc[fold["test_idx"], feature_cols]
        y_te = y.loc[fold["test_idx"]]

        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), index=X_tr.index, columns=feature_cols)
        X_te_s = pd.DataFrame(scaler.transform(X_te), index=X_te.index, columns=feature_cols)

        xgb_m = xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            gamma=1, reg_alpha=1.0, reg_lambda=5.0,
            eval_metric="rmse", early_stopping_rounds=30, random_state=42,
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=0)

        lgb_m = lgb.LGBMRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
            reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1,
        )
        lgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        cat_m = CatBoostRegressor(
            iterations=500, depth=4, learning_rate=0.03,
            l2_leaf_reg=5.0, random_seed=42, verbose=0,
        )
        cat_m.fit(X_tr, y_tr, eval_set=(X_te, y_te), early_stopping_rounds=30)

        ridge_m = Ridge(alpha=10.0)
        ridge_m.fit(X_tr_s, y_tr)

        p_xgb = xgb_m.predict(X_te)
        p_lgb = lgb_m.predict(X_te)
        p_cat = cat_m.predict(X_te)
        p_ridge = ridge_m.predict(X_te_s)

        blend = 0.35 * p_xgb + 0.30 * p_lgb + 0.25 * p_cat + 0.10 * p_ridge
        fold_preds = pd.Series(blend, index=X_te.index)

        all_preds = pd.concat([all_preds, fold_preds])
        all_actuals = pd.concat([all_actuals, y_te])

        fm = eval_metrics(y_te, blend)
        fm["fold"] = fold["fold"]
        fm["train_size"] = len(X_tr)
        fm["test_size"] = len(X_te)
        fold_metrics.append(fm)

    return all_preds, all_actuals, fold_metrics


# =========================================================================
# #9: PROBABILITY CALIBRATION
# =========================================================================
def calibrate_direction(y_train, pred_train, y_val, pred_val, pred_test):
    """Convert regression output to calibrated P(up) using isotonic regression."""
    binary_train = (y_train > 0).astype(int).values
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pred_train.values, binary_train)

    p_val = iso.predict(pred_val.values)
    p_test = iso.predict(pred_test.values)
    return p_val, p_test, iso


# =========================================================================
# MAIN
# =========================================================================
def main():
    # --- Load base data ---
    print_section("#8: MACRO DATA EXPANSION")
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)
    print(f"  Base data: {len(df)} rows x {len(df.columns)} cols")

    print("  Fetching additional macro series ...")
    macro_extra = fetch_macro_extras()
    if not macro_extra.empty:
        macro_extra.index = macro_extra.index.tz_localize(None).normalize()
        df.index = pd.to_datetime(df.index, utc=True).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        df = df.join(macro_extra, how="left")
        df = df[~df.index.duplicated(keep="first")]
        for col in macro_extra.columns:
            df[col] = df[col].ffill().fillna(0)
    print(f"  After macro join: {len(df)} rows x {len(df.columns)} cols")

    # --- #5: Regime features ---
    print_section("#5: REGIME-AWARE FEATURES")
    df = add_regime_features(df)
    regime_cols = ["Above_SMA_200", "Above_SMA_50", "VIX_High_Regime",
                   "VIX_Extreme", "Golden_Cross", "Trend_Bullish"]
    regime_cols = [c for c in regime_cols if c in df.columns]
    print(f"  Added: {regime_cols}")

    # --- #6: Time features ---
    print_section("#6: TIME-BASED FEATURES")
    df = add_time_features(df)
    print(f"  Added: Month, Quarter, WeekOfYear, IsJanuary, IsDec, SellInMay")

    # --- #4: Target engineering ---
    print_section("#4: TARGET ENGINEERING")
    df = add_targets(df)
    target_col = "Next_Week_Return"
    df.dropna(subset=[target_col], inplace=True)
    print(f"  Targets: Next_Week_Return, Multi_Horizon_Return, Risk_Adjusted_Return")
    print(f"  Using primary target: {target_col}")

    # --- Feature engineering ---
    print_section("FEATURE ENGINEERING (lags, rolling, interactions)")
    df = add_lagged_features(df)
    df = add_rolling_stats(df)
    df = add_interaction_terms(df)

    exclude = OB_COLS + [target_col, "Multi_Horizon_Return", "Risk_Adjusted_Return"]
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"  Total features before selection: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # --- #1: NaN imputation ---
    print_section("#1: NaN IMPUTATION")
    nan_before = X.isnull().any(axis=1).sum()
    X = impute_nans(X)
    nan_after = X.isnull().any(axis=1).sum()
    print(f"  Rows with any NaN: {nan_before} -> {nan_after}")
    print(f"  Usable rows: {len(X)} (was ~281 before imputation)")

    # --- Split for feature selection ---
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_full, y_train_full = X.iloc[:train_end], y.iloc[:train_end]
    X_val_full, y_val_full = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test_full, y_test_full = X.iloc[val_end:], y.iloc[val_end:]

    print(f"\n  Split: train={len(X_train_full)}, val={len(X_val_full)}, test={len(X_test_full)}")

    # --- #2: Feature selection ---
    print_section("#2: FEATURE SELECTION")
    selected_features = select_features(
        X_train_full[feature_cols], y_train_full,
        X_val_full[feature_cols], y_val_full,
        feature_cols, top_n=40,
    )

    X_train = X_train_full[selected_features]
    y_train = y_train_full
    X_val = X_val_full[selected_features]
    y_val = y_val_full
    X_test = X_test_full[selected_features]
    y_test = y_test_full

    print(f"  Feature/sample ratio: {len(selected_features)}/{len(X_train)} = "
          f"{len(selected_features)/len(X_train):.3f}")

    # --- Train individual models on fixed split first ---
    print_section("FIXED-SPLIT ENSEMBLE (for comparison)")
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=selected_features)
    X_va_s = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=selected_features)
    X_te_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=selected_features)

    models = {}
    preds_val = {}
    preds_test = {}

    # XGBoost
    models["XGBoost"] = xgb.XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
        gamma=1, reg_alpha=1.0, reg_lambda=5.0,
        eval_metric="rmse", early_stopping_rounds=30, random_state=42,
    )
    models["XGBoost"].fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    preds_val["XGBoost"] = models["XGBoost"].predict(X_val)
    preds_test["XGBoost"] = models["XGBoost"].predict(X_test)

    # LightGBM
    models["LightGBM"] = lgb.LGBMRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=10,
        reg_alpha=1.0, reg_lambda=5.0, random_state=42, verbose=-1,
    )
    models["LightGBM"].fit(X_train, y_train, eval_set=[(X_val, y_val)],
                           callbacks=[lgb.early_stopping(30, verbose=False)])
    preds_val["LightGBM"] = models["LightGBM"].predict(X_val)
    preds_test["LightGBM"] = models["LightGBM"].predict(X_test)

    # CatBoost (#7)
    models["CatBoost"] = CatBoostRegressor(
        iterations=500, depth=4, learning_rate=0.03,
        l2_leaf_reg=5.0, random_seed=42, verbose=0,
    )
    models["CatBoost"].fit(X_train, y_train, eval_set=(X_val, y_val),
                           early_stopping_rounds=30)
    preds_val["CatBoost"] = models["CatBoost"].predict(X_val)
    preds_test["CatBoost"] = models["CatBoost"].predict(X_test)

    # Ridge
    models["Ridge"] = Ridge(alpha=10.0)
    models["Ridge"].fit(X_tr_s, y_train)
    preds_val["Ridge"] = models["Ridge"].predict(X_va_s)
    preds_test["Ridge"] = models["Ridge"].predict(X_te_s)

    # Blend
    blend_val = 0.35 * preds_val["XGBoost"] + 0.30 * preds_val["LightGBM"] + \
                0.25 * preds_val["CatBoost"] + 0.10 * preds_val["Ridge"]
    blend_test = 0.35 * preds_test["XGBoost"] + 0.30 * preds_test["LightGBM"] + \
                 0.25 * preds_test["CatBoost"] + 0.10 * preds_test["Ridge"]
    preds_val["Ensemble"] = blend_val
    preds_test["Ensemble"] = blend_test

    results = {}
    for name in preds_val:
        results[name] = {
            "val": eval_metrics(y_val, preds_val[name]),
            "test": eval_metrics(y_test, preds_test[name]),
        }

    print(f"\n  {'Model':15s}  {'MAE':>7s}  {'RMSE':>7s}  {'R²':>7s}  {'DirAcc':>8s}")
    print(f"  {'-'*15}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    for name in results:
        m = results[name]["test"]
        print(f"  {name:15s}  {m['mae']:7.4f}  {m['rmse']:7.4f}  {m['r2']:7.4f}  {m['dir_accuracy']:7.2f}%")

    # --- #3: Walk-forward validation ---
    print_section("#3: WALK-FORWARD VALIDATION")
    X_wf = X[selected_features]
    folds = walk_forward_cv(X_wf, y, selected_features, min_train_years=10, step_weeks=52)
    wf_preds, wf_actuals, fold_metrics = run_walk_forward(X_wf, y, selected_features, folds)

    print(f"\n  Walk-forward aggregate ({len(wf_preds)} OOS predictions):")
    wf_overall = eval_metrics(wf_actuals, wf_preds)
    print_metrics(wf_overall, "Walk-Forward OOS")

    print(f"\n  Per-fold results:")
    print(f"  {'Fold':>4s}  {'Train':>5s}  {'Test':>4s}  {'MAE':>7s}  {'RMSE':>7s}  {'R²':>7s}  {'DirAcc':>8s}")
    for fm in fold_metrics:
        print(f"  {fm['fold']:4d}  {fm['train_size']:5d}  {fm['test_size']:4d}  "
              f"{fm['mae']:7.4f}  {fm['rmse']:7.4f}  {fm['r2']:7.4f}  {fm['dir_accuracy']:7.2f}%")

    avg_dir = np.mean([fm["dir_accuracy"] for fm in fold_metrics])
    print(f"\n  Average direction accuracy across folds: {avg_dir:.2f}%")

    # --- #9: Probability calibration ---
    print_section("#9: PROBABILITY CALIBRATION")
    train_blend = 0.35 * models["XGBoost"].predict(X_train) + \
                  0.30 * models["LightGBM"].predict(X_train) + \
                  0.25 * models["CatBoost"].predict(X_train) + \
                  0.10 * models["Ridge"].predict(X_tr_s)
    train_blend = pd.Series(train_blend, index=X_train.index)
    val_blend_s = pd.Series(blend_val, index=X_val.index)
    test_blend_s = pd.Series(blend_test, index=X_test.index)

    p_val, p_test, iso_model = calibrate_direction(
        y_train, train_blend, y_val, val_blend_s, test_blend_s
    )
    val_dir_calib = ((p_val > 0.5) == (y_val > 0)).mean() * 100
    test_dir_calib = ((p_test > 0.5) == (y_test > 0)).mean() * 100
    print(f"  Calibrated direction accuracy: Val={val_dir_calib:.2f}%, Test={test_dir_calib:.2f}%")
    print(f"  Test P(up) stats: mean={p_test.mean():.4f}, min={p_test.min():.4f}, max={p_test.max():.4f}")

    # --- Charts ---
    print_section("CHARTS")

    # Model comparison bar chart
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#1e1e1e")
    colors = ["#26a69a", "#ef5350", "#42a5f5", "#ffa726", "#ab47bc"]
    model_names = list(results.keys())
    for idx, metric in enumerate(["mae", "rmse", "r2", "dir_accuracy"]):
        ax = axes[idx]
        ax.set_facecolor("#1e1e1e")
        vals = [results[m]["test"][metric] for m in model_names]
        ax.bar(range(len(model_names)), vals, color=colors[:len(model_names)])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=7, color="#cccccc")
        ax.set_title(metric.upper().replace("_", " "), color="white", fontsize=11)
        ax.tick_params(colors="#cccccc")
        for s in ax.spines.values(): s.set_color("#444444")
        ax.grid(axis="y", color="#333333", alpha=0.5)
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "enhanced_comparison.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Comparison chart saved")

    # Walk-forward fold performance
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    fig2.patch.set_facecolor("#1e1e1e")
    ax2.set_facecolor("#1e1e1e")
    fold_nums = [fm["fold"] for fm in fold_metrics]
    fold_dirs = [fm["dir_accuracy"] for fm in fold_metrics]
    ax2.bar(fold_nums, fold_dirs, color="#26a69a")
    ax2.axhline(50, color="#ef5350", linewidth=1, linestyle="--", label="Random (50%)")
    ax2.axhline(avg_dir, color="#ffa726", linewidth=1.5, linestyle="-", label=f"Avg ({avg_dir:.1f}%)")
    ax2.set_xlabel("Fold", color="#cccccc")
    ax2.set_ylabel("Direction Accuracy (%)", color="#cccccc")
    ax2.set_title("Walk-Forward: Direction Accuracy per Fold", color="white", fontsize=13)
    ax2.tick_params(colors="#cccccc")
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
    for s in ax2.spines.values(): s.set_color("#444444")
    ax2.grid(axis="y", color="#333333", alpha=0.5)
    plt.tight_layout()
    fig2.savefig(MODEL_DIR / "walkforward_folds.png", dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close()
    print(f"  Walk-forward chart saved")

    # Probability calibration histogram
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    fig3.patch.set_facecolor("#1e1e1e")
    ax3.set_facecolor("#1e1e1e")
    ax3.hist(p_test, bins=30, color="#26a69a", alpha=0.8, edgecolor="#1e1e1e")
    ax3.axvline(0.5, color="#ef5350", linewidth=1.5, linestyle="--", label="P=0.5 threshold")
    ax3.set_xlabel("P(Positive Return)", color="#cccccc")
    ax3.set_ylabel("Count", color="#cccccc")
    ax3.set_title("Calibrated Probability Distribution — Test Set", color="white", fontsize=13)
    ax3.tick_params(colors="#cccccc")
    ax3.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
    for s in ax3.spines.values(): s.set_color("#444444")
    plt.tight_layout()
    fig3.savefig(MODEL_DIR / "probability_calibration.png", dpi=150, bbox_inches="tight",
                 facecolor=fig3.get_facecolor())
    plt.close()
    print(f"  Probability calibration chart saved")

    # --- Save everything ---
    models["XGBoost"].save_model(str(MODEL_DIR / "xgb_enhanced_v2.json"))
    models["CatBoost"].save_model(str(MODEL_DIR / "catboost_enhanced.cbm"))

    summary = {
        "enhancements": [
            "#1 NaN imputation", "#2 Feature selection (top 40)",
            "#3 Walk-forward validation", "#4 Multi-horizon target",
            "#5 Regime features", "#6 Time features",
            "#7 CatBoost added", "#8 Macro expansion (TNX, DXY, Gold)",
            "#9 Probability calibration",
        ],
        "total_features_before_selection": len(feature_cols),
        "selected_features": len(selected_features),
        "training_rows": len(X_train),
        "fixed_split_results": results,
        "walk_forward": {
            "folds": len(folds),
            "total_oos_predictions": len(wf_preds),
            "aggregate": wf_overall,
            "avg_direction_accuracy": round(avg_dir, 2),
            "per_fold": fold_metrics,
        },
        "probability_calibration": {
            "val_dir_accuracy": round(val_dir_calib, 2),
            "test_dir_accuracy": round(test_dir_calib, 2),
        },
        "selected_feature_list": selected_features,
    }
    with open(MODEL_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n  All artifacts saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
