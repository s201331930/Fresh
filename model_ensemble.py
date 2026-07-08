"""
Enhanced ensemble model for next-week return prediction.

Improvements over baseline XGBoost:
  1. Lagged features (1-4 weeks) on key signals — captures momentum,
     mean-reversion, and persistence patterns.
  2. Interaction terms — cross-variable products that encode non-linear
     relationships (e.g., VIX × momentum, RSI × MACD).
  3. Ensemble of XGBoost + LightGBM + Ridge — blended with weights
     optimized on the validation set.

Target: (Close[t+1] - Close[t]) / Close[t] * 100
All OB group columns excluded. No look-ahead bias.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_enriched.csv"
MODEL_DIR = Path(__file__).parent / "model"
MODEL_DIR.mkdir(exist_ok=True)

OB_COLS = ["OB_Bull", "OB_Bear", "OB_Type", "OB_High", "OB_Low", "OB_Mid", "Target"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# --- Features to lag (1-4 weeks back) ---
LAG_FEATURES = [
    "Close_PctChg", "Open_PctChg", "High_PctChg", "Low_PctChg",
    "VIX_Close", "VIX_Close_PctChg",
    "Volume_Log_Chg",
    "RSI_14", "MACD_Hist", "BB_PctB",
    "ADX_14", "DI_Plus_14", "DI_Minus_14",
    "Brent_Close_PctChg",
    "VIX_RSI_14",
]
LAG_PERIODS = [1, 2, 3, 4]

# --- Interaction term pairs ---
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


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged versions of key features."""
    new_cols = {}
    for col in LAG_FEATURES:
        if col not in df.columns:
            continue
        for lag in LAG_PERIODS:
            new_cols[f"{col}_Lag{lag}"] = df[col].shift(lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-variable interaction products."""
    new_cols = {}
    for col_a, col_b in INTERACTIONS:
        if col_a in df.columns and col_b in df.columns:
            name = f"{col_a}_x_{col_b}"
            new_cols[name] = df[col_a] * df[col_b]
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics that capture recent regime."""
    new_cols = {}
    for window in [4, 8, 12]:
        new_cols[f"Close_PctChg_RollMean_{window}"] = df["Close_PctChg"].rolling(window).mean()
        new_cols[f"Close_PctChg_RollStd_{window}"] = df["Close_PctChg"].rolling(window).std()
        new_cols[f"VIX_Close_RollMean_{window}"] = df["VIX_Close"].rolling(window).mean()
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def load_and_prepare():
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)

    df["Next_Week_Return"] = df["Close"].pct_change(1).shift(-1) * 100
    df.dropna(subset=["Next_Week_Return"], inplace=True)

    df = add_lagged_features(df)
    df = add_rolling_stats(df)
    df = add_interaction_terms(df)

    feature_cols = [c for c in df.columns if c not in OB_COLS + ["Next_Week_Return"]]

    X = df[feature_cols]
    y = df["Next_Week_Return"]

    return X, y, feature_cols


def chronological_split(X, y):
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    splits = {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "val": (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        "test": (X.iloc[val_end:], y.iloc[val_end:]),
    }

    for name, (Xi, yi) in splits.items():
        print(f"  {name.capitalize():6s}: {Xi.index.min()} -> {Xi.index.max()}  ({len(Xi)} rows)")

    return splits


def drop_nan_rows(X, y, label):
    mask = X.notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  {label}: dropped {dropped} NaN rows ({len(X) - dropped} remaining)")
    return X[mask], y[mask]


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def eval_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    dir_acc = ((y_true > 0) == (y_pred > 0)).mean() * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "dir_accuracy": round(dir_acc, 2)}


def print_metrics(metrics, label):
    print(f"  {label:20s}  MAE={metrics['mae']:.4f}%  RMSE={metrics['rmse']:.4f}%  "
          f"R²={metrics['r2']:.4f}  DirAcc={metrics['dir_accuracy']:.2f}%")


def optimize_blend_weights(preds_val: dict, y_val: np.ndarray):
    """Find blend weights that minimize RMSE on validation set."""
    names = list(preds_val.keys())
    pred_matrix = np.column_stack([preds_val[n] for n in names])

    def objective(weights):
        w = weights / weights.sum()
        blended = pred_matrix @ w
        return np.sqrt(mean_squared_error(y_val, blended))

    n_models = len(names)
    x0 = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0)] * n_models
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method="SLSQP")
    weights = {n: round(w, 4) for n, w in zip(names, result.x)}
    return weights


def plot_comparison(results: dict, y_test, test_preds: dict):
    models = list(results.keys())
    metrics_names = ["mae", "rmse", "r2", "dir_accuracy"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#1e1e1e")
    colors = ["#26a69a", "#ef5350", "#42a5f5", "#ffa726", "#ab47bc"]

    for idx, metric in enumerate(metrics_names):
        ax = axes[idx]
        ax.set_facecolor("#1e1e1e")
        vals = [results[m]["test"][metric] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors[:len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8, color="#cccccc")
        ax.set_title(metric.upper().replace("_", " "), color="white", fontsize=11)
        ax.tick_params(colors="#cccccc")
        for spine in ax.spines.values():
            spine.set_color("#444444")
        ax.grid(axis="y", color="#333333", alpha=0.5)

    plt.tight_layout()
    path = MODEL_DIR / "ensemble_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Comparison chart saved to {path}")

    # Test predictions overlay
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    fig2.patch.set_facecolor("#1e1e1e")
    ax2.set_facecolor("#1e1e1e")
    x = range(len(y_test))
    ax2.plot(x, y_test.values, color="#666666", alpha=0.6, linewidth=1, label="Actual")
    style = [("#26a69a", 1.5), ("#ef5350", 1.5), ("#42a5f5", 1.0), ("#ffa726", 1.0), ("#ab47bc", 2.0)]
    for i, name in enumerate(models):
        c, lw = style[i % len(style)]
        ax2.plot(x, test_preds[name], color=c, alpha=0.8, linewidth=lw, label=name)
    ax2.axhline(0, color="#555555", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Week", color="#cccccc")
    ax2.set_ylabel("Return (%)", color="#cccccc")
    ax2.set_title("Test Set — All Models vs Actual", color="white", fontsize=13)
    ax2.tick_params(colors="#cccccc")
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#444444")
    ax2.grid(color="#333333", alpha=0.5)
    plt.tight_layout()
    path2 = MODEL_DIR / "ensemble_test_overlay.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
    plt.close()
    print(f"  Test overlay chart saved to {path2}")


def main():
    print_section("DATA PREPARATION (with lagged + interaction features)")
    X, y, feature_cols = load_and_prepare()
    print(f"Total features: {len(feature_cols)} columns")
    n_lagged = sum(1 for c in feature_cols if "_Lag" in c)
    n_roll = sum(1 for c in feature_cols if "Roll" in c)
    n_interact = sum(1 for c in feature_cols if "_x_" in c)
    print(f"  Lagged: {n_lagged},  Rolling: {n_roll},  Interactions: {n_interact},  "
          f"Base: {len(feature_cols) - n_lagged - n_roll - n_interact}")

    print_section("CHRONOLOGICAL SPLIT")
    splits = chronological_split(X, y)

    clean = {}
    for name, (Xi, yi) in splits.items():
        Xi_c, yi_c = drop_nan_rows(Xi, yi, name.capitalize())
        clean[name] = (Xi_c, yi_c)

    X_train, y_train = clean["train"]
    X_val, y_val = clean["val"]
    X_test, y_test = clean["test"]

    print(f"\n  Final sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- Scale features for Ridge ---
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    results = {}
    val_preds = {}
    test_preds = {}

    # =====================================================================
    # MODEL 1: XGBoost
    # =====================================================================
    print_section("MODEL 1: XGBoost")
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=1,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    print(f"  Best iteration: {xgb_model.best_iteration}")

    val_preds["XGBoost"] = xgb_model.predict(X_val)
    test_preds["XGBoost"] = xgb_model.predict(X_test)
    results["XGBoost"] = {
        "val": eval_metrics(y_val, val_preds["XGBoost"]),
        "test": eval_metrics(y_test, test_preds["XGBoost"]),
    }
    print_metrics(results["XGBoost"]["val"], "Val")
    print_metrics(results["XGBoost"]["test"], "Test")

    # =====================================================================
    # MODEL 2: LightGBM
    # =====================================================================
    print_section("MODEL 2: LightGBM")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    print(f"  Best iteration: {lgb_model.best_iteration_}")

    val_preds["LightGBM"] = lgb_model.predict(X_val)
    test_preds["LightGBM"] = lgb_model.predict(X_test)
    results["LightGBM"] = {
        "val": eval_metrics(y_val, val_preds["LightGBM"]),
        "test": eval_metrics(y_test, test_preds["LightGBM"]),
    }
    print_metrics(results["LightGBM"]["val"], "Val")
    print_metrics(results["LightGBM"]["test"], "Test")

    # =====================================================================
    # MODEL 3: Ridge Regression (linear baseline)
    # =====================================================================
    print_section("MODEL 3: Ridge Regression")
    ridge_model = Ridge(alpha=10.0)
    ridge_model.fit(X_train_scaled, y_train)

    val_preds["Ridge"] = ridge_model.predict(X_val_scaled)
    test_preds["Ridge"] = ridge_model.predict(X_test_scaled)
    results["Ridge"] = {
        "val": eval_metrics(y_val, val_preds["Ridge"]),
        "test": eval_metrics(y_test, test_preds["Ridge"]),
    }
    print_metrics(results["Ridge"]["val"], "Val")
    print_metrics(results["Ridge"]["test"], "Test")

    # =====================================================================
    # MODEL 4: XGBoost (tuned, more aggressive)
    # =====================================================================
    print_section("MODEL 4: XGBoost Tuned")
    xgb_tuned = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_weight=10,
        gamma=2,
        reg_alpha=2.0,
        reg_lambda=10.0,
        eval_metric="rmse",
        early_stopping_rounds=80,
        random_state=42,
    )
    xgb_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    print(f"  Best iteration: {xgb_tuned.best_iteration}")

    val_preds["XGB_Tuned"] = xgb_tuned.predict(X_val)
    test_preds["XGB_Tuned"] = xgb_tuned.predict(X_test)
    results["XGB_Tuned"] = {
        "val": eval_metrics(y_val, val_preds["XGB_Tuned"]),
        "test": eval_metrics(y_test, test_preds["XGB_Tuned"]),
    }
    print_metrics(results["XGB_Tuned"]["val"], "Val")
    print_metrics(results["XGB_Tuned"]["test"], "Test")

    # =====================================================================
    # ENSEMBLE: Optimized blend
    # =====================================================================
    print_section("ENSEMBLE — OPTIMIZED BLEND")
    blend_weights = optimize_blend_weights(val_preds, y_val.values)
    print(f"  Optimized weights: {blend_weights}")

    val_blend = sum(w * val_preds[n] for n, w in blend_weights.items())
    test_blend = sum(w * test_preds[n] for n, w in blend_weights.items())

    val_preds["Ensemble"] = val_blend
    test_preds["Ensemble"] = test_blend
    results["Ensemble"] = {
        "val": eval_metrics(y_val, val_blend),
        "test": eval_metrics(y_test, test_blend),
    }
    print_metrics(results["Ensemble"]["val"], "Val")
    print_metrics(results["Ensemble"]["test"], "Test")

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print_section("SUMMARY — ALL MODELS (TEST SET)")
    print(f"  {'Model':20s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  {'DirAcc':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for name in results:
        m = results[name]["test"]
        print(f"  {name:20s}  {m['mae']:8.4f}  {m['rmse']:8.4f}  {m['r2']:8.4f}  {m['dir_accuracy']:7.2f}%")

    print_section("CHARTS")
    plot_comparison(results, y_test, test_preds)

    # Feature importance from best tree model
    best_tree = xgb_tuned if results["XGB_Tuned"]["test"]["dir_accuracy"] >= results["XGBoost"]["test"]["dir_accuracy"] else xgb_model
    importance = best_tree.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        {"feature": list(importance.keys()), "gain": list(importance.values())}
    ).sort_values("gain", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    top = imp_df.head(30)
    ax.barh(top["feature"][::-1], top["gain"][::-1], color="#26a69a")
    ax.set_xlabel("Gain", color="#cccccc")
    ax.set_title("Top 30 Feature Importances — Enhanced Model", color="white", fontsize=13)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    imp_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    print(f"  Feature importance saved")

    print("\n  Top 15 features:")
    print(imp_df.head(15).to_string(index=False))

    # Save models
    xgb_model.save_model(str(MODEL_DIR / "xgb_enhanced.json"))
    xgb_tuned.save_model(str(MODEL_DIR / "xgb_tuned.json"))
    lgb_model.booster_.save_model(str(MODEL_DIR / "lgb_enhanced.txt"))

    summary = {
        "total_features": len(feature_cols),
        "lagged_features": n_lagged,
        "rolling_features": n_roll,
        "interaction_features": n_interact,
        "blend_weights": blend_weights,
        "results": results,
    }
    with open(MODEL_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  All models and summary saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
