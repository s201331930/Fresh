"""
XGBoost regressor to predict next-week return magnitude.

Target: (Close[t+1] - Close[t]) / Close[t]  (percentage return, shifted forward)

Key design decisions to avoid look-ahead bias:
  1. Target is SHIFTED FORWARD by one week — we predict next week's return
     using only information available up to the current week.
  2. OB group columns excluded entirely (they are confirmed retroactively).
  3. Chronological split — no shuffling. Train on the past, test on the future.
     Train: first 70%  |  Validation: next 15%  |  Test: final 15%
  4. Last row is dropped (no next-week return available).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
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


def load_and_prepare():
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)

    df["Next_Week_Return"] = df["Close"].pct_change(1).shift(-1) * 100

    df.dropna(subset=["Next_Week_Return"], inplace=True)

    feature_cols = [c for c in df.columns if c not in OB_COLS + ["Next_Week_Return"]]

    X = df[feature_cols]
    y = df["Next_Week_Return"]

    return X, y, feature_cols


def chronological_split(X, y):
    """Split into train/val/test by time — no shuffling."""
    n = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"  Train: {X_train.index.min()} -> {X_train.index.max()}  ({len(X_train)} rows)")
    print(f"  Val:   {X_val.index.min()} -> {X_val.index.max()}  ({len(X_val)} rows)")
    print(f"  Test:  {X_test.index.min()} -> {X_test.index.max()}  ({len(X_test)} rows)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def drop_nan_rows(X, y, label):
    """Drop rows with NaN features; return cleaned X, y."""
    mask = X.notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        print(f"  {label}: dropped {dropped} rows with NaN features ({len(X) - dropped} remaining)")
    return X[mask], y[mask]


def print_section(title):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


def evaluate(model, X, y, label):
    """Evaluate regression metrics."""
    y_pred = model.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    direction_actual = (y > 0).astype(int)
    direction_pred = (y_pred > 0).astype(int)
    dir_accuracy = (direction_actual == direction_pred).mean() * 100

    print(f"\n--- {label} ---")
    print(f"  MAE:                {mae:.4f}%")
    print(f"  RMSE:               {rmse:.4f}%")
    print(f"  R²:                 {r2:.4f}")
    print(f"  Direction Accuracy: {dir_accuracy:.2f}%")
    print(f"  Actual mean return: {y.mean():.4f}%  (std: {y.std():.4f}%)")
    print(f"  Predicted mean:     {y_pred.mean():.4f}%  (std: {y_pred.std():.4f}%)")

    return y_pred, {"mae": mae, "rmse": rmse, "r2": r2, "dir_accuracy": dir_accuracy}


def plot_feature_importance(model, top_n=30):
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        {"feature": list(importance.keys()), "gain": list(importance.values())}
    ).sort_values("gain", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.35)))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    top = imp_df.head(top_n)
    ax.barh(top["feature"][::-1], top["gain"][::-1], color="#26a69a")
    ax.set_xlabel("Gain", color="#cccccc")
    ax.set_title(f"Top {top_n} Feature Importances (Gain)", color="white", fontsize=13)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    plt.tight_layout()
    path = MODEL_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Feature importance chart saved to {path}")

    imp_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    return imp_df


def plot_predictions_vs_actual(y_true, y_pred, label):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1e1e1e")

    # Scatter: predicted vs actual
    ax = axes[0]
    ax.set_facecolor("#1e1e1e")
    ax.scatter(y_true, y_pred, alpha=0.5, s=12, color="#26a69a")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="#ef5350", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Return (%)", color="#cccccc")
    ax.set_ylabel("Predicted Return (%)", color="#cccccc")
    ax.set_title(f"Predicted vs Actual — {label}", color="white", fontsize=12)
    ax.tick_params(colors="#cccccc")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(color="#333333", alpha=0.5)

    # Time series overlay
    ax2 = axes[1]
    ax2.set_facecolor("#1e1e1e")
    x_range = range(len(y_true))
    ax2.plot(x_range, y_true.values, color="#26a69a", alpha=0.7, linewidth=1, label="Actual")
    ax2.plot(x_range, y_pred, color="#ef5350", alpha=0.7, linewidth=1, label="Predicted")
    ax2.axhline(0, color="#666666", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Week", color="#cccccc")
    ax2.set_ylabel("Return (%)", color="#cccccc")
    ax2.set_title(f"Return Time Series — {label}", color="white", fontsize=12)
    ax2.tick_params(colors="#cccccc")
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
    for spine in ax2.spines.values():
        spine.set_color("#444444")
    ax2.grid(color="#333333", alpha=0.5)

    plt.tight_layout()
    path = MODEL_DIR / f"predictions_{label.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Prediction chart saved to {path}")


def main():
    print_section("DATA PREPARATION")
    X, y, feature_cols = load_and_prepare()
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target: Next_Week_Return (% change)")
    print(f"  Mean: {y.mean():.4f}%,  Std: {y.std():.4f}%,  Min: {y.min():.4f}%,  Max: {y.max():.4f}%")

    print_section("CHRONOLOGICAL SPLIT")
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    X_train, y_train = drop_nan_rows(X_train, y_train, "Train")
    X_val, y_val = drop_nan_rows(X_val, y_val, "Val")
    X_test, y_test = drop_nan_rows(X_test, y_test, "Test")

    print(f"\n  Train target stats: mean={y_train.mean():.4f}%, std={y_train.std():.4f}%")
    print(f"  Val target stats:   mean={y_val.mean():.4f}%, std={y_val.std():.4f}%")
    print(f"  Test target stats:  mean={y_test.mean():.4f}%, std={y_test.std():.4f}%")

    print_section("TRAINING XGBoost REGRESSOR")
    model = xgb.XGBRegressor(
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

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter}")

    print_section("EVALUATION")
    train_pred, train_metrics = evaluate(model, X_train, y_train, "Train")
    val_pred, val_metrics = evaluate(model, X_val, y_val, "Validation")
    test_pred, test_metrics = evaluate(model, X_test, y_test, "Test")

    print_section("FEATURE IMPORTANCE")
    imp_df = plot_feature_importance(model)
    print("\n  Top 15 features by gain:")
    print(imp_df.head(15).to_string(index=False))

    print_section("PREDICTION CHARTS")
    plot_predictions_vs_actual(y_val, val_pred, "Validation")
    plot_predictions_vs_actual(y_test, test_pred, "Test")

    model.save_model(str(MODEL_DIR / "xgb_next_week_return.json"))
    print(f"\n  Model saved to {MODEL_DIR / 'xgb_next_week_return.json'}")

    summary = {
        "model_type": "XGBRegressor",
        "target": "Next_Week_Return (% change)",
        "features": len(feature_cols),
        "best_iteration": best_iter,
        "train": {"rows": len(X_train), **{k: round(v, 4) for k, v in train_metrics.items()}},
        "validation": {"rows": len(X_val), **{k: round(v, 4) for k, v in val_metrics.items()}},
        "test": {"rows": len(X_test), **{k: round(v, 4) for k, v in test_metrics.items()}},
    }
    with open(MODEL_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {MODEL_DIR / 'model_summary.json'}")


if __name__ == "__main__":
    main()
