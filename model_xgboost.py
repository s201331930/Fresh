"""
XGBoost classifier to predict Bullish Order Blocks (Target=1).

Key design decisions to avoid look-ahead bias:
  1. Chronological split — no shuffling. Train on the past, test on the future.
     Train: first 70%  |  Validation: next 15%  |  Test: final 15%
  2. No future-leaking features — OB group columns are excluded entirely.
  3. NaN handling — rows with NaN in features are dropped only AFTER the split
     boundaries are determined, so the time boundaries stay clean.
  4. scale_pos_weight used to handle severe class imbalance (~3% positive).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
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

OB_COLS = ["OB_Bull", "OB_Bear", "OB_Type", "OB_High", "OB_Low", "OB_Mid"]
TARGET_COL = "Target"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15


def load_and_prepare():
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)

    drop_cols = OB_COLS + [TARGET_COL]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[TARGET_COL]

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


def evaluate(model, X, y, label, threshold=0.5):
    """Evaluate and print metrics for a given split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n--- {label} (threshold={threshold}) ---")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, zero_division=0))

    if y.sum() > 0 and len(y.unique()) > 1:
        auc = roc_auc_score(y, y_prob)
        ap = average_precision_score(y, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Average Precision (PR-AUC): {ap:.4f}")
    else:
        auc, ap = None, None
        print(f"  (ROC-AUC/AP not computable — only one class in {label})")

    return y_prob, y_pred, auc, ap


def plot_feature_importance(model, feature_cols, top_n=30):
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


def plot_precision_recall(y_true, y_prob, label):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    ax.plot(recall, precision, color="#26a69a", linewidth=2)
    ax.set_xlabel("Recall", color="#cccccc")
    ax.set_ylabel("Precision", color="#cccccc")
    ax.set_title(f"Precision-Recall Curve — {label} (AP={ap:.4f})", color="white", fontsize=13)
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#444444")
    ax.grid(color="#333333", alpha=0.5)

    plt.tight_layout()
    path = MODEL_DIR / f"pr_curve_{label.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  PR curve saved to {path}")


def find_best_threshold(y_true, y_prob):
    """Find threshold that maximises F1 on the given set."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1


def main():
    print_section("DATA PREPARATION")
    X, y, feature_cols = load_and_prepare()
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target distribution: {dict(y.value_counts())}")

    print_section("CHRONOLOGICAL SPLIT")
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    X_train, y_train = drop_nan_rows(X_train, y_train, "Train")
    X_val, y_val = drop_nan_rows(X_val, y_val, "Val")
    X_test, y_test = drop_nan_rows(X_test, y_test, "Test")

    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos if pos > 0 else 1
    print(f"\n  Train class balance: {int(neg)} neg / {int(pos)} pos")
    print(f"  scale_pos_weight: {spw:.1f}")

    print_section("TRAINING XGBoost")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_weight=3,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = model.best_iteration
    print(f"\n  Best iteration: {best_iter}")

    print_section("EVALUATION — DEFAULT THRESHOLD (0.5)")
    val_prob, _, val_auc, val_ap = evaluate(model, X_val, y_val, "Validation")
    test_prob, _, test_auc, test_ap = evaluate(model, X_test, y_test, "Test")

    print_section("THRESHOLD TUNING (on Validation set)")
    best_t, best_f1 = find_best_threshold(y_val, val_prob)
    print(f"  Best threshold: {best_t:.2f} (F1={best_f1:.4f} on validation)")

    print_section(f"EVALUATION — TUNED THRESHOLD ({best_t:.2f})")
    evaluate(model, X_val, y_val, "Validation (tuned)", threshold=best_t)
    evaluate(model, X_test, y_test, "Test (tuned)", threshold=best_t)

    print_section("FEATURE IMPORTANCE")
    imp_df = plot_feature_importance(model, feature_cols)
    print("\n  Top 15 features by gain:")
    print(imp_df.head(15).to_string(index=False))

    print_section("PRECISION-RECALL CURVES")
    if y_val.sum() > 0:
        plot_precision_recall(y_val, val_prob, "Validation")
    if y_test.sum() > 0:
        plot_precision_recall(y_test, test_prob, "Test")

    model.save_model(str(MODEL_DIR / "xgb_bullish_ob.json"))
    print(f"\n  Model saved to {MODEL_DIR / 'xgb_bullish_ob.json'}")

    summary = {
        "features": len(feature_cols),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_positives": int(y_train.sum()),
        "val_positives": int(y_val.sum()),
        "test_positives": int(y_test.sum()),
        "scale_pos_weight": round(spw, 1),
        "best_iteration": best_iter,
        "best_threshold": round(best_t, 2),
        "val_roc_auc": round(val_auc, 4) if val_auc else None,
        "val_pr_auc": round(val_ap, 4) if val_ap else None,
        "test_roc_auc": round(test_auc, 4) if test_auc else None,
        "test_pr_auc": round(test_ap, 4) if test_ap else None,
    }
    with open(MODEL_DIR / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {MODEL_DIR / 'model_summary.json'}")


if __name__ == "__main__":
    main()
