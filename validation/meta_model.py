"""Strategy-selection meta-model versus the majority class.

Uses the saved Study A characteristics. Does not refit the trading models.
Depth was chosen on the same leave-one-out loop in expanded_study.py, so the
published 47.1% is slightly optimistic. This script also runs a nested
leave-one-out (depth chosen inside each training fold) and 1,000 label
permutations of the depth-3 tree and the random forest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
CHARS = ROOT / "study" / "expanded_characteristics.csv"
OUT = ROOT / "validation" / "meta_model.json"

STRATS = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
FEATURES = [
    "ann_volatility", "hurst", "autocorr_1", "autocorr_4", "skewness", "kurtosis",
    "regime_changes_per_year", "pct_above_sma50", "avg_true_range_pct",
    "up_week_ratio", "max_drawdown", "mean_reversion_score",
    "downside_vol", "tail_ratio", "volume_cv", "calmar_ratio",
    "mean_abs_return", "mean_weekly_ret",
]


def load():
    frame = pd.read_csv(CHARS)
    rows, labels, names = [], [], []
    for _, rec in frame.sort_values("asset").iterrows():
        vals = [float(rec[c]) if pd.notna(rec[c]) else np.nan for c in FEATURES]
        if not np.isfinite(vals).all():
            continue
        rows.append(vals)
        labels.append(rec["best_strategy_sharpe"])
        names.append(rec["asset"])
    return np.asarray(rows, dtype=float), np.asarray(labels), names, frame


def top2_map(frame):
    out = {}
    for _, rec in frame.iterrows():
        sharpes = [(name, float(rec[f"sharpe_{name}"])) for name in STRATS]
        sharpes.sort(key=lambda kv: kv[1], reverse=True)
        out[rec["asset"]] = {sharpes[0][0], sharpes[1][0]}
    return out


def loo_scores(X, y, names, top2, factory):
    exact = 0
    top = 0
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = factory()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])[0]
        actual = y[test_idx[0]]
        if pred == actual:
            exact += 1
        if pred in top2[names[test_idx[0]]]:
            top += 1
    n = len(y)
    return exact, top, n


def nested_depth_loo(X, y, names, top2):
    """Pick depth in {3, 4, 5} by leave-one-out inside the training fold only."""
    exact = 0
    top = 0
    chosen = []
    n = len(y)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        best_depth, best_acc = 3, -1
        inner_idx = np.flatnonzero(train)
        for depth in (3, 4, 5):
            correct = 0
            for j in inner_idx:
                inner = train.copy()
                inner[j] = False
                model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=2, random_state=42)
                model.fit(X[inner], y[inner])
                if model.predict(X[j:j + 1])[0] == y[j]:
                    correct += 1
            if correct > best_acc:
                best_acc = correct
                best_depth = depth
        model = DecisionTreeClassifier(max_depth=best_depth, min_samples_leaf=2, random_state=42)
        model.fit(X[train], y[train])
        pred = model.predict(X[i:i + 1])[0]
        chosen.append(int(best_depth))
        if pred == y[i]:
            exact += 1
        if pred in top2[names[i]]:
            top += 1
    return exact, top, n, chosen


def permute(X, y, names, top2, factory, n_perm, seed, observed_exact):
    rng = np.random.default_rng(seed)
    hits = 0
    exacts = np.empty(n_perm, dtype=int)
    for k in range(n_perm):
        shuffled = rng.permutation(y)
        # Top-2 is a property of the asset's Sharpe ranks, not of the shuffled label.
        # The permutation test is for exact accuracy. Top-2 under a shuffled label
        # is not a meaningful null for "is the predicted name in the true top two",
        # so only exact accuracy is tested.
        exact, _, _ = loo_scores(X, shuffled, names, top2, factory)
        exacts[k] = exact
        if exact >= observed_exact:
            hits += 1
    return {
        "n_perm": n_perm,
        "n_ge_observed": int(hits),
        "p_value": (hits + 1) / (n_perm + 1),
        "null_mean_exact": float(exacts.mean()),
        "null_p95_exact": float(np.quantile(exacts, 0.95)),
    }


def main():
    X, y, names, frame = load()
    top2 = top2_map(frame)
    counts = pd.Series(y).value_counts()
    majority = counts.index[0]
    majority_exact = int((y == majority).sum())
    majority_top2 = int(sum(majority in top2[name] for name in names))
    always_opt_exact = int((y == "Binary (t=opt)").sum())
    always_opt_top2 = int(sum("Binary (t=opt)" in top2[name] for name in names))

    dt_exact, dt_top, n = loo_scores(
        X, y, names, top2,
        lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42),
    )
    rf_exact, rf_top, _ = loo_scores(
        X, y, names, top2,
        lambda: RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42
        ),
    )
    nested_exact, nested_top, _, depths = nested_depth_loo(X, y, names, top2)
    dt_perm = permute(
        X, y, names, top2,
        lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, random_state=42),
        1000, 42, dt_exact,
    )
    rf_perm = permute(
        X, y, names, top2,
        lambda: RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=42
        ),
        1000, 42, rf_exact,
    )
    payload = {
        "n": n,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "majority_label": str(majority),
        "majority_exact": majority_exact,
        "majority_exact_pct": round(100 * majority_exact / n, 1),
        "majority_top2": majority_top2,
        "majority_top2_pct": round(100 * majority_top2 / n, 1),
        "always_t_opt_exact": always_opt_exact,
        "always_t_opt_exact_pct": round(100 * always_opt_exact / n, 1),
        "always_t_opt_top2": always_opt_top2,
        "always_t_opt_top2_pct": round(100 * always_opt_top2 / n, 1),
        "tree_depth3_loo_exact": dt_exact,
        "tree_depth3_loo_exact_pct": round(100 * dt_exact / n, 1),
        "tree_depth3_loo_top2": dt_top,
        "tree_depth3_loo_top2_pct": round(100 * dt_top / n, 1),
        "forest_loo_exact": rf_exact,
        "forest_loo_exact_pct": round(100 * rf_exact / n, 1),
        "forest_loo_top2": rf_top,
        "forest_loo_top2_pct": round(100 * rf_top / n, 1),
        "nested_tree_loo_exact": nested_exact,
        "nested_tree_loo_exact_pct": round(100 * nested_exact / n, 1),
        "nested_tree_loo_top2": nested_top,
        "nested_tree_loo_top2_pct": round(100 * nested_top / n, 1),
        "nested_depth_counts": {str(d): int(depths.count(d)) for d in (3, 4, 5)},
        "tree_permutation": dt_perm,
        "forest_permutation": rf_perm,
        "seed": 42,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
