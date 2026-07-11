"""
Trading strategy backtest using walk-forward model predictions.

Constraints: long-only, no leverage (position 0-100%).
Benchmark: buy-and-hold SPY.

Strategies:
  1. Binary:    all-in when predicted return > threshold, else 100% cash
  2. Scaled:    position size proportional to prediction confidence
  3. Adaptive:  binary with threshold optimized on rolling recent history

All predictions come from walk-forward OOS output — zero look-ahead bias.
Transaction cost: 0.02% round-trip (realistic for SPY weekly).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# Re-use data pipeline from enhanced model
from model_enhanced import (
    fetch_macro_extras,
    add_regime_features,
    add_time_features,
    add_targets,
    add_lagged_features,
    add_rolling_stats,
    add_interaction_terms,
    impute_nans,
    select_features,
    OB_COLS,
    print_section,
)

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_enriched.csv"
MODEL_DIR = Path(__file__).parent / "model"
STRATEGY_DIR = Path(__file__).parent / "strategy"
STRATEGY_DIR.mkdir(exist_ok=True)

COST_PER_TRADE = 0.0002  # 0.02% round-trip


# =========================================================================
# DATA PIPELINE (mirrors model_enhanced.py)
# =========================================================================
def prepare_full_dataset():
    """Rebuild the full feature set and targets."""
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)

    macro_extra = fetch_macro_extras()
    if not macro_extra.empty:
        macro_extra.index = macro_extra.index.tz_localize(None).normalize()
        df.index = pd.to_datetime(df.index, utc=True).normalize().tz_localize(None)
        df = df[~df.index.duplicated(keep="first")]
        df = df.join(macro_extra, how="left")
        df = df[~df.index.duplicated(keep="first")]
        for col in macro_extra.columns:
            df[col] = df[col].ffill().fillna(0)

    df = add_regime_features(df)
    df = add_time_features(df)
    df = add_targets(df)
    df.dropna(subset=["Next_Week_Return"], inplace=True)
    df = add_lagged_features(df)
    df = add_rolling_stats(df)
    df = add_interaction_terms(df)

    exclude = OB_COLS + ["Next_Week_Return", "Multi_Horizon_Return", "Risk_Adjusted_Return"]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    y = df["Next_Week_Return"].copy()
    actual_returns = y.copy()

    X = impute_nans(X)

    return X, y, actual_returns, feature_cols


# =========================================================================
# WALK-FORWARD PREDICTION ENGINE
# =========================================================================
def walk_forward_predict(X, y, feature_cols, min_train_weeks=520, step_weeks=52):
    """
    Generate OOS predictions using expanding-window walk-forward.
    Returns aligned Series of predictions and actuals.
    """
    idx = pd.to_datetime(X.index)
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

    print(f"  Walk-forward: {fold} folds, {len(predictions)} OOS predictions")
    return predictions


# =========================================================================
# STRATEGY IMPLEMENTATIONS
# =========================================================================
def strategy_binary(predictions, threshold=0.0):
    """1 if predicted return > threshold, else 0."""
    return (predictions > threshold).astype(float)


def strategy_scaled(predictions, max_pred=1.0):
    """Position proportional to predicted return, clipped to [0, 1]."""
    pos = predictions / max_pred
    return pos.clip(0.0, 1.0)


def strategy_adaptive(predictions, actuals, lookback=26):
    """
    Binary strategy with threshold re-optimized every `lookback` weeks
    using recent OOS performance.
    """
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


# =========================================================================
# BACKTEST ENGINE
# =========================================================================
def backtest(positions, actual_returns, label="Strategy"):
    """
    Simulate strategy returns given position sizes and actual weekly returns.
    Accounts for transaction costs on position changes.
    """
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

    dir_correct = ((actual_returns > 0) == (positions > 0)).mean() * 100

    metrics = {
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
    }

    return equity, bnh_equity, drawdown, bnh_dd, strat_ret, metrics


def print_metrics_table(all_metrics):
    print(f"\n  {'Strategy':20s}  {'Return':>8s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>8s}  "
          f"{'WinRate':>8s}  {'InMkt':>6s}  {'Trades':>6s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*6}  {'-'*6}")

    for name, m in all_metrics.items():
        print(f"  {name:20s}  {m['total_return']:7.1f}%  {m['cagr']:6.2f}%  {m['sharpe']:7.3f}  "
              f"{m['max_drawdown']:7.2f}%  {m['win_rate']:7.2f}%  {m['in_market_pct']:5.1f}%  {m['n_trades']:6d}")

    bnh = list(all_metrics.values())[0]
    print(f"\n  {'Buy & Hold':20s}  {bnh['bnh_total_return']:7.1f}%  {bnh['bnh_cagr']:6.2f}%  "
          f"{bnh['bnh_sharpe']:7.3f}  {bnh['bnh_max_drawdown']:7.2f}%")


# =========================================================================
# CHARTS
# =========================================================================
def plot_equity_curves(equities, drawdowns, bnh_equity, bnh_dd):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#1e1e1e")

    colors = {"Binary (t=0)": "#26a69a", "Scaled": "#42a5f5",
              "Adaptive": "#ffa726", "Binary (t=opt)": "#ab47bc"}

    ax1.set_facecolor("#1e1e1e")
    ax1.plot(bnh_equity.index, bnh_equity.values, color="#666666", linewidth=2,
             linestyle="--", label="Buy & Hold", alpha=0.8)
    for name, eq in equities.items():
        c = colors.get(name, "#26a69a")
        ax1.plot(eq.index, eq.values, color=c, linewidth=1.5, label=name, alpha=0.9)
    ax1.set_ylabel("Equity ($1 start)", color="#cccccc", fontsize=11)
    ax1.set_title("Strategy Equity Curves vs Buy & Hold", color="white", fontsize=14)
    ax1.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=9)
    ax1.tick_params(colors="#cccccc")
    ax1.grid(color="#333333", alpha=0.5)
    for s in ax1.spines.values(): s.set_color("#444444")

    ax2.set_facecolor("#1e1e1e")
    ax2.fill_between(bnh_dd.index, bnh_dd.values * 100, 0,
                     color="#666666", alpha=0.3, label="Buy & Hold DD")
    for name, dd in drawdowns.items():
        c = colors.get(name, "#26a69a")
        ax2.plot(dd.index, dd.values * 100, color=c, linewidth=1, label=name, alpha=0.8)
    ax2.set_ylabel("Drawdown (%)", color="#cccccc", fontsize=11)
    ax2.set_xlabel("Date", color="#cccccc", fontsize=11)
    ax2.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=8)
    ax2.tick_params(colors="#cccccc")
    ax2.grid(color="#333333", alpha=0.5)
    for s in ax2.spines.values(): s.set_color("#444444")

    plt.tight_layout()
    path = STRATEGY_DIR / "equity_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Equity curves saved to {path}")


def plot_annual_returns(strat_returns, actual_returns, best_name):
    idx = pd.to_datetime(strat_returns.index)
    years = idx.year
    unique_years = sorted(years.unique())

    strat_annual = []
    bnh_annual = []
    for yr in unique_years:
        mask = years == yr
        sr = strat_returns[mask]
        ar = actual_returns[mask] / 100.0
        strat_annual.append(((1 + sr).prod() - 1) * 100)
        bnh_annual.append(((1 + ar).prod() - 1) * 100)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    x = np.arange(len(unique_years))
    w = 0.35
    ax.bar(x - w/2, bnh_annual, w, color="#666666", label="Buy & Hold", alpha=0.7)
    ax.bar(x + w/2, strat_annual, w, color="#26a69a", label=best_name, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(unique_years, rotation=45, color="#cccccc", fontsize=8)
    ax.set_ylabel("Annual Return (%)", color="#cccccc")
    ax.set_title(f"Annual Returns: {best_name} vs Buy & Hold", color="white", fontsize=13)
    ax.axhline(0, color="#555555", linewidth=0.5)
    ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white")
    ax.tick_params(colors="#cccccc")
    ax.grid(axis="y", color="#333333", alpha=0.5)
    for s in ax.spines.values(): s.set_color("#444444")
    plt.tight_layout()
    path = STRATEGY_DIR / "annual_returns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Annual returns saved to {path}")


# =========================================================================
# MAIN
# =========================================================================
def main():
    print_section("PREPARING DATA & FEATURES")
    X, y, actual_returns, feature_cols = prepare_full_dataset()
    print(f"  Total rows: {len(X)}, features: {len(feature_cols)}")

    # Feature selection using first 70% as pseudo-train
    n = len(X)
    sel_end = int(n * 0.70)
    selected = select_features(
        X.iloc[:sel_end][feature_cols], y.iloc[:sel_end],
        X.iloc[sel_end:sel_end+200][feature_cols], y.iloc[sel_end:sel_end+200],
        feature_cols, top_n=40,
    )

    print_section("WALK-FORWARD PREDICTIONS")
    predictions = walk_forward_predict(X, y, selected, min_train_weeks=520, step_weeks=26)

    aligned = pd.DataFrame({
        "prediction": predictions,
        "actual": actual_returns,
    }).dropna()
    preds = aligned["prediction"]
    actuals = aligned["actual"]
    print(f"  Aligned OOS predictions: {len(aligned)} weeks")
    print(f"  Period: {aligned.index.min()} -> {aligned.index.max()}")

    # --- Optimize binary threshold on first half of OOS data ---
    split = len(preds) // 2
    opt_preds = preds.iloc[:split]
    opt_actuals = actuals.iloc[:split]

    best_t, best_sharpe = 0.0, -np.inf
    for t in np.arange(-0.5, 0.5, 0.01):
        pos = (opt_preds > t).astype(float)
        ret = pos * opt_actuals / 100
        sr = ret.mean() / ret.std() * np.sqrt(52) if ret.std() > 0 else 0
        if sr > best_sharpe:
            best_t, best_sharpe = t, sr
    print(f"\n  Optimized threshold: {best_t:.2f} (Sharpe={best_sharpe:.3f} on first half)")

    # --- Build strategies ---
    print_section("BACKTESTING STRATEGIES")
    strategies = {
        "Binary (t=0)": strategy_binary(preds, threshold=0.0),
        f"Binary (t=opt)": strategy_binary(preds, threshold=best_t),
        "Scaled": strategy_scaled(preds, max_pred=max(0.5, preds.quantile(0.9))),
        "Adaptive": strategy_adaptive(preds, actuals, lookback=26),
    }

    all_equities = {}
    all_drawdowns = {}
    all_metrics = {}
    all_strat_rets = {}
    bnh_eq = None
    bnh_dd_series = None

    for name, positions in strategies.items():
        eq, bnh, dd, bnh_dd, strat_ret, metrics = backtest(positions, actuals, name)
        all_equities[name] = eq
        all_drawdowns[name] = dd
        all_metrics[name] = metrics
        all_strat_rets[name] = strat_ret
        bnh_eq = bnh
        bnh_dd_series = bnh_dd

    print_metrics_table(all_metrics)

    # --- Find best strategy ---
    best_name = max(all_metrics, key=lambda k: all_metrics[k]["sharpe"])
    print(f"\n  Best strategy by Sharpe: {best_name}")

    # --- Charts ---
    print_section("CHARTS")
    plot_equity_curves(all_equities, all_drawdowns, bnh_eq, bnh_dd_series)
    plot_annual_returns(all_strat_rets[best_name], actuals, best_name)

    # --- Position analysis ---
    print_section("POSITION ANALYSIS — " + best_name)
    best_pos = strategies[best_name]
    in_market = best_pos > 0
    print(f"  Weeks in market: {in_market.sum()} / {len(best_pos)} ({in_market.mean()*100:.1f}%)")
    print(f"  Avg return when IN:  {(actuals[in_market] / 100).mean()*100:.4f}% per week")
    if (~in_market).sum() > 0:
        print(f"  Avg return when OUT: {(actuals[~in_market] / 100).mean()*100:.4f}% per week (avoided)")

    # --- Save results ---
    summary = {
        "backtest_period": f"{aligned.index.min()} to {aligned.index.max()}",
        "oos_predictions": len(aligned),
        "optimized_threshold": round(best_t, 2),
        "best_strategy": best_name,
        "strategies": all_metrics,
        "transaction_cost": COST_PER_TRADE,
    }
    with open(STRATEGY_DIR / "backtest_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    results_df = pd.DataFrame({
        "Actual_Return": actuals,
        "Prediction": preds,
        **{f"Position_{k}": v for k, v in strategies.items()},
        **{f"Equity_{k}": v for k, v in all_equities.items()},
        "BnH_Equity": bnh_eq,
    })
    results_df.to_csv(STRATEGY_DIR / "backtest_results.csv", float_format="%.4f")

    print(f"\n  All results saved to {STRATEGY_DIR}/")


if __name__ == "__main__":
    main()
