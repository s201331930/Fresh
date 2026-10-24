"""
SPY daily execution study.

Three matched pipelines, same ensemble and the same four strategy variants:

  weekly  weekly features, weekly target, weekly execution
  daily   daily features, next-day target, daily execution
  mixed   daily features plus last COMPLETED week's features, next-day target,
          daily execution

No look-ahead:
  - Indicators, lags, and HMM states at bar t use data through bar t only.
  - A weekly bar labeled Friday is released on the next calendar day, so the
    Friday session itself still sees the prior week.
  - HMM is refit on past bars only. Between refits a forward filter updates
    the state one bar at a time.
  - Early stopping and feature selection use only the trailing part of the
    training window, never the test bars.
  - Binary (t=opt) threshold and the Scaled divisor are fit on 2010-2012 and
    frozen. Every strategy is scored on 2013 onward.
"""

import json
import time
import warnings
from collections import Counter
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from catboost import CatBoostRegressor
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from technical_indicators import add_indicators

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent / "daily_study"
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

COST_BPS = 2.0  # headline round-trip cost, in basis points of traded notional
COST_GRID = (1.0, 2.0, 5.0)

OOS_START = pd.Timestamp("2010-01-01")
CALIB_END = pd.Timestamp("2013-01-01")  # threshold fit on [OOS_START, CALIB_END)
EVAL_START = pd.Timestamp("2013-01-01")

BLEND = (0.35, 0.30, 0.25, 0.10)  # xgb, lgb, cat, ridge

LAG_COLS = [
    "Close_PctChg", "Volume_Log_Chg", "RSI_14", "MACD_Hist", "BB_PctB",
    "ADX_14", "DI_Plus_14", "DI_Minus_14",
    "HMM_Regime", "HMM_Prob_Bear", "HMM_Prob_Bull",
]
LAGS_DAILY = [1, 2, 3, 5, 10]
LAGS_WEEKLY = [1, 2, 3, 4]

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


def psection(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}", flush=True)


def normalize_index(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


# =========================================================================
# DATA
# =========================================================================

def fetch_daily_spy():
    cache = CACHE_DIR / "spy_daily_ohlcv.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return normalize_index(df)

    raw = yf.Ticker("SPY").history(start="2000-01-01", auto_adjust=False)
    if raw.empty:
        raise RuntimeError("No SPY daily data returned")
    df = pd.DataFrame(index=raw.index)
    df["Open"] = raw["Open"]
    df["High"] = raw["High"]
    df["Low"] = raw["Low"]
    df["Close"] = raw["Close"]
    df["Volume"] = raw["Volume"]
    if "Adj Close" in raw.columns:
        a2p = raw["Adj Close"] / raw["Close"]
        df["Open"] *= a2p
        df["High"] *= a2p
        df["Low"] *= a2p
        df["Close"] = raw["Adj Close"]
    df = normalize_index(df)
    df = df.dropna(subset=["Close"])
    df = df[df["Volume"] > 0]
    df.to_csv(cache)
    return df


def fetch_macro_daily():
    cache = CACHE_DIR / "macro_daily.csv"
    if cache.exists():
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return normalize_index(df)

    frames = {}
    for ticker, name in {
        "^VIX": "VIX",
        "^TNX": "TNX",
        "DX-Y.NYB": "DXY",
        "GC=F": "Gold",
        "BZ=F": "Brent",
    }.items():
        try:
            hist = yf.Ticker(ticker).history(start="2000-01-01", auto_adjust=False)
            if hist.empty or "Close" not in hist.columns:
                print(f"    {name}: no data", flush=True)
                continue
            s = hist["Close"].copy()
            s.index = pd.to_datetime(s.index)
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            s.index = s.index.normalize()
            s = s[~s.index.duplicated(keep="last")].sort_index()
            frames[f"{name}_Close"] = s
            print(f"    {name}: {len(s)} days", flush=True)
        except Exception as exc:
            print(f"    {name}: FAILED ({exc})", flush=True)
    df = pd.DataFrame(frames)
    df.to_csv(cache)
    return df


def to_weekly(daily):
    weekly = pd.DataFrame({
        "Open": daily["Open"].resample("W-FRI").first(),
        "High": daily["High"].resample("W-FRI").max(),
        "Low": daily["Low"].resample("W-FRI").min(),
        "Close": daily["Close"].resample("W-FRI").last(),
        "Volume": daily["Volume"].resample("W-FRI").sum(),
    }).dropna(subset=["Close"])
    weekly = weekly[weekly["Volume"] > 0]
    return weekly


def to_weekly_macro(macro_daily):
    return macro_daily.resample("W-FRI").last().dropna(how="all")


# =========================================================================
# HMM — causal forward filter
# =========================================================================

def _log_emit_diag(obs, means, covars):
    var = np.maximum(covars, 1e-6)
    diff = obs[:, None, :] - means[None, :, :]
    quad = np.sum(diff * diff / var[None, :, :], axis=2)
    logdet = np.sum(np.log(2.0 * np.pi * var), axis=1)
    return -0.5 * (quad + logdet)


def _forward_posteriors(log_start, log_trans, log_emit):
    n, k = log_emit.shape
    log_alpha = np.empty((n, k))
    log_alpha[0] = log_start + log_emit[0]
    for i in range(1, n):
        log_alpha[i] = log_emit[i] + logsumexp(log_alpha[i - 1][:, None] + log_trans, axis=0)
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha - log_norm)


def hmm_walk_forward(returns, volatility, min_fit, refit_every, n_states=3):
    """Regime at t is estimated from observations through t. The model itself
    was fit on observations strictly before the current refit segment."""
    ret = pd.Series(returns).astype(float)
    vol = pd.Series(volatility).astype(float).replace([np.inf, -np.inf], np.nan)
    vol = vol.ffill()
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = vol.fillna(vol.median() if np.isfinite(vol.median()) else 0.0)
    obs = np.column_stack([ret.values, vol.values])
    n = len(obs)

    regime = np.full(n, np.nan)
    probs = np.full((n, n_states), np.nan)

    t = int(min_fit)
    n_fits = 0
    while t < n:
        t_end = min(t + int(refit_every), n)
        train = obs[:t]
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        train_s = (train - mu) / sd
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=40,
                random_state=42,
                tol=0.05,
                min_covar=1e-3,
            )
            model.fit(train_s)
            n_fits += 1
        except Exception:
            t = t_end
            continue

        order = np.argsort(model.means_[:, 0])  # low return -> bear
        mapping = {int(raw): int(rank) for rank, raw in enumerate(order)}
        covars = np.asarray(model.covars_, dtype=float)
        if covars.ndim == 3:
            covars = np.diagonal(covars, axis1=1, axis2=2)
        seg = (obs[:t_end] - mu) / sd
        log_emit = _log_emit_diag(seg, model.means_, covars)
        log_start = np.log(np.clip(model.startprob_, 1e-12, None))
        log_trans = np.log(np.clip(model.transmat_, 1e-12, None))
        post = _forward_posteriors(log_start, log_trans, log_emit)
        for i in range(t, t_end):
            mapped = np.zeros(n_states)
            for raw, rank in mapping.items():
                mapped[rank] = post[i, raw]
            probs[i] = mapped
            regime[i] = int(np.argmax(mapped))
        t = t_end

    out = pd.DataFrame(index=returns.index)
    out["HMM_Regime"] = regime
    out["HMM_Prob_Bear"] = probs[:, 0]
    out["HMM_Prob_Sideways"] = probs[:, 1]
    out["HMM_Prob_Bull"] = probs[:, 2]
    out["HMM_Regime_Bear"] = (out["HMM_Regime"] == 0).astype(float)
    out["HMM_Regime_Sideways"] = (out["HMM_Regime"] == 1).astype(float)
    out["HMM_Regime_Bull"] = (out["HMM_Regime"] == 2).astype(float)
    dur = np.zeros(n, dtype=float)
    for i in range(1, n):
        if np.isnan(regime[i]):
            dur[i] = np.nan
        elif regime[i] == regime[i - 1]:
            dur[i] = (0 if np.isnan(dur[i - 1]) else dur[i - 1]) + 1
        else:
            dur[i] = 1
    out["HMM_Regime_Duration"] = dur
    out["HMM_Regime_Change"] = (out["HMM_Regime"].diff().abs() > 0).astype(float)
    # Leading NaNs stay NaN. Do not back-fill: that would leak the first fitted state.
    out["HMM_Regime"] = out["HMM_Regime"].ffill()
    for col in ["HMM_Prob_Bear", "HMM_Prob_Sideways", "HMM_Prob_Bull",
                "HMM_Regime_Bear", "HMM_Regime_Sideways", "HMM_Regime_Bull",
                "HMM_Regime_Duration"]:
        out[col] = out[col].ffill()
    print(f"    HMM fits: {n_fits}, valid bars: {int(np.isfinite(regime).sum())}/{n}", flush=True)
    return out


# =========================================================================
# FEATURES
# =========================================================================

def add_macro(df, macro):
    df = normalize_index(df)
    macro = normalize_index(macro)
    df = df.join(macro, how="left")
    for col in macro.columns:
        df[col] = df[col].ffill()
        name = col.replace("_Close", "")
        df[f"{name}_PctChg"] = df[col].pct_change() * 100
    return df


def build_base_features(df):
    df = df.copy()
    df["Volume_Log"] = np.log1p(df["Volume"].clip(lower=0))
    for col in ["Open", "High", "Low", "Close"]:
        df[f"{col}_PctChg"] = df[col].pct_change() * 100
    df["Volume_Log_Chg"] = df["Volume_Log"].diff()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = add_indicators(df, prefix="", close_col="Close", high_col="High",
                        low_col="Low", volume_col="Volume")
    if "SMA_50" in df.columns:
        df["Dist_SMA_50"] = df["Close"] / df["SMA_50"] - 1
    if "SMA_200" in df.columns:
        df["Dist_SMA_200"] = df["Close"] / df["SMA_200"] - 1
    if "SMA_50" in df.columns and "SMA_200" in df.columns:
        df["SMA_50_200"] = df["SMA_50"] / df["SMA_200"] - 1
        df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(float)
    if "SMA_200" in df.columns:
        df["Above_SMA_200"] = (df["Close"] > df["SMA_200"]).astype(float)
    if "DI_Plus_14" in df.columns and "DI_Minus_14" in df.columns:
        df["Trend_Bullish"] = (df["DI_Plus_14"] > df["DI_Minus_14"]).astype(float)
    if "VIX_Close" in df.columns:
        df["VIX_High_Regime"] = (df["VIX_Close"] > 25).astype(float)
        df["VIX_Extreme"] = (df["VIX_Close"] > 35).astype(float)
    return df


def add_hmm(df, is_daily):
    vol = df["VIX_Close"] if "VIX_Close" in df.columns else df["Close_PctChg"].rolling(20).std()
    if is_daily:
        hmm = hmm_walk_forward(df["Close_PctChg"], vol, min_fit=504, refit_every=63)
    else:
        hmm = hmm_walk_forward(df["Close_PctChg"], vol, min_fit=104, refit_every=26)
    hmm.index = df.index
    return pd.concat([df, hmm], axis=1)


def add_calendar(df, is_daily):
    idx = pd.to_datetime(df.index)
    df["Month"] = idx.month.astype(float)
    df["Quarter"] = idx.quarter.astype(float)
    df["WeekOfYear"] = idx.isocalendar().week.astype(int).astype(float)
    df["SellInMay"] = ((idx.month >= 5) & (idx.month <= 10)).astype(float)
    if is_daily:
        df["DayOfWeek"] = idx.dayofweek.astype(float)
        df["IsMonday"] = (idx.dayofweek == 0).astype(float)
        df["IsFriday"] = (idx.dayofweek == 4).astype(float)
    return df


def add_lags(df, lags):
    for col in LAG_COLS:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_Lag{lag}"] = df[col].shift(lag)
    windows = [5, 10, 20] if max(lags) >= 5 else [4, 8, 12]
    for window in windows:
        df[f"Close_PctChg_RollMean_{window}"] = df["Close_PctChg"].rolling(window).mean()
        df[f"Close_PctChg_RollStd_{window}"] = df["Close_PctChg"].rolling(window).std()
        if "VIX_Close" in df.columns:
            df[f"VIX_RollMean_{window}"] = df["VIX_Close"].rolling(window).mean()
    for a, b in INTERACTIONS:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]
    return df


LEVEL_PREFIXES = ("SMA_", "EMA_", "BB_Upper", "BB_Lower", "OBV")
LEVEL_EXACT = {
    "Open", "High", "Low", "Close", "Volume", "Volume_Log",
    "Gold_Close", "Brent_Close", "DXY_Close",
    "MACD", "MACD_Signal", "Target",
}


def feature_columns(df):
    cols = []
    for col in df.columns:
        if col in LEVEL_EXACT or col.startswith("W_Target"):
            continue
        if col.startswith("W_"):
            raw = col[2:]
            if raw in LEVEL_EXACT or raw.startswith(LEVEL_PREFIXES):
                continue
            if raw in {"MACD", "MACD_Signal"}:
                continue
        elif col.startswith(LEVEL_PREFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        cols.append(col)
    return cols


def impute(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill()
    return X.fillna(0.0)


def release_weekly_to_daily(weekly_features, daily_index):
    """Friday's completed weekly bar becomes visible the next calendar day."""
    released = weekly_features.copy()
    released.index = released.index + pd.Timedelta(days=1)
    released = released[~released.index.duplicated(keep="last")].sort_index()
    return released.reindex(daily_index, method="ffill")


# =========================================================================
# PIPELINES
# =========================================================================

def _finalize(df, is_daily, lags):
    df = df.replace([np.inf, -np.inf], np.nan)
    if "HMM_Regime" in df.columns:
        df = df[df["HMM_Regime"].notna()].copy()
    df["Target"] = df["Close"].pct_change().shift(-1) * 100
    df = add_lags(df, lags)
    df = df.dropna(subset=["Target"])
    cols = feature_columns(df)
    X = impute(df[cols].copy())
    y = df["Target"].copy()
    # Keep X and y on the same index after imputation (impute does not drop rows).
    mask = y.replace([np.inf, -np.inf], np.nan).notna()
    X = X.loc[mask]
    y = y.loc[mask]
    print(f"    rows={len(X)}  features={len(cols)}  "
          f"{X.index.min().date()} -> {X.index.max().date()}", flush=True)
    return X, y, cols


def build_weekly():
    psection("BUILD WEEKLY")
    daily = fetch_daily_spy()
    macro = to_weekly_macro(fetch_macro_daily())
    df = to_weekly(daily)
    print(f"    weekly bars: {len(df)}", flush=True)
    df = add_macro(df, macro)
    df = build_base_features(df)
    df = add_hmm(df, is_daily=False)
    df = add_calendar(df, is_daily=False)
    return _finalize(df, False, LAGS_WEEKLY)


def build_daily():
    psection("BUILD DAILY")
    df = fetch_daily_spy()
    macro = fetch_macro_daily()
    print(f"    daily bars: {len(df)}", flush=True)
    df = add_macro(df, macro)
    df = build_base_features(df)
    df = add_hmm(df, is_daily=True)
    df = add_calendar(df, is_daily=True)
    return _finalize(df, True, LAGS_DAILY)


def build_mixed():
    psection("BUILD MIXED")
    daily = fetch_daily_spy()
    macro_d = fetch_macro_daily()
    weekly = to_weekly(daily)
    macro_w = to_weekly_macro(macro_d)

    daily = add_macro(daily, macro_d)
    daily = build_base_features(daily)
    daily = add_hmm(daily, is_daily=True)
    daily = add_calendar(daily, is_daily=True)

    weekly = add_macro(weekly, macro_w)
    weekly = build_base_features(weekly)
    weekly = add_hmm(weekly, is_daily=False)
    week_close = weekly[["Close"]].copy()
    week_close.columns = ["_WeekCloseAudit"]

    drop_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in weekly.columns]
    weekly_feats = weekly.drop(columns=drop_cols)
    weekly_feats.columns = [f"W_{c}" for c in weekly_feats.columns]
    released = release_weekly_to_daily(weekly_feats, daily.index)
    audit = release_weekly_to_daily(week_close, daily.index)
    daily = daily.join(released, how="left")
    daily = daily.join(audit, how="left")
    _assert_weekly_release(daily, weekly)

    if "RSI_14" in daily.columns and "W_RSI_14" in daily.columns:
        daily["RSI_minus_Weekly"] = daily["RSI_14"] - daily["W_RSI_14"]
    if "Close_PctChg" in daily.columns and "W_Close_PctChg" in daily.columns:
        daily["Ret_minus_Weekly"] = daily["Close_PctChg"] - daily["W_Close_PctChg"]
    if "HMM_Prob_Bull" in daily.columns and "W_HMM_Prob_Bull" in daily.columns:
        daily["BullProb_minus_Weekly"] = daily["HMM_Prob_Bull"] - daily["W_HMM_Prob_Bull"]

    daily = daily.drop(columns=["_WeekCloseAudit"])
    return _finalize(daily, True, LAGS_DAILY)


def _assert_weekly_release(daily, weekly):
    """Friday still sees the previous week. The next session sees Friday's week."""
    fridays = weekly.index.intersection(daily.index)
    fridays = fridays[1:]
    checked = 0
    for friday in fridays[-30:]:
        loc = weekly.index.get_loc(friday)
        if isinstance(loc, slice) or not isinstance(loc, (int, np.integer)):
            continue
        prev_friday = weekly.index[loc - 1]
        later = daily.index[daily.index > friday]
        if len(later) == 0:
            continue
        nxt = later[0]
        got_fri = daily.at[friday, "_WeekCloseAudit"]
        got_nxt = daily.at[nxt, "_WeekCloseAudit"]
        if pd.isna(got_fri) or pd.isna(got_nxt):
            continue
        if not np.isclose(got_fri, weekly.at[prev_friday, "Close"], rtol=1e-6, atol=1e-4):
            raise AssertionError(
                f"Look-ahead: {friday.date()} weekly close audit {got_fri} "
                f"!= prior week {weekly.at[prev_friday, 'Close']}"
            )
        if not np.isclose(got_nxt, weekly.at[friday, "Close"], rtol=1e-6, atol=1e-4):
            raise AssertionError(
                f"Stale weekly feature: {nxt.date()} should see week {friday.date()} "
                f"close {weekly.at[friday, 'Close']} but has {got_nxt}"
            )
        checked += 1
    print(f"    weekly-release checks passed: {checked}", flush=True)
    if checked < 5:
        raise AssertionError("Weekly release alignment could not be verified")


# =========================================================================
# MODEL
# =========================================================================

def eval_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    dir_acc = ((y_true > 0) == (y_pred > 0)).mean() * 100
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 2 else np.nan
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "dir_accuracy": round(float(dir_acc), 2),
        "corr": round(float(corr), 4) if np.isfinite(corr) else None,
    }


def select_features(X_fit, y_fit, X_val, y_val, candidates, top_n):
    top_n = min(top_n, len(candidates))
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, eval_metric="rmse", early_stopping_rounds=20,
        random_state=42, n_jobs=4, verbosity=0,
    )
    model.fit(X_fit[candidates], y_fit, eval_set=[(X_val[candidates], y_val)], verbose=False)
    gain = model.get_booster().get_score(importance_type="gain")
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    chosen = [name for name, _ in ranked if name in candidates][:top_n]
    if len(chosen) < 8:
        chosen = list(candidates[:top_n])
    return chosen


def _fit_ensemble(X_fit, y_fit, X_val, y_val, X_te, cols):
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit[cols])
    X_val_s = scaler.transform(X_val[cols])
    X_te_s = scaler.transform(X_te[cols])

    xgb_m = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
        gamma=0.5, reg_alpha=0.5, reg_lambda=5.0,
        eval_metric="rmse", early_stopping_rounds=25,
        random_state=42, n_jobs=4, verbosity=0,
    )
    xgb_m.fit(X_fit[cols], y_fit, eval_set=[(X_val[cols], y_val)], verbose=False)

    lgb_m = lgb.LGBMRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=20,
        reg_alpha=0.5, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=4,
    )
    lgb_m.fit(
        X_fit[cols], y_fit, eval_set=[(X_val[cols], y_val)],
        callbacks=[lgb.early_stopping(25, verbose=False)],
    )

    cat_m = CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.04,
        l2_leaf_reg=5.0, random_seed=42, verbose=0,
        allow_writing_files=False,
    )
    cat_m.fit(X_fit[cols], y_fit, eval_set=(X_val[cols], y_val), early_stopping_rounds=25)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_fit_s, y_fit)

    pred = (
        BLEND[0] * xgb_m.predict(X_te[cols])
        + BLEND[1] * lgb_m.predict(X_te[cols])
        + BLEND[2] * cat_m.predict(X_te[cols])
        + BLEND[3] * ridge.predict(X_te_s)
    )
    return pred


def walk_forward(X, y, candidates, step, top_n):
    idx = X.index
    start = int(idx.searchsorted(OOS_START))
    n = len(X)
    if start < 200 or start >= n - 50:
        raise RuntimeError(f"Bad OOS start location {start} of {n}")

    parts = []
    picked = Counter()
    fold = 0
    cursor = start
    while cursor < n:
        end = min(cursor + step, n)
        n_val = max(40, int(cursor * 0.12))
        fit_end = cursor - n_val
        if fit_end < 150:
            raise RuntimeError("Training window too short for a validation tail")

        chosen = select_features(
            X.iloc[:fit_end], y.iloc[:fit_end],
            X.iloc[fit_end:cursor], y.iloc[fit_end:cursor],
            candidates, top_n,
        )
        picked.update(chosen)
        pred = _fit_ensemble(
            X.iloc[:fit_end], y.iloc[:fit_end],
            X.iloc[fit_end:cursor], y.iloc[fit_end:cursor],
            X.iloc[cursor:end], chosen,
        )
        parts.append(pd.Series(pred, index=idx[cursor:end], name="prediction"))
        fold += 1
        print(f"    fold {fold:02d}  test {idx[cursor].date()} -> {idx[end - 1].date()}  "
              f"bars={end - cursor}  features={len(chosen)}", flush=True)
        cursor = end

    predictions = pd.concat(parts)
    predictions = predictions[~predictions.index.duplicated(keep="last")].sort_index()
    print(f"    OOS predictions: {len(predictions)}", flush=True)
    return predictions, picked


# =========================================================================
# STRATEGIES
# =========================================================================

def positions_binary(predictions, threshold):
    return (predictions > threshold).astype(float)


def positions_scaled(predictions, scale):
    scale = max(float(scale), 1e-4)
    return (predictions / scale).clip(0.0, 1.0)


def positions_adaptive(predictions, actuals, lookback):
    preds = predictions.to_numpy()
    rets = actuals.to_numpy()
    out = np.zeros(len(preds))
    grid = np.arange(-0.40, 0.401, 0.02)
    for i in range(len(preds)):
        if i < lookback:
            out[i] = 1.0 if preds[i] > 0 else 0.0
            continue
        window_p = preds[i - lookback:i]
        window_r = rets[i - lookback:i] / 100.0
        best_t, best_ret = 0.0, -1e18
        for t in grid:
            pos = (window_p > t).astype(float)
            turnover = np.abs(np.diff(pos, prepend=pos[0])).sum()
            total = float((pos * window_r).sum() - turnover * (COST_BPS / 10000.0))
            if total > best_ret:
                best_t, best_ret = float(t), total
        out[i] = 1.0 if preds[i] > best_t else 0.0
    return pd.Series(out, index=predictions.index)


def fit_threshold(predictions, actuals):
    grid = np.arange(-0.50, 0.501, 0.005)
    rets = actuals.to_numpy() / 100.0
    preds = predictions.to_numpy()
    best_t, best_s = 0.0, -1e18
    for t in grid:
        pos = (preds > t).astype(float)
        turnover = np.abs(np.diff(pos, prepend=pos[0]))
        strat = pos * rets - turnover * (COST_BPS / 10000.0)
        vol = strat.std()
        sharpe = strat.mean() / vol if vol > 1e-12 else -1e18
        if sharpe > best_s:
            best_t, best_s = float(t), float(sharpe)
    return best_t


def score_window(positions, actuals, bars_per_year, cost_bps, start, end):
    rets = actuals.astype(float) / 100.0
    turnover = positions.diff().abs()
    turnover.iloc[0] = positions.iloc[0]
    strat = positions * rets - turnover * (cost_bps / 10000.0)

    window_strat = strat.loc[start:end]
    window_rets = rets.loc[start:end]
    window_pos = positions.loc[start:end]
    window_turn = turnover.loc[start:end]
    if len(window_strat) < 20:
        return None

    equity = (1.0 + window_strat).cumprod()
    bnh = (1.0 + window_rets).cumprod()
    n_years = max((window_strat.index[-1] - window_strat.index[0]).days / 365.25, 0.25)

    def _cagr(eq):
        return float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)

    def _sharpe(r):
        vol = float(r.std())
        return float(r.mean() / vol * np.sqrt(bars_per_year)) if vol > 1e-12 else 0.0

    def _maxdd(eq):
        dd = eq / eq.cummax() - 1.0
        return float(dd.min())

    active = window_pos > 0
    wins = ((window_strat > 0) & active).sum()
    return {
        "total_return": round(float(equity.iloc[-1] - 1.0) * 100, 2),
        "cagr": round(_cagr(equity) * 100, 2),
        "sharpe": round(_sharpe(window_strat), 3),
        "max_drawdown": round(_maxdd(equity) * 100, 2),
        "win_rate": round(float(wins / max(int(active.sum()), 1) * 100), 2),
        "in_market_pct": round(float(active.mean() * 100), 1),
        "n_trades": int((window_turn > 1e-6).sum()),
        "turnover_per_year": round(float(window_turn.sum() / n_years), 2),
        "n_bars": int(len(window_strat)),
        "n_years": round(float(n_years), 2),
        "bnh_total_return": round(float(bnh.iloc[-1] - 1.0) * 100, 2),
        "bnh_cagr": round(_cagr(bnh) * 100, 2),
        "bnh_sharpe": round(_sharpe(window_rets), 3),
        "bnh_max_drawdown": round(_maxdd(bnh) * 100, 2),
        "cost_bps": cost_bps,
    }, equity, bnh


# =========================================================================
# RUN
# =========================================================================

def run_mode(name, X, y, candidates, step, top_n, bars_per_year, lookback):
    psection(f"WALK-FORWARD {name.upper()}")
    t0 = time.time()
    predictions, picked = walk_forward(X, y, candidates, step, top_n)
    frame = pd.DataFrame({"prediction": predictions, "actual": y}).dropna().sort_index()
    frame = frame[frame.index >= OOS_START]
    calib = frame[(frame.index >= OOS_START) & (frame.index < CALIB_END)]
    eval_frame = frame[frame.index >= EVAL_START]
    print(f"    calib bars={len(calib)}  eval bars={len(eval_frame)}  "
          f"{eval_frame.index.min().date()} -> {eval_frame.index.max().date()}", flush=True)
    if len(calib) < 30 or len(eval_frame) < 50:
        raise RuntimeError(f"{name}: not enough calibration or evaluation bars")

    model_stats = eval_metrics(eval_frame["actual"], eval_frame["prediction"])
    print(f"    eval DirAcc={model_stats['dir_accuracy']:.2f}%  "
          f"corr={model_stats['corr']}  R2={model_stats['r2']}", flush=True)

    threshold = fit_threshold(calib["prediction"], calib["actual"])
    positive = calib["prediction"][calib["prediction"] > 0]
    scale = float(positive.quantile(0.90)) if len(positive) else float(calib["prediction"].abs().quantile(0.90))
    print(f"    frozen threshold={threshold:.3f}  scaled divisor={scale:.3f}", flush=True)

    # Adaptive history includes the calibration window, so the eval start is not cold.
    adaptive = positions_adaptive(frame["prediction"], frame["actual"], lookback)
    strategies = {
        "Binary (t=0)": positions_binary(frame["prediction"], 0.0),
        "Binary (t=opt)": positions_binary(frame["prediction"], threshold),
        "Scaled": positions_scaled(frame["prediction"], scale),
        "Adaptive": adaptive,
    }

    scored = {}
    equities = {}
    bnh_equity = None
    cost_sensitivity = {}
    for sname, pos in strategies.items():
        metrics, equity, bnh = score_window(
            pos, frame["actual"], bars_per_year, COST_BPS, EVAL_START, frame.index.max()
        )
        scored[sname] = metrics
        equities[sname] = equity
        bnh_equity = bnh
        print(f"    {sname:18s}  Sharpe={metrics['sharpe']:.3f}  CAGR={metrics['cagr']:.2f}%  "
              f"MaxDD={metrics['max_drawdown']:.1f}%  trades/yr context n={metrics['n_trades']}  "
              f"turnover/yr={metrics['turnover_per_year']:.1f}", flush=True)
        cost_sensitivity[sname] = {}
        for bps in COST_GRID:
            m_bps, _, _ = score_window(
                pos, frame["actual"], bars_per_year, bps, EVAL_START, frame.index.max()
            )
            cost_sensitivity[sname][str(bps)] = {
                "sharpe": m_bps["sharpe"],
                "cagr": m_bps["cagr"],
                "max_drawdown": m_bps["max_drawdown"],
            }

    bnh_m = scored["Binary (t=0)"]
    print(f"    {'Buy & Hold':18s}  Sharpe={bnh_m['bnh_sharpe']:.3f}  "
          f"CAGR={bnh_m['bnh_cagr']:.2f}%  MaxDD={bnh_m['bnh_max_drawdown']:.1f}%", flush=True)

    top_features = [{"feature": f, "folds": int(c)} for f, c in picked.most_common(25)]
    weekly_picks = sum(c for f, c in picked.items() if f.startswith("W_"))
    total_picks = sum(picked.values())
    result = {
        "mode": name,
        "model": model_stats,
        "threshold": round(threshold, 4),
        "scale": round(scale, 4),
        "strategies": scored,
        "cost_sensitivity": cost_sensitivity,
        "top_features": top_features,
        "weekly_feature_share": round(weekly_picks / max(total_picks, 1), 4),
        "n_candidates": len(candidates),
        "folds_feature_count": len(picked),
        "eval_start": str(eval_frame.index.min().date()),
        "eval_end": str(eval_frame.index.max().date()),
        "seconds": round(time.time() - t0, 1),
    }
    frame.to_csv(OUTPUT_DIR / f"{name}_predictions.csv")
    eq_df = pd.DataFrame({s: equities[s] for s in equities})
    eq_df["BuyHold"] = bnh_equity
    eq_df.to_csv(OUTPUT_DIR / f"{name}_equity.csv")
    with open(OUTPUT_DIR / f"{name}_summary.json", "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"    {name} done in {result['seconds'] / 60:.1f} min", flush=True)
    return result


def save_feature_sample(path, X, y):
    sample = X.copy()
    sample.insert(0, "Target", y)
    # Keep the file inspectable without writing every intermediate level column twice.
    sample.to_csv(path)
    print(f"    saved {path.name}: {sample.shape[0]} x {sample.shape[1]}", flush=True)


def plot_comparison(results):
    strat_list = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
    colors = {"weekly": "#42a5f5", "daily": "#ffa726", "mixed": "#26a69a"}
    strat_colors = {
        "Binary (t=0)": "#26a69a", "Binary (t=opt)": "#ab47bc",
        "Scaled": "#42a5f5", "Adaptive": "#ffa726",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#1e1e1e")
    fig.suptitle("SPY: Weekly vs Daily vs Mixed (evaluation window, 2 bps)",
                 color="white", fontsize=15)

    ax = axes[0][0]
    ax.set_facecolor("#1e1e1e")
    for res in results:
        eq = pd.read_csv(OUTPUT_DIR / f"{res['mode']}_equity.csv", index_col=0, parse_dates=True)
        best = max(strat_list, key=lambda s: res["strategies"][s]["sharpe"])
        series = eq[best]
        rebased = series / series.iloc[0]
        ax.plot(rebased.index, rebased.values, color=colors[res["mode"]], lw=1.4,
                label=f"{res['mode']} ({best})")
    weekly_eq = pd.read_csv(OUTPUT_DIR / "weekly_equity.csv", index_col=0, parse_dates=True)
    bnh = weekly_eq["BuyHold"] / weekly_eq["BuyHold"].iloc[0]
    ax.plot(bnh.index, bnh.values, color="#888888", lw=1.1, ls="--", label="Buy & Hold (weekly)")
    ax.set_title("Growth of $1, best strategy per mode", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#ccc")
    ax.grid(color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax = axes[0][1]
    ax.set_facecolor("#1e1e1e")
    x = np.arange(len(strat_list))
    width = 0.25
    for i, res in enumerate(results):
        vals = [res["strategies"][s]["sharpe"] for s in strat_list]
        ax.bar(x + i * width, vals, width, color=colors[res["mode"]], label=res["mode"], alpha=0.9)
    ax.axhline(results[0]["strategies"]["Binary (t=0)"]["bnh_sharpe"], color="#888", ls="--", lw=1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strat_list, color="#ccc", fontsize=8)
    ax.set_title("Annualized Sharpe", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#ccc")
    ax.grid(axis="y", color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax = axes[1][0]
    ax.set_facecolor("#1e1e1e")
    for i, res in enumerate(results):
        vals = [res["strategies"][s]["cagr"] for s in strat_list]
        ax.bar(x + i * width, vals, width, color=colors[res["mode"]], label=res["mode"], alpha=0.9)
    ax.axhline(results[0]["strategies"]["Binary (t=0)"]["bnh_cagr"], color="#888", ls="--", lw=1)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strat_list, color="#ccc", fontsize=8)
    ax.set_title("CAGR (%)", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#ccc")
    ax.grid(axis="y", color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")

    ax = axes[1][1]
    ax.set_facecolor("#1e1e1e")
    for i, res in enumerate(results):
        vals = [abs(res["strategies"][s]["max_drawdown"]) for s in strat_list]
        ax.bar(x + i * width, vals, width, color=colors[res["mode"]], label=res["mode"], alpha=0.9)
    ax.set_xticks(x + width)
    ax.set_xticklabels(strat_list, color="#ccc", fontsize=8)
    ax.set_title("Max drawdown (%)", color="white")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=8)
    ax.tick_params(colors="#ccc")
    ax.grid(axis="y", color="#333", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig2.patch.set_facecolor("#1e1e1e")
    for ax, res in zip(axes2, results):
        ax.set_facecolor("#1e1e1e")
        eq = pd.read_csv(OUTPUT_DIR / f"{res['mode']}_equity.csv", index_col=0, parse_dates=True)
        for sname in strat_list:
            series = eq[sname] / eq[sname].iloc[0]
            ax.plot(series.index, series.values, color=strat_colors[sname], lw=1.1, label=sname)
        hold = eq["BuyHold"] / eq["BuyHold"].iloc[0]
        ax.plot(hold.index, hold.values, color="#888", lw=1.0, ls="--", label="Buy & Hold")
        ax.set_title(res["mode"].upper(), color="white")
        ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="white", fontsize=7)
        ax.tick_params(colors="#ccc", labelsize=7)
        ax.grid(color="#333", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color("#444")
    fig2.suptitle("All strategy variants", color="white", fontsize=14)
    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "equity_by_mode.png", dpi=150, bbox_inches="tight",
                 facecolor=fig2.get_facecolor())
    plt.close()


def main():
    started = time.time()
    weekly_X, weekly_y, weekly_cols = build_weekly()
    save_feature_sample(OUTPUT_DIR / "spy_weekly_features.csv", weekly_X, weekly_y)
    daily_X, daily_y, daily_cols = build_daily()
    save_feature_sample(OUTPUT_DIR / "spy_daily_features.csv", daily_X, daily_y)
    mixed_X, mixed_y, mixed_cols = build_mixed()
    save_feature_sample(OUTPUT_DIR / "spy_mixed_features.csv", mixed_X, mixed_y)

    results = [
        run_mode("weekly", weekly_X, weekly_y, weekly_cols, step=26, top_n=40,
                 bars_per_year=52, lookback=26),
        run_mode("daily", daily_X, daily_y, daily_cols, step=126, top_n=50,
                 bars_per_year=252, lookback=126),
        run_mode("mixed", mixed_X, mixed_y, mixed_cols, step=126, top_n=60,
                 bars_per_year=252, lookback=126),
    ]
    plot_comparison(results)

    payload = {
        "oos_start": str(OOS_START.date()),
        "calibration": f"{OOS_START.date()} to {CALIB_END.date()} (threshold and scale only)",
        "evaluation": f"{EVAL_START.date()} onward",
        "cost_bps_headline": COST_BPS,
        "blend": {"xgboost": BLEND[0], "lightgbm": BLEND[1], "catboost": BLEND[2], "ridge": BLEND[3]},
        "modes": {r["mode"]: r for r in results},
        "seconds": round(time.time() - started, 1),
    }
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    psection("DONE")
    strat_list = ["Binary (t=0)", "Binary (t=opt)", "Scaled", "Adaptive"]
    print(f"  {'Mode':8s} {'Strategy':18s} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>8} {'Turn/yr':>8}")
    for res in results:
        for sname in strat_list:
            m = res["strategies"][sname]
            print(f"  {res['mode']:8s} {sname:18s} {m['sharpe']:7.3f} {m['cagr']:7.2f} "
                  f"{m['max_drawdown']:8.2f} {m['turnover_per_year']:8.1f}")
        b = res["strategies"]["Binary (t=0)"]
        print(f"  {res['mode']:8s} {'Buy & Hold':18s} {b['bnh_sharpe']:7.3f} {b['bnh_cagr']:7.2f} "
              f"{b['bnh_max_drawdown']:8.2f}")
    print(f"  Total minutes: {payload['seconds'] / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
