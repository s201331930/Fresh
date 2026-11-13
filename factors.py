"""Factor construction. No strategy and no future label.

Indicators, lags, macro changes, calendar fields, and a causal HMM regime
are known at the close of the bar that produces them. A completed weekly
bar is released on the next calendar day, so the week-ending session still
sees the prior week.

The stored tables in factors/ were built with this recipe. Their old
next-bar target column has been removed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from technical_indicators import add_indicators

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

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
MACRO_TICKERS = {
    "^VIX": "VIX",
    "^TNX": "TNX",
    "DX-Y.NYB": "DXY",
    "GC=F": "Gold",
    "BZ=F": "Brent",
}


def normalize_index(df):
    df = df.copy()
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_ohlcv(ticker, cache_name=None):
    """Adjusted OHLC, volume greater than zero. Cached under data/cache."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_name = cache_name or f"{ticker.replace('=', '_').replace('.', '_')}_ohlcv.csv"
    cache = CACHE_DIR / cache_name
    if cache.exists():
        return normalize_index(pd.read_csv(cache, index_col=0, parse_dates=True))

    raw = yf.Ticker(ticker).history(start="2000-01-01", auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"No daily data returned for {ticker}")
    df = pd.DataFrame(index=raw.index)
    df["Open"] = raw["Open"]
    df["High"] = raw["High"]
    df["Low"] = raw["Low"]
    df["Close"] = raw["Close"]
    df["Volume"] = raw["Volume"]
    if "Adj Close" in raw.columns:
        factor = raw["Adj Close"] / raw["Close"].replace(0, np.nan)
        for col in ["Open", "High", "Low"]:
            df[col] = df[col] * factor
        df["Close"] = raw["Adj Close"]
    df = normalize_index(df).dropna(subset=["Close"])
    df = df[df["Volume"] > 0]
    df.to_csv(cache)
    return df


def fetch_macro_daily():
    """VIX, 10-year yield, dollar index, gold, and Brent closes."""
    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "macro_daily.csv"
    if cache.exists():
        return normalize_index(pd.read_csv(cache, index_col=0, parse_dates=True))

    frames = {}
    for ticker, name in MACRO_TICKERS.items():
        hist = yf.Ticker(ticker).history(start="2000-01-01", auto_adjust=False)
        if hist.empty or "Close" not in hist.columns:
            continue
        series = hist["Close"].copy()
        series.index = pd.to_datetime(series.index)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.index = series.index.normalize()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        frames[f"{name}_Close"] = series
    df = pd.DataFrame(frames)
    df.to_csv(cache)
    return df


def to_weekly(daily, rule="W-FRI"):
    weekly = pd.DataFrame({
        "Open": daily["Open"].resample(rule).first(),
        "High": daily["High"].resample(rule).max(),
        "Low": daily["Low"].resample(rule).min(),
        "Close": daily["Close"].resample(rule).last(),
        "Volume": daily["Volume"].resample(rule).sum(),
    }).dropna(subset=["Close"])
    return weekly[weekly["Volume"] > 0]


def to_weekly_macro(macro_daily, rule="W-FRI"):
    return macro_daily.resample(rule).last().dropna(how="all")


def _log_emit_diag(obs, means, covars):
    var = np.maximum(covars, 1e-6)
    diff = obs[:, None, :] - means[None, :, :]
    quad = np.sum(diff * diff / var[None, :, :], axis=2)
    logdet = np.sum(np.log(2.0 * np.pi * var), axis=1)
    return -0.5 * (quad + logdet)


def _forward_posteriors(startprob, transmat, log_emit):
    """Filtered state probabilities. The backward pass is not used."""
    from hmmlearn import _hmmc

    _, fwd = _hmmc.forward_log(
        np.ascontiguousarray(startprob, dtype=np.float64),
        np.ascontiguousarray(transmat, dtype=np.float64),
        np.ascontiguousarray(log_emit, dtype=np.float64),
    )
    row_max = np.max(fwd, axis=1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    post = np.exp(fwd - row_max)
    denom = post.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0.0, denom, 1.0)
    return post / denom


def hmm_walk_forward(returns, volatility, min_fit, refit_every, n_states=3):
    """Regime at t uses observations through t. The fit uses only earlier bars."""
    from hmmlearn.hmm import GaussianHMM

    ret = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = pd.Series(volatility).astype(float).replace([np.inf, -np.inf], np.nan).ffill()
    vol = vol.fillna(vol.median() if np.isfinite(vol.median()) else 0.0)
    obs = np.column_stack([ret.values, vol.values])
    n = len(obs)
    regime = np.full(n, np.nan)
    probs = np.full((n, n_states), np.nan)

    t = int(min_fit)
    while t < n:
        t_end = min(t + int(refit_every), n)
        train = obs[:t]
        mu = train.mean(axis=0)
        sd = train.std(axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        try:
            model = GaussianHMM(
                n_components=n_states, covariance_type="diag", n_iter=40,
                random_state=42, tol=0.05, min_covar=1e-3,
            )
            model.fit((train - mu) / sd)
        except Exception:
            t = t_end
            continue
        order = np.argsort(model.means_[:, 0])
        mapping = {int(raw): int(rank) for rank, raw in enumerate(order)}
        covars = np.asarray(model.covars_, dtype=float)
        if covars.ndim == 3:
            covars = np.diagonal(covars, axis1=1, axis2=2)
        seg = (obs[:t_end] - mu) / sd
        log_emit = _log_emit_diag(seg, model.means_, covars)
        post = _forward_posteriors(model.startprob_, model.transmat_, log_emit)
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
    out["HMM_Regime"] = out["HMM_Regime"].ffill()
    for col in ["HMM_Prob_Bear", "HMM_Prob_Sideways", "HMM_Prob_Bull",
                "HMM_Regime_Bear", "HMM_Regime_Sideways", "HMM_Regime_Bull",
                "HMM_Regime_Duration"]:
        out[col] = out[col].ffill()
    return out


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
    for left, right in INTERACTIONS:
        if left in df.columns and right in df.columns:
            df[f"{left}_x_{right}"] = df[left] * df[right]
    return df


LEVEL_PREFIXES = ("SMA_", "EMA_", "BB_Upper", "BB_Lower", "OBV")
LEVEL_EXACT = {
    "Open", "High", "Low", "Close", "Volume", "Volume_Log",
    "Gold_Close", "Brent_Close", "DXY_Close",
    "MACD", "MACD_Signal",
}


def feature_columns(df):
    cols = []
    for col in df.columns:
        if col in LEVEL_EXACT or "Target" in col:
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


def impute(frame):
    frame = frame.replace([np.inf, -np.inf], np.nan).ffill()
    return frame.fillna(0.0)


def release_weekly_to_daily(weekly_features, daily_index):
    """Friday's completed weekly bar becomes visible the next calendar day."""
    released = weekly_features.copy()
    released.index = released.index + pd.Timedelta(days=1)
    released = released[~released.index.duplicated(keep="last")].sort_index()
    return released.reindex(daily_index, method="ffill")


def _assert_weekly_release(daily, weekly):
    fridays = weekly.index.intersection(daily.index)[1:]
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
            raise AssertionError(f"Look-ahead: {friday.date()} sees its own week")
        if not np.isclose(got_nxt, weekly.at[friday, "Close"], rtol=1e-6, atol=1e-4):
            raise AssertionError(f"Stale weekly factor on {nxt.date()}")
        checked += 1
    if checked < 5:
        raise AssertionError("Weekly release alignment could not be verified")


def _matrix(df, lags):
    df = df.replace([np.inf, -np.inf], np.nan)
    if "HMM_Regime" in df.columns:
        df = df[df["HMM_Regime"].notna()].copy()
    df = add_lags(df, lags)
    cols = feature_columns(df)
    return impute(df[cols].copy())


def _weekly_table(daily, macro_daily, week_rule):
    df = to_weekly(daily, week_rule)
    df = add_macro(df, to_weekly_macro(macro_daily, week_rule))
    df = build_base_features(df)
    df = add_hmm(df, is_daily=False)
    return add_calendar(df, is_daily=False)


def _daily_table(daily, macro_daily):
    df = add_macro(daily, macro_daily)
    df = build_base_features(df)
    df = add_hmm(df, is_daily=True)
    return add_calendar(df, is_daily=True)


def _attach_weekly(daily_df, weekly_df):
    daily = daily_df.copy()
    weekly = weekly_df
    week_close = weekly[["Close"]].copy()
    week_close.columns = ["_WeekCloseAudit"]
    drop_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in weekly.columns]
    weekly_feats = weekly.drop(columns=drop_cols)
    weekly_feats.columns = [f"W_{c}" for c in weekly_feats.columns]
    daily = daily.join(release_weekly_to_daily(weekly_feats, daily.index), how="left")
    daily = daily.join(release_weekly_to_daily(week_close, daily.index), how="left")
    _assert_weekly_release(daily, weekly)
    if "RSI_14" in daily.columns and "W_RSI_14" in daily.columns:
        daily["RSI_minus_Weekly"] = daily["RSI_14"] - daily["W_RSI_14"]
    if "Close_PctChg" in daily.columns and "W_Close_PctChg" in daily.columns:
        daily["Ret_minus_Weekly"] = daily["Close_PctChg"] - daily["W_Close_PctChg"]
    if "HMM_Prob_Bull" in daily.columns and "W_HMM_Prob_Bull" in daily.columns:
        daily["BullProb_minus_Weekly"] = daily["HMM_Prob_Bull"] - daily["W_HMM_Prob_Bull"]
    return daily.drop(columns=["_WeekCloseAudit"])


def build_weekly_factors(daily, macro_daily, week_rule="W-FRI"):
    return _matrix(_weekly_table(daily, macro_daily, week_rule), LAGS_WEEKLY)


def build_daily_factors(daily, macro_daily):
    return _matrix(_daily_table(daily, macro_daily), LAGS_DAILY)


def build_mixed_factors(daily, macro_daily, week_rule="W-FRI"):
    weekly = _weekly_table(daily, macro_daily, week_rule)
    daily_df = _daily_table(daily, macro_daily)
    return _matrix(_attach_weekly(daily_df, weekly), LAGS_DAILY)
