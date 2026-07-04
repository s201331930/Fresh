"""
Technical indicators computed from OHLCV data.

All functions operate on pandas Series/DataFrames and return Series
that can be directly assigned as new columns.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Moving Averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# 2. Bollinger Bands (20-period, 2 std)
# ---------------------------------------------------------------------------

def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid * 100
    pct_b = (close - lower) / (upper - lower)
    return mid, upper, lower, width, pct_b


# ---------------------------------------------------------------------------
# 3. MACD (12, 26, 9)
# ---------------------------------------------------------------------------

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# 4. OBV (On Balance Volume) — requires close + volume
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# 5. RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# 6. ADX & DI (Average Directional Index)
# ---------------------------------------------------------------------------

def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return adx_val, plus_di, minus_di


# ---------------------------------------------------------------------------
# Apply all indicators to a single instrument
# ---------------------------------------------------------------------------

MA_PERIODS = [5, 10, 20, 50, 100, 200]


def add_indicators(
    df: pd.DataFrame,
    prefix: str,
    close_col: str,
    high_col: str,
    low_col: str,
    volume_col: str | None = None,
) -> pd.DataFrame:
    """
    Add all technical indicators for one instrument.

    Parameters
    ----------
    df : the master DataFrame (modified in place and returned)
    prefix : column name prefix, e.g. "" for SPY, "VIX_", "Brent_"
    close_col, high_col, low_col : column names for OHLC
    volume_col : column name for volume (None if unavailable)
    """
    c = df[close_col]
    h = df[high_col]
    lo = df[low_col]
    p = prefix

    # 1. Moving Averages
    for period in MA_PERIODS:
        df[f"{p}SMA_{period}"] = sma(c, period)
        df[f"{p}EMA_{period}"] = ema(c, period)

    # 2. Bollinger Bands
    bb_mid, bb_upper, bb_lower, bb_width, bb_pctb = bollinger_bands(c)
    df[f"{p}BB_Upper"] = bb_upper
    df[f"{p}BB_Lower"] = bb_lower
    df[f"{p}BB_Width"] = bb_width
    df[f"{p}BB_PctB"] = bb_pctb

    # 3. MACD
    macd_line, signal_line, histogram = macd(c)
    df[f"{p}MACD"] = macd_line
    df[f"{p}MACD_Signal"] = signal_line
    df[f"{p}MACD_Hist"] = histogram

    # 4. OBV (only if volume is available)
    if volume_col is not None:
        df[f"{p}OBV"] = obv(c, df[volume_col])

    # 5. RSI
    df[f"{p}RSI_14"] = rsi(c)

    # 6. ADX & DI
    adx_val, plus_di, minus_di = adx_di(h, lo, c)
    df[f"{p}ADX_14"] = adx_val
    df[f"{p}DI_Plus_14"] = plus_di
    df[f"{p}DI_Minus_14"] = minus_di

    return df
