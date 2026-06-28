"""
Fetch SPY weekly OHLCV data from Yahoo Finance.

Downloads weekly-resampled data from 2000-01-01 to present, persists as
both CSV and Parquet for downstream modelling, and prints a concise
data-quality report.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2000-01-01"
TICKER = "SPY"


def fetch_weekly_ohlcv(ticker: str = TICKER, start: str = START_DATE) -> pd.DataFrame:
    """Download weekly OHLCV bars for *ticker* starting from *start*."""
    spy = yf.Ticker(ticker)

    # Pull daily data first — yfinance weekly resampling can have quirks
    # around partial weeks and date alignment, so we resample ourselves
    # to guarantee Monday-anchored ISO weeks.
    daily = spy.history(start=start, interval="1d", auto_adjust=True)

    if daily.empty:
        raise RuntimeError(f"No data returned for {ticker} from {start}")

    daily.index = pd.to_datetime(daily.index).tz_localize(None)

    # Resample to weekly (week ending Friday, standard US equity convention)
    weekly = (
        daily
        .resample("W-FRI")
        .agg({
            "Open":   "first",
            "High":   "max",
            "Low":    "min",
            "Close":  "last",
            "Volume": "sum",
        })
        .dropna(subset=["Open", "Close"])
    )

    weekly.index.name = "Date"
    weekly = weekly.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    return weekly


def quality_report(df: pd.DataFrame) -> None:
    """Print a compact data-quality summary to stdout."""
    print("=" * 60)
    print(f"SPY Weekly OHLCV — Data Quality Report")
    print("=" * 60)
    print(f"Date range     : {df.index.min().date()} → {df.index.max().date()}")
    print(f"Total bars     : {len(df):,}")
    print(f"Columns        : {list(df.columns)}")
    print()

    # Null / inf check
    nulls = df.isnull().sum()
    infs = np.isinf(df.select_dtypes(include=[np.number])).sum()
    print("Null counts:")
    print(nulls.to_string())
    print()
    print("Inf counts:")
    print(infs.to_string())
    print()

    # Basic sanity: high >= low, high >= open/close, low <= open/close
    bad_hl = (df["high"] < df["low"]).sum()
    bad_ho = (df["high"] < df["open"]).sum()
    bad_hc = (df["high"] < df["close"]).sum()
    bad_lo = (df["low"] > df["open"]).sum()
    bad_lc = (df["low"] > df["close"]).sum()
    neg_vol = (df["volume"] <= 0).sum()

    print("OHLC consistency checks (should all be 0):")
    print(f"  high < low    : {bad_hl}")
    print(f"  high < open   : {bad_ho}")
    print(f"  high < close  : {bad_hc}")
    print(f"  low  > open   : {bad_lo}")
    print(f"  low  > close  : {bad_lc}")
    print(f"  volume <= 0   : {neg_vol}")
    print()

    # Summary stats
    print("Descriptive statistics:")
    print(df.describe().round(2).to_string())
    print()

    # Head / tail
    print("First 5 bars:")
    print(df.head().to_string())
    print()
    print("Last 5 bars:")
    print(df.tail().to_string())
    print("=" * 60)


def main() -> None:
    print(f"Fetching {TICKER} weekly data from {START_DATE} …")
    df = fetch_weekly_ohlcv()

    csv_path = DATA_DIR / "spy_weekly_ohlcv.csv"
    parquet_path = DATA_DIR / "spy_weekly_ohlcv.parquet"

    df.to_csv(csv_path)
    df.to_parquet(parquet_path, engine="pyarrow")

    print(f"Saved CSV     → {csv_path}")
    print(f"Saved Parquet → {parquet_path}")
    print()

    quality_report(df)


if __name__ == "__main__":
    main()
