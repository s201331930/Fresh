"""
Fetch SPY weekly OHLCV data from 2000-01-01 to present.

Outputs:
  - data/spy_weekly.csv   : raw weekly bars
  - prints summary statistics and data quality report
"""

import os
import sys
from datetime import datetime

import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "spy_weekly.csv")

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
TICKER = "SPY"


def fetch_weekly_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download weekly OHLCV from Yahoo Finance and return a clean DataFrame."""
    spy = yf.Ticker(ticker)
    df = spy.history(start=start, end=end, interval="1wk", auto_adjust=False)

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} → {end})")

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = df.index.tz_localize(None)

    df.columns = ["open", "high", "low", "close", "volume"]

    df.sort_index(inplace=True)
    return df


def quality_report(df: pd.DataFrame) -> None:
    """Print a concise data-quality summary."""
    print("=" * 60)
    print(f"  SPY Weekly Data  |  {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    print("=" * 60)

    print("\n-- Missing values --")
    print(df.isnull().sum().to_string())

    print("\n-- Descriptive statistics --")
    print(df.describe().round(2).to_string())

    print("\n-- First 5 rows --")
    print(df.head().to_string())

    print("\n-- Last 5 rows --")
    print(df.tail().to_string())

    pct = df["close"].pct_change().dropna()
    print("\n-- Weekly return stats --")
    print(f"  Mean:   {pct.mean():.5f}")
    print(f"  Std:    {pct.std():.5f}")
    print(f"  Min:    {pct.min():.5f}")
    print(f"  Max:    {pct.max():.5f}")
    print(f"  Skew:   {pct.skew():.3f}")
    print(f"  Kurt:   {pct.kurtosis():.3f}")


def main() -> None:
    print(f"Fetching {TICKER} weekly data  {START_DATE} → {END_DATE} …")
    df = fetch_weekly_ohlcv(TICKER, START_DATE, END_DATE)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV)
    print(f"\nSaved {len(df):,} rows → {OUTPUT_CSV}")

    quality_report(df)


if __name__ == "__main__":
    main()
