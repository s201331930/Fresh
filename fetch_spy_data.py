"""
Fetch SPY weekly OHLCV data from 2000-01-01 to present.
Stores the result as a CSV for downstream modeling.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = DATA_DIR / "spy_weekly_ohlcv.csv"

TICKER = "SPY"
START_DATE = "2000-01-01"
INTERVAL = "1wk"


def fetch_spy_weekly() -> pd.DataFrame:
    """Download SPY weekly data from Yahoo Finance."""
    spy = yf.Ticker(TICKER)
    df = spy.history(start=START_DATE, interval=INTERVAL, auto_adjust=False)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    df = df.dropna()

    return df


def main():
    print(f"Fetching {TICKER} weekly data from {START_DATE}...")
    df = fetch_spy_weekly()

    df.to_csv(OUTPUT_PATH)
    print(f"Saved {len(df)} weekly bars to {OUTPUT_PATH}")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    print(f"\nBasic statistics:\n{df.describe()}")


if __name__ == "__main__":
    main()
