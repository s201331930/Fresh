"""
Fetch SPY weekly OHLCV data from 1/1/2000 to present.

Downloads daily data from Yahoo Finance and resamples to weekly bars
to ensure accurate OHLCV aggregation (yfinance's native weekly bars
can have calendar-alignment quirks).
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = DATA_DIR / "spy_weekly_ohlcv.csv"

TICKER = "SPY"
START_DATE = "2000-01-01"


def fetch_daily_data(ticker: str, start: str) -> pd.DataFrame:
    """Pull daily OHLCV from Yahoo Finance."""
    tk = yf.Ticker(ticker)
    df = tk.history(start=start, interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    return df


def resample_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily bars into weekly bars (Mon-Fri week ending Friday).

    Aggregation rules mirror market convention:
      Open      -> first trading day of the week
      High      -> highest intra-week high
      Low       -> lowest intra-week low
      Close     -> last trading day of the week (raw)
      Adj Close -> last trading day of the week (split & dividend adjusted)
      Volume    -> sum of daily volumes
    """
    weekly = daily.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Adj Close": "last",
            "Volume": "sum",
        }
    )
    weekly.dropna(subset=["Open"], inplace=True)
    weekly["A2P"] = weekly["Adj Close"] / weekly["Close"]
    weekly["Adj Open"] = weekly["Open"] * weekly["A2P"]
    weekly["Adj High"] = weekly["High"] * weekly["A2P"]
    weekly["Adj Low"] = weekly["Low"] * weekly["A2P"]
    weekly.drop(columns=["Open", "High", "Low", "Close"], inplace=True)
    weekly.index.name = "Week_Ending"
    return weekly


def main():
    print(f"Fetching daily {TICKER} data from {START_DATE} ...")
    daily = fetch_daily_data(TICKER, START_DATE)
    print(f"  Daily rows retrieved: {len(daily)}")
    print(f"  Date range: {daily.index.min().date()} -> {daily.index.max().date()}")

    print("Resampling to weekly bars (Mon-Fri, week ending Friday) ...")
    weekly = resample_to_weekly(daily)
    print(f"  Weekly rows: {len(weekly)}")
    print(f"  Date range: {weekly.index.min().date()} -> {weekly.index.max().date()}")

    weekly.to_csv(OUTPUT_CSV, float_format="%.4f")
    print(f"\nSaved to {OUTPUT_CSV}")

    print("\n--- Data snapshot (first 5 rows) ---")
    print(weekly.head().to_string())
    print("\n--- Data snapshot (last 5 rows) ---")
    print(weekly.tail().to_string())

    print("\n--- Summary statistics ---")
    print(weekly.describe().to_string())

    nulls = weekly.isnull().sum()
    if nulls.any():
        print("\n⚠  Null values detected:")
        print(nulls[nulls > 0])
    else:
        print("\nNo null values — data is clean.")


if __name__ == "__main__":
    main()
