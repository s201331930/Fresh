"""
Fetch macro variables (VIX, Brent Crude) and join them to SPY weekly data.

All series are resampled to the same weekly cadence (W-FRI) as SPY
before joining, so every row aligns by Week_Ending date.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from order_block_finder import detect_order_blocks
from technical_indicators import add_indicators

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SPY_CSV = DATA_DIR / "spy_weekly_ohlcv.csv"
OUTPUT_CSV = DATA_DIR / "spy_weekly_enriched.csv"

START_DATE = "2000-01-01"

MACRO_TICKERS = {
    "^VIX": {
        "name": "VIX",
        "agg": {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        },
        "keep_cols": {
            "Open": "VIX_Open",
            "High": "VIX_High",
            "Low": "VIX_Low",
            "Close": "VIX_Close",
        },
    },
    "BZ=F": {
        "name": "Brent",
        "agg": {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        },
        "keep_cols": {
            "Open": "Brent_Open",
            "High": "Brent_High",
            "Low": "Brent_Low",
            "Close": "Brent_Close",
        },
    },
}


def fetch_and_resample(ticker: str, config: dict) -> pd.DataFrame:
    """Fetch daily data and resample to weekly (W-FRI)."""
    name = config["name"]
    print(f"  Fetching {name} ({ticker}) ...")
    tk = yf.Ticker(ticker)
    daily = tk.history(start=START_DATE, interval="1d")

    if daily.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    print(f"    Daily rows: {len(daily)}, range: {daily.index.min().date()} -> {daily.index.max().date()}")

    weekly = daily.resample("W-FRI").agg(config["agg"])
    weekly.dropna(subset=["Open"], inplace=True)
    weekly.rename(columns=config["keep_cols"], inplace=True)
    weekly = weekly[list(config["keep_cols"].values())]
    weekly.index.name = "Week_Ending"

    print(f"    Weekly rows: {len(weekly)}")
    return weekly


def main():
    spy = pd.read_csv(SPY_CSV, index_col="Week_Ending", parse_dates=True)
    print(f"Loaded SPY: {len(spy)} weekly rows\n")

    macro_frames = {}
    for ticker, config in MACRO_TICKERS.items():
        macro_frames[config["name"]] = fetch_and_resample(ticker, config)
        print()

    merged = spy.copy()
    merged.index = pd.to_datetime(merged.index, utc=True).normalize().tz_localize(None)
    for name, frame in macro_frames.items():
        frame.index = frame.index.tz_localize(None).normalize()
        merged = merged.join(frame, how="left")
        matched = merged[list(frame.columns)].notna().all(axis=1).sum()
        total = len(merged)
        print(f"  {name}: {matched}/{total} weeks matched ({100*matched/total:.1f}%)")

    print(f"\n--- Merged dataset: {len(merged)} rows x {len(merged.columns)} columns ---")
    print(f"Columns: {list(merged.columns)}")

    print("\n--- First 5 rows ---")
    print(merged.head().to_string())
    print("\n--- Last 5 rows ---")
    print(merged.tail().to_string())

    # --- Add Order Block signals ---
    print("\nRunning Order Block detection (periods=5, threshold=0%) ...")
    merged = detect_order_blocks(merged, periods=5, threshold=0.0)
    ob_bull = int(merged["OB_Bull"].sum())
    ob_bear = int(merged["OB_Bear"].sum())
    print(f"  Bullish OBs: {ob_bull}, Bearish OBs: {ob_bear}")

    merged["Target"] = (merged["OB_Type"] == "Bullish").astype(int)

    # --- Add technical indicators ---
    print("\nComputing technical indicators ...")

    merged = add_indicators(
        merged, prefix="", close_col="Close", high_col="High",
        low_col="Low", volume_col="Volume",
    )
    print("  SPY: MA, EMA, BB, MACD, OBV, RSI, ADX/DI")

    merged = add_indicators(
        merged, prefix="VIX_", close_col="VIX_Close", high_col="VIX_High",
        low_col="VIX_Low", volume_col=None,
    )
    print("  VIX: MA, EMA, BB, MACD, RSI, ADX/DI")

    merged = add_indicators(
        merged, prefix="Brent_", close_col="Brent_Close", high_col="Brent_High",
        low_col="Brent_Low", volume_col=None,
    )
    print("  Brent: MA, EMA, BB, MACD, RSI, ADX/DI")

    # --- Log-transform Volume ---
    print("\nApplying log transform to Volume ...")
    merged["Volume_Log"] = np.log(merged["Volume"])

    # --- Compute week-over-week percentage change for base price columns only ---
    price_cols = [
        "Open", "Low", "High", "Close",
        "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close",
        "Brent_Open", "Brent_High", "Brent_Low", "Brent_Close",
    ]

    print(f"Computing week-over-week % change for: {price_cols}")
    for col in price_cols:
        pct_name = f"{col}_PctChg"
        merged[pct_name] = merged[col].pct_change() * 100

    # Volume change computed on the log-transformed series
    merged["Volume_Log_Chg"] = merged["Volume_Log"].diff()

    # Fill Brent pct-change NaNs with 0 (pre-2007 gap)
    brent_pct_cols = [c for c in merged.columns if c.startswith("Brent_") and c.endswith("_PctChg")]
    merged[brent_pct_cols] = merged[brent_pct_cols].fillna(0)

    # First row has no prior week — set all derived columns to 0
    derived_cols = [c for c in merged.columns if c.endswith("_PctChg") or c.endswith("_Log_Chg")]
    merged.loc[merged.index[0], derived_cols] = 0

    print(f"  Added {len(derived_cols)} derived columns (pct change + volume log change)")

    nulls = merged.isnull().sum()
    if nulls.any():
        print("\n--- Null counts per column ---")
        print(nulls[nulls > 0].to_string())

    merged.to_csv(OUTPUT_CSV, float_format="%.4f")
    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Final shape: {merged.shape[0]} rows x {merged.shape[1]} columns")
    print(f"Columns: {list(merged.columns)}")


if __name__ == "__main__":
    main()
