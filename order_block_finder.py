"""
Order Block Finder — Python implementation of wugamlo's Pine Script indicator.

Original: https://www.tradingview.com/script/R8g2YHdg-Order-Block-Finder-Experimental/

Logic:
  An Order Block (OB) is the last opposing candle before a strong directional
  move of N consecutive same-direction candles.

  Bullish OB: last DOWN candle before N consecutive UP candles
    -> marks a demand zone (OB Low to OB High)
  Bearish OB: last UP candle before N consecutive DOWN candles
    -> marks a supply zone (OB Low to OB High)

  A minimum percentage move threshold can filter out weak setups.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
INPUT_CSV = DATA_DIR / "spy_weekly_ohlcv.csv"
OUTPUT_CSV = DATA_DIR / "spy_weekly_with_ob.csv"


def detect_order_blocks(
    df: pd.DataFrame,
    periods: int = 7,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Detect bullish and bearish Order Blocks.

    Parameters
    ----------
    df : DataFrame with Open, High, Low, Close columns.
    periods : number of consecutive same-direction candles required
              after the OB candle.
    threshold : minimum absolute percent move from the OB candle's close
                to the last candle in the sequence (close[1] in Pine terms).

    Returns the input DataFrame augmented with OB columns.
    """
    n = len(df)
    ob_period = periods + 1

    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values

    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    ob_high = np.full(n, np.nan)
    ob_low = np.full(n, np.nan)
    ob_mid = np.full(n, np.nan)
    ob_type = np.full(n, "", dtype=object)

    for i in range(ob_period, n):
        ob_idx = i - ob_period

        absmove = abs(closes[ob_idx] - closes[i - 1]) / closes[ob_idx] * 100
        relmove = absmove >= threshold

        # --- Bullish OB: OB candle is a down candle, followed by `periods` up candles ---
        if closes[ob_idx] < opens[ob_idx]:
            up_count = 0
            for j in range(1, periods + 1):
                if closes[i - j] > opens[i - j]:
                    up_count += 1
            if up_count == periods and relmove:
                bull_ob[i] = True
                ob_high[i] = highs[ob_idx]
                ob_low[i] = lows[ob_idx]
                ob_mid[i] = (highs[ob_idx] + lows[ob_idx]) / 2
                ob_type[i] = "Bullish"

        # --- Bearish OB: OB candle is an up candle, followed by `periods` down candles ---
        if closes[ob_idx] > opens[ob_idx]:
            down_count = 0
            for j in range(1, periods + 1):
                if closes[i - j] < opens[i - j]:
                    down_count += 1
            if down_count == periods and relmove:
                bear_ob[i] = True
                ob_high[i] = highs[ob_idx]
                ob_low[i] = lows[ob_idx]
                ob_mid[i] = (highs[ob_idx] + lows[ob_idx]) / 2
                ob_type[i] = "Bearish"

    result = df.copy()
    result["OB_Bull"] = bull_ob.astype(int)
    result["OB_Bear"] = bear_ob.astype(int)
    result["OB_Type"] = ob_type
    result["OB_High"] = ob_high
    result["OB_Low"] = ob_low
    result["OB_Mid"] = ob_mid

    return result


def print_section(title: str):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    df = pd.read_csv(INPUT_CSV, index_col="Week_Ending", parse_dates=True)
    print(f"Loaded {len(df)} weekly bars")

    periods = 7
    threshold = 0.0
    print(f"Parameters: periods={periods}, threshold={threshold}%")

    result = detect_order_blocks(df, periods=periods, threshold=threshold)

    bull_count = result["OB_Bull"].sum()
    bear_count = result["OB_Bear"].sum()

    print_section("ORDER BLOCK SUMMARY")
    print(f"  Bullish Order Blocks: {bull_count}")
    print(f"  Bearish Order Blocks: {bear_count}")
    print(f"  Total:                {bull_count + bear_count}")

    # --- Bullish OBs ---
    if bull_count > 0:
        print_section("BULLISH ORDER BLOCKS (demand zones)")
        bull_rows = result[result["OB_Bull"] == 1][
            ["Close", "OB_High", "OB_Low", "OB_Mid"]
        ]
        print(bull_rows.to_string())

    # --- Bearish OBs ---
    if bear_count > 0:
        print_section("BEARISH ORDER BLOCKS (supply zones)")
        bear_rows = result[result["OB_Bear"] == 1][
            ["Close", "OB_High", "OB_Low", "OB_Mid"]
        ]
        print(bear_rows.to_string())

    # --- Distribution of OBs over time ---
    print_section("OB FREQUENCY BY YEAR")
    result["Year"] = pd.to_datetime(result.index, utc=True).year
    yearly = result.groupby("Year")[["OB_Bull", "OB_Bear"]].sum()
    yearly["Total"] = yearly["OB_Bull"] + yearly["OB_Bear"]
    print(yearly.to_string())
    result.drop(columns=["Year"], inplace=True)

    result.to_csv(OUTPUT_CSV, float_format="%.4f")
    print(f"\nSaved enriched dataset to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
