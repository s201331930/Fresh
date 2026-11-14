# Data catalog

Ten files. Three kinds of thing: who can be traded, the SPY weekly prices, and the factor tables already built from those prices. Nothing here is a strategy result.

Prices are Yahoo Finance, split- and dividend-adjusted. Factor changes such as `Close_PctChg` are in percent. Distances such as `Dist_SMA_200` are ratios (`close / average - 1`), not percent.

## Files

| File | Grain | Rows | Columns | Span | Size |
|---|---|---:|---:|---|---:|
| `universes/global.json` | one asset | 102 | 5 fields | membership list | 14 KB |
| `universes/tasi.json` | one Tadawul name | 252 | 5 fields | list dated 24 Oct 2026 | 41 KB |
| `data/spy_weekly_ohlcv.csv` | week ending Friday | 1,382 | 7 | 7 Jan 2000 – 26 Jun 2026 | 104 KB |
| `data/spy_weekly_with_ob.csv` | same weeks | 1,382 | 13 | same span | 117 KB |
| `data/spy_weekly_enriched.csv` | same weeks | 1,382 | 106 | same span | 1.0 MB |
| `data/spy_weekly_enriched.xlsx` | same table as the CSV | 1,382 | 106 | same span | 916 KB |
| `factors/spy_daily_features.csv` | US session | 6,238 | 117 | 8 Jan 2002 – 22 Oct 2026 | 11.6 MB |
| `factors/spy_weekly_features.csv` | week | 1,294 | 103 | 4 Jan 2002 – 16 Oct 2026 | 2.1 MB |
| `factors/spy_mixed_features.csv` | US session | 6,238 | 155 | 8 Jan 2002 – 22 Oct 2026 | 15.0 MB |
| `factors/asset_descriptors.csv` | one asset, a single summary | 102 | 25 | the 102-name weekly history | 16 KB |

The weekly price files stop on 26 Jun 2026. The factor matrices were built from a later pull and run through 22 Oct 2026. They are not the same vintage.

## Universes

`universes/global.json` is `{ n, assets[] }`. Each asset has `symbol`, `name`, `label`, `market`, `asset_class`.

| Class | Names | Market |
|---|---:|---|
| US Stock | 20 | US |
| Intl ETF | 13 | US |
| TASI Stock | 12 | TASI |
| Sector ETF | 9 | US |
| Commodity | 9 | US |
| Bond ETF | 9 | US |
| Currency | 8 | US |
| Intl Stock | 8 | US |
| Broad ETF | 5 | US |
| Thematic ETF | 5 | US |
| REIT | 4 | US |

Ninety names are US-listed. Twelve are Tadawul names inside this 102 (`1120.SR` and the others). Currency names are currency ETFs, not spot FX. Commodity names are commodity ETFs.

`universes/tasi.json` is the All Share list from whichindex.com on 24 Oct 2026. Each asset has `ticker` (the four-digit code), `yahoo` (`1120.SR`), `name`, `index_rank`, and `in_original_12`. Twelve names are flagged as the original set. This list is the membership. It does not contain prices.

## SPY weekly prices

`data/spy_weekly_ohlcv.csv`

| Column | Content |
|---|---|
| `Week_Ending` | Friday timestamp, US Eastern offset |
| `Open`, `High`, `Low`, `Close` | Adjusted dollars |
| `A2P` | Adjustment factor, adjusted close divided by raw close. It is 1.0 on the last row. |
| `Volume` | Shares for the week |

Last row: week of 26 Jun 2026, close 728.99, volume 295,781,600.

`data/spy_weekly_with_ob.csv` is that table plus order-block fields. An order block is the last opposing candle before a run of candles in one direction.

| Column | Content |
|---|---|
| `OB_Bull` | 1 on a bullish block, else 0 |
| `OB_Bear` | 1 on a bearish block, else 0 |
| `OB_Type` | `Bullish`, `Bearish`, or blank |
| `OB_High`, `OB_Low`, `OB_Mid` | Zone prices, blank when there is no block |

41 weeks are bullish blocks. 10 are bearish. The other 1,331 weeks are blank.

`data/spy_weekly_enriched.csv` (and the xlsx twin) stacks four groups on the same 1,382 weeks:

1. The price columns above, including the order-block fields.
2. VIX and Brent open, high, low, close. Brent is empty on 396 of 1,382 weeks.
3. SMA and EMA at 5, 10, 20, 50, 100, and 200 sessions, for SPY, VIX, and Brent, plus Bollinger, MACD, OBV, RSI(14), ADX, and DI for each of the three.
4. Log volume and percent changes of the price, VIX, and Brent fields.

This file still contains a `Target` column, the next week's return from the old join. The factor matrices do not. Do not use `Target` as a factor.

## Factor matrices

All three use the same recipe. A value on a row is known at that bar's close. Percent changes are already multiplied by 100. Ratios are not. Empty cells were filled forward and then with zero when the matrix was written, so these tables have no blank factor cells.

Shared families, present on the daily and the weekly matrix:

| Family | Columns | Unit |
|---|---|---|
| Macro level | `VIX_Close`, `TNX_Close` | index points, and the 10-year yield in percent |
| Macro change | `VIX_PctChg`, `TNX_PctChg`, `DXY_PctChg`, `Gold_PctChg`, `Brent_PctChg` | percent |
| Price change | `Open_PctChg`, `High_PctChg`, `Low_PctChg`, `Close_PctChg` | percent |
| Volume | `Volume_Log_Chg` | change in log(1 + volume) |
| Range and trend | `BB_Width`, `BB_PctB`, `MACD_Hist`, `RSI_14`, `ADX_14`, `DI_Plus_14`, `DI_Minus_14`, `Dist_SMA_50`, `Dist_SMA_200`, `Golden_Cross`, `Above_SMA_200`, `Trend_Bullish` | RSI and ADX in their usual 0–100 scale. `BB_PctB` is the position inside the band. Distances are ratios. The last three are 0 or 1. |
| Volatility flags | `VIX_High_Regime` (VIX above 25), `VIX_Extreme` (VIX above 35) | 0 or 1 |
| Regime | `HMM_Regime` (0 bear, 1 sideways, 2 bull), three probabilities, three one-hot flags, `HMM_Regime_Duration`, `HMM_Regime_Change` | probability, or bars since the state changed |
| Calendar | `Month`, `Quarter`, `WeekOfYear`, `SellInMay` | `SellInMay` is 1 from May through October |

The daily matrix adds `DayOfWeek`, `IsMonday`, and `IsFriday`. Lags on the daily matrix are 1, 2, 3, 5, and 10 sessions, on close change, volume change, RSI, MACD histogram, Bollinger %b, ADX, both DI lines, HMM regime, and the bear and bull probabilities. Rolling mean and standard deviation of the close change use 5, 10, and 20 sessions. VIX rolling means use the same windows. Ten products are stored, such as `RSI_14_x_BB_PctB` and `Close_PctChg_x_Volume_Log_Chg`.

The weekly matrix uses the same families with lags 1, 2, 3, and 4 weeks, and rolling windows of 4, 8, and 12 weeks. It has no day-of-week columns.

`factors/spy_mixed_features.csv` is the daily matrix plus the last completed week. Weekly columns are prefixed `W_`. A Friday bar still sees the previous week. The next session sees the week that just finished. Three extra columns compare the two clocks: `RSI_minus_Weekly`, `Ret_minus_Weekly`, and `BullProb_minus_Weekly`.

`factors.py` rebuilds these three matrices. `technical_indicators.py` is the indicator layer underneath it.

## Asset descriptors

One row per name in `universes/global.json`. These are summaries of that name's weekly history, not a time series.

| Column | Content |
|---|---|
| `asset`, `label`, `asset_class`, `market` | identity |
| `n_weeks` | 576 to 1,398 weeks |
| `ann_volatility`, `downside_vol`, `avg_true_range_pct` | percent |
| `mean_weekly_ret`, `ann_return` | percent |
| `raw_sharpe`, `calmar_ratio` | the asset's own path, not a strategy |
| `skewness`, `kurtosis` | of weekly returns |
| `autocorr_1`, `autocorr_4` | weekly return autocorrelation |
| `hurst` | around 0.5 is a random walk |
| `max_drawdown` | percent, negative |
| `up_week_ratio`, `pct_above_sma50` | fraction of weeks |
| `mean_abs_return`, `tail_ratio`, `volume_cv`, `mean_reversion_score` | scale-free descriptors |
| `regime_changes_per_year` | count |

Annualized volatility in this file runs from 1.37 (SHY) to 439 (Jarir). The Jarir, Al Rajhi (199), Maaden (105), and Tawuniya (97) figures are not believable for those businesses. They come from the weekly tape used when this table was built. Treat those four rows as bad inputs, not as a property of the company.
