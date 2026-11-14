# Data and factors

Strategy code, trained models, and backtest results have been removed.
What remains is the market data, the asset lists, and the factor tables
for a later model. The column-level catalog is `DATA_CATALOG.md`.

## Universes

- `universes/global.json` — 102 names, with market and asset class
- `universes/tasi.json` — 252 Tadawul names as listed on 24 Oct 2026

## Prices

- `data/spy_weekly_ohlcv.csv` — SPY weekly bars from 2000
- `data/spy_weekly_enriched.csv` — the same weeks with VIX, Brent, indicators, and order-block zones
- `data/spy_weekly_with_ob.csv` — order-block columns on the weekly bars

## Factor tables

- `factors/spy_daily_features.csv` — 6,238 rows, 117 factor columns
- `factors/spy_weekly_features.csv` — 1,294 rows, 103 factor columns
- `factors/spy_mixed_features.csv` — daily factors plus the last completed week, 6,238 rows, 155 columns
- `factors/asset_descriptors.csv` — volatility, autocorrelation, Hurst, drawdown, and related descriptors for the 102 names

The old next-bar target column has been removed from the factor tables.

## Rebuild

`technical_indicators.py` computes the moving averages, bands, MACD, OBV, RSI, and DI.
`order_block_finder.py` marks demand and supply zones.
`fetch_spy_data.py` and `fetch_macro_data.py` refresh the weekly SPY files.
`factors.py` rebuilds the daily, weekly, and mixed factor matrices, including lags and a causal HMM regime. A finished weekly bar is released on the next calendar day.
