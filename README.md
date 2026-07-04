# Trading System Prototype

Predictive-model prototype for identifying favorable entry points on **SPY** using weekly time-frame data.

## Data

### SPY Weekly OHLCV (`data/spy_weekly_ohlcv.csv`)

| Field | Description |
|-------|-------------|
| `Week_Ending` | Friday date marking the end of each trading week |
| `Open` | Adjusted opening price (first trading day of the week) |
| `Low` | Adjusted lowest price during the week |
| `High` | Adjusted highest price during the week |
| `Close` | Adjusted closing price (last trading day of the week) |
| `A2P` | Adjustment-to-price ratio (`Adj Close / Close`) |
| `Volume` | Total shares traded during the week |

### Enriched Dataset (`data/spy_weekly_enriched.csv`)

SPY weekly data joined with macro variables on matching `Week_Ending`:

| Field | Description | Coverage |
|-------|-------------|----------|
| `VIX_Open/High/Low/Close` | CBOE Volatility Index weekly OHLC | 2000–present (100%) |
| `Brent_Open/High/Low/Close` | Brent Crude Oil futures weekly OHLC | 2007–present (71%) |

### Order Blocks (`data/spy_weekly_with_ob.csv`)

SPY data with bullish/bearish Order Block signals (periods=5, threshold=0%).

## Quick Start

```bash
pip install -r requirements.txt
python fetch_spy_data.py          # SPY weekly OHLCV
python fetch_macro_data.py        # join VIX + Brent to SPY
python order_block_finder.py      # detect Order Blocks
python analyze_distributions.py   # frequency & outlier analysis
python plot_ob_chart.py           # candlestick charts with OBs
```

## Project Structure

```
.
├── data/
│   ├── spy_weekly_ohlcv.csv
│   ├── spy_weekly_enriched.csv
│   ├── spy_weekly_with_ob.csv
│   ├── distribution_report.csv
│   ├── outliers_detail.csv
│   ├── spy_weekly_ob_chart.png
│   └── spy_weekly_ob_chart_zoom.png
├── fetch_spy_data.py
├── fetch_macro_data.py
├── order_block_finder.py
├── analyze_distributions.py
├── plot_ob_chart.py
├── requirements.txt
└── README.md
```
