# Trading System Prototype

Predictive-model prototype for identifying favorable entry points on **SPY** using weekly time-frame data.

## Data

| Field | Description |
|-------|-------------|
| `Week_Ending` | Friday date marking the end of each trading week |
| `Open` | Opening price (first trading day of the week) |
| `High` | Highest price during the week |
| `Low` | Lowest price during the week |
| `Close` | Raw closing price (last trading day of the week) |
| `Adj Close` | Adjusted closing price (split & dividend adjusted) |
| `Volume` | Total shares traded during the week |

`Close` is the raw unadjusted price; `Adj Close` is adjusted for splits and dividends.

## Quick Start

```bash
pip install -r requirements.txt
python fetch_spy_data.py        # downloads & saves data/spy_weekly_ohlcv.csv
```

## Project Structure

```
.
├── data/
│   └── spy_weekly_ohlcv.csv    # SPY weekly OHLCV since 2000-01-01
├── fetch_spy_data.py           # data download & resample script
├── requirements.txt
└── README.md
```
