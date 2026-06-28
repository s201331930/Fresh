# Trading System Prototype

Predictive-model prototype for identifying favorable entry points on **SPY** using weekly time-frame data.

## Data

| Field | Description |
|-------|-------------|
| `Week_Ending` | Friday date marking the end of each trading week |
| `Adj Close` | Adjusted closing price (split & dividend adjusted) |
| `Volume` | Total shares traded during the week |
| `A2P` | Adjustment-to-price ratio (`Adj Close / Close`) |
| `Adj Open` | Adjusted open (`Open * A2P`) |
| `Adj High` | Adjusted high (`High * A2P`) |
| `Adj Low` | Adjusted low (`Low * A2P`) |

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
