# SPY Trading Signal Prototype

Predictive model prototype for identifying good entry points in SPY using weekly timeframe data.

## Data

- **Ticker:** SPY (S&P 500 ETF)
- **Timeframe:** Weekly bars
- **Start date:** 2000-01-01
- **Fields:** Open, High, Low, Close, Volume

## Setup

```bash
pip install -r requirements.txt
python fetch_spy_data.py
```

## Project Structure

```
├── fetch_spy_data.py      # Downloads SPY weekly OHLCV data
├── data/
│   └── spy_weekly_ohlcv.csv   # Raw weekly price/volume data
└── requirements.txt
```
