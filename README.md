# SPY Trading Signal Prototype

Prototype predictive-model pipeline for identifying entry points on SPY using weekly OHLCV data.

## Quick Start

```bash
pip install -r requirements.txt
python src/fetch_spy_data.py
```

This downloads SPY weekly bars (Open, High, Low, Close, Volume) from **2000-01-01** to the present via Yahoo Finance, saves them under `data/`, and prints a data-quality report.

## Project Layout

```
src/
  fetch_spy_data.py   – data ingestion & quality checks
data/                  – generated data files (git-ignored)
requirements.txt       – Python dependencies
```

## Data Notes

- Source: Yahoo Finance via `yfinance`
- Granularity: weekly bars anchored to Friday close (standard US equity convention)
- Prices are **split- and dividend-adjusted** (`auto_adjust=True`)
- Resampled from daily to weekly in-house to avoid yfinance weekly-alignment quirks
