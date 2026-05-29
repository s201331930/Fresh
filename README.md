# Multi-Strategy Long-Only Trading System

A research-grade backtesting framework for a **long-only, leverage-free** book
built from several **deliberately uncorrelated** strategy sleeves. Every open
position is risk-managed with a **stop-loss**, and — depending on the sleeve — a
**take-profit** and/or a **trailing stop**. The whole pipeline produces a
desk-ready tear-sheet with the full set of backtesting KPIs.

> **Design mandate**
> - **Long only** — we never short.
> - **Leverage free** — every purchase is gated by available cash, so aggregate
>   exposure can never exceed equity.
> - **Risk managed** — no position is ever held without a stop.
> - **Uncorrelated** — sleeves are built around different return drivers and
>   instruments so their P&L streams diversify each other.

---

## Why this design

A single long-only strategy is hostage to one return driver and one regime.
Blending several *uncorrelated* sleeves is the cleanest, leverage-free way to
raise risk-adjusted return: when one sleeve is flat or drawing down, another is
usually working. The objective here is **capital preservation with steady
compounding and shallow drawdowns**, not chasing the index.

In the reference backtest (2008→present, US-listed ETF proxies) the blended
book delivered a far smaller drawdown than buy-and-hold equities at a comparable
Sharpe and a much higher Calmar, with a market beta near zero — i.e. it behaves
like a genuine diversifier rather than a closet index fund. See
[`reports/backtest_report.md`](reports/backtest_report.md) for the full numbers.

## The four sleeves

| Sleeve | Return driver | Instruments | Entry logic | Risk exits |
|---|---|---|---|---|
| **Trend Following** | Multi-month price trends | Diversified ETFs (equities, bonds, gold, commodities, REITs) | Price > 200d MA, 50d > 200d MA, new 100d high | ATR hard stop + wide ATR trailing stop; MA-cross exit |
| **Mean Reversion** | Short-term oversold bounces *in an uptrend* | US equity index/sector ETFs | Close > 200d MA **and** RSI(2) < 10 | % stop, % take-profit, RSI snap-back exit, 10-day time stop |
| **Dual Momentum** | Cross-sectional + absolute momentum rotation | Multi-asset ETFs with a T-bill safe asset | Monthly: hold top-N by 6-month momentum, else go to cash/bills | Protective trailing stop between rebalances |
| **Breakout** | Volatility expansion / range breakouts | Higher-vol / trendy assets (Nasdaq, metals, energy, EM, crypto proxy) | Close ≥ 55-day Donchian high | ATR stop + ATR trailing stop; 20-day Donchian exit |

These are combined with configurable base weights and an optional
**inverse-volatility (vol-parity)** overlay that tilts capital toward whichever
sleeves are currently calmest. Because the portfolio return is a convex
combination of sleeve returns (weights ≥ 0, summing to 1), the **book stays
leverage-free** at every level.

## KPIs reported

Return & risk: Total Return, CAGR, Annualised Volatility, Sharpe, Sortino,
Calmar. Drawdown: Max Drawdown, Average Drawdown, Max Drawdown Duration. Tail
risk: daily VaR/CVaR (95%), skew, kurtosis, best/worst day. Benchmark-relative:
Beta, annualised Alpha, correlation to benchmark. Trade-level: number of trades,
win rate, profit factor, payoff ratio, expectancy, average win/loss, best/worst
trade, average holding period, and an exit-reason breakdown (stop / target /
trailing / signal / time). Plus a **cross-strategy correlation matrix** — the
key diversification check — and a monthly/annual returns table.

## Project layout

```
config.yaml                  # all knobs: universe, params, costs, weights
requirements.txt
scripts/run_backtest.py      # end-to-end runner -> reports/
src/quant/
  data/loader.py             # yfinance download + CSV cache + synthetic fallback
  engine/backtester.py       # long-only, leverage-free, SL/TP/trailing engine
  engine/portfolio.py        # sleeve combiner + vol-parity overlay
  metrics/kpis.py            # all backtesting KPIs
  strategies/                # trend / mean-reversion / dual-momentum / breakout
  reporting/report.py        # markdown tear-sheet + PNG charts
tests/                       # engine & metrics unit tests
reports/                     # generated tear-sheet, figures, CSVs (committed sample)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_backtest.py --config config.yaml
```

Outputs land in `reports/`:
- `backtest_report.md` — the full tear-sheet
- `figures/` — equity curves, drawdown, correlation heatmap, weight evolution
- `trades.csv`, `portfolio_equity.csv`, `strategy_weights.csv`, `correlation_matrix.csv`

The data loader caches downloads under `data/cache/`. If a symbol cannot be
downloaded (offline / rate-limited) it falls back to a stale cache and, as a last
resort, a **reproducible synthetic series** so the pipeline always runs — any
synthetic symbols are flagged at the top of the report.

## Backtest hygiene (no look-ahead, honest fills)

- Signals are computed on the **close of day _t_** and filled at the **open of
  day _t+1_**. Rolling highs used for breakouts are shifted one bar.
- Stop / target / trailing exits are **intrabar**. If a bar could hit both the
  stop and the target, the **stop is assumed to fill first** (worst case); gaps
  through the stop fill at the open.
- **Per-side commission and adverse slippage** are charged on every fill; idle
  cash earns the configured risk-free rate.
- Prices are split/dividend **adjusted** so total-return is honest.

## Tests

```bash
python -m pytest tests/ -q
```

Covers take-profit, stop-loss and trailing-stop exits, the **leverage-free
invariant** (exposure ≤ 1, cash ≥ 0 at all times), no-look-ahead fills, and the
KPI calculations.

## Going live — caveats

This is a backtest, not a guarantee. Before risking capital: validate against an
out-of-sample / walk-forward window, stress costs and slippage, confirm the data
vendor for live use, and size positions to your own risk budget. ETF proxies are
used per asset class; swap in your tradable instruments in `config.yaml`.
