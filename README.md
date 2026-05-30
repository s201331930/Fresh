# Quant Trading & Portfolio-Optimisation Toolkit

A research-grade Python toolkit for **long-only, leverage-free** systematic
investing. It contains two complementary systems that share one data layer,
metrics engine and reporting stack:

1. **Multi-Strategy System** — a book of several *uncorrelated* signal
   strategies (trend, mean-reversion, dual-momentum, breakout), each position
   risk-managed with stop-loss / take-profit / trailing stops, blended into one
   portfolio. See [`scripts/run_backtest.py`](scripts/run_backtest.py).
2. **Portfolio Optimiser** — *mathematical* mean-variance / risk-based
   optimisation of a **specific** basket of holdings, with monthly rebalancing,
   a trend/cash switch, volatility targeting, trailing stops and a circuit
   breaker, validated by a 15+ year **train / test / validate** walk-forward.
   See [`scripts/optimize_portfolio.py`](scripts/optimize_portfolio.py).

> **Mandate (both systems):** long only, leverage free (exposure ≤ equity at all
> times), every risk position carries a stop, and capital is diversified across
> low-correlation return drivers.

---

## System 2: Portfolio Optimiser (mathematical optimisation)

Built for an investor's actual buy-and-hold basket — by default
`QQQ, VEU, STC (7010.SR), Al Rajhi (1120.SR), Bonyan REIT (4347.SR),
Alinma Hospitality REIT (4349.SR)` — and answers: *what mix maximises
risk-adjusted return while controlling risk?*

**What it does**
- Races optimisation methods — `equal_weight`, `inverse_vol`, `min_variance`,
  `max_sharpe`, `risk_parity`, `max_diversification` — each under a **core**
  (fully-invested) and a **defensive** (trend filter + 10% vol target + 20%
  trailing stops + 20% circuit breaker) profile.
- Estimation-error defences: **Ledoit-Wolf covariance shrinkage** and
  EWM expected returns shrunk toward the cross-sectional mean, plus a 35%
  per-name cap. (Without these, max-Sharpe overfits and min-variance with cash
  degenerates to an all-cash solution.)
- **Walk-forward** with **train / test / validate** splits over **16+ years**.
  The champion is chosen on the *development* window (train+test) and then
  judged on the **untouched validate** holdout — the standard guard against
  curve-fitting.
- **Dynamic universe:** because the REITs listed recently (Bonyan 2018, Alinma
  Hospitality 2023), each holding joins the optimisation only once it has enough
  listed history. This preserves the long backtest while still using all current
  holdings. (SAR is USD-pegged at 3.75, so USD and Saudi names combine without
  FX adjustment.)
- **Data hygiene:** a Hampel median filter removes vendor bad ticks (several-fold
  one-day price spikes that appear in some Saudi adjusted closes) before any
  statistics are computed.

**Reference result (2010→2026, see `reports/portfolio_optimization_report.md`)**

| | Champion (risk-parity, core) | Your Buy & Hold | Defensive variant |
|---|---|---|---|
| Full-sample Sharpe | 0.70 | 0.69 | ~0.55 |
| Full-sample CAGR | 9.1% | 9.3% | ~7% |
| Max Drawdown | -23% | -23% | **-21%** |
| Validate max DD | -13% | -16% | **-13%** |

**Honest verdict (this is the important part).** For a small, already-diversified
6-name book, naive **1/N is very hard to beat** (the DeMiguel–Garlappi–Uppal
result). The optimiser's value is therefore *robust risk budgeting and
discipline*, not extra return: risk-parity matches buy & hold's risk-adjusted
return with lower single-name concentration. The **defensive** profile is genuine
**crash insurance** — it materially reduced the 2020 and 2022 drawdowns — at the
cost of lagging in the 2023–25 bull. The surest way to push the frontier out is
to add more *uncorrelated* assets, which this engine supports out of the box.

```bash
python scripts/optimize_portfolio.py --config config_portfolio.yaml
```

Edit `config_portfolio.yaml` to change holdings (name → Yahoo ticker), methods,
caps, the risk overlays, or the split fractions.

---

## System 1: Multi-Strategy Signal Book

Four deliberately uncorrelated long-only sleeves, each risk-managed and blended
with an inverse-volatility overlay.

| Sleeve | Return driver | Risk exits |
|---|---|---|
| **Trend Following** | Multi-month trends across diversified ETFs | ATR stop + wide ATR trailing stop; MA-cross exit |
| **Mean Reversion** | RSI(2) oversold bounces above the 200d trend | % stop, % take-profit, snap-back exit, 10-day time stop |
| **Dual Momentum** | Cross-sectional + absolute momentum rotation with a T-bill safe asset | Protective trailing stop between monthly rebalances |
| **Breakout** | 55-day Donchian breakouts on trendy/vol assets | ATR stop + ATR trailing stop; 20-day Donchian exit |

Reference run (2008→present): blended Sharpe ≈ 0.57 with a -6% max drawdown vs
SPY's -52%, beta ≈ 0.10. See `reports/backtest_report.md`.

```bash
python scripts/run_backtest.py --config config.yaml
```

---

## KPIs reported (both systems)

Return & risk: Total Return, CAGR, Annualised Volatility, Sharpe, Sortino,
Calmar. Drawdown: Max/Average Drawdown, Max Drawdown Duration. Tail risk: daily
VaR/CVaR (95%), skew, kurtosis, best/worst day. Benchmark-relative: Beta,
annualised Alpha, correlation. Trade-level: number of trades, win rate, profit
factor, payoff ratio, expectancy, average win/loss, holding period. Plus
correlation matrices, efficient-frontier and allocation charts, and monthly /
per-era return tables.

## Project layout

```
config.yaml                    # multi-strategy config
config_portfolio.yaml          # portfolio-optimiser config (holdings, methods, risk)
requirements.txt
scripts/
  run_backtest.py              # System 1 runner
  optimize_portfolio.py        # System 2 runner (optimisation + walk-forward)
src/quant/
  data/loader.py               # yfinance download + cache + de-spike + synthetic fallback
  engine/backtester.py         # long-only, leverage-free, SL/TP/trailing engine
  engine/portfolio.py          # multi-strategy combiner + vol-parity overlay
  optimization/optimizer.py    # mean-variance / risk-parity / etc. + shrinkage
  optimization/allocation.py   # rebalancing + trend/vol/stop overlays, dynamic universe
  optimization/walkforward.py  # train/test/validate splitting & champion selection
  metrics/kpis.py              # all backtesting KPIs
  strategies/                  # trend / mean-reversion / dual-momentum / breakout
  reporting/report.py          # markdown tear-sheet + charts
tests/                         # engine, metrics & optimiser unit tests (pytest)
reports/                       # generated reports, figures, CSVs (committed samples)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/optimize_portfolio.py --config config_portfolio.yaml   # System 2
python scripts/run_backtest.py        --config config.yaml            # System 1
python -m pytest tests/ -q                                            # 14 tests
```

Downloads are cached under `data/cache/`; if a symbol can't be fetched the loader
falls back to stale cache and finally to a reproducible synthetic series (flagged
in the report) so the pipeline always runs.

## Backtest hygiene (no look-ahead, honest fills)

- Signals use only trailing data; optimiser inputs are estimated from windows
  that end at the rebalance date and are filled at that close. Breakout highs in
  System 1 are shifted one bar.
- Per-position stop / target / trailing exits are intrabar; if a bar could hit
  both stop and target the **stop is assumed first** (worst case); gaps fill at
  the open.
- Per-side commission and adverse slippage are charged on every fill; idle cash
  earns the configured risk-free rate; prices are split/dividend adjusted.

## Going live — caveats

Backtests are not guarantees. Validate against out-of-sample / walk-forward
windows (built in here), stress costs and slippage (Saudi REIT liquidity in
particular), confirm your live data vendor, and size positions to your own risk
budget. The toolkit is designed to be extended — add holdings, strategies or
assets in the config and re-run.
