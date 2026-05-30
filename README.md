# Quant Trading & Portfolio-Optimisation Toolkit

A research-grade Python toolkit for **long-only, leverage-free** systematic
investing. It contains three complementary pieces sharing one data layer,
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
3. **Signal-Alpha Research** — measures whether technical signals (RSI,
   Bollinger, MA, momentum, VWAP, volume) and a VIX macro overlay can *smartly*
   over/under-weight the book to beat buy & hold, using information coefficients
   and an out-of-sample momentum-vs-contrarian race.
   See [`scripts/signal_research.py`](scripts/signal_research.py).

> **Mandate (everywhere):** long only, leverage free (exposure ≤ equity at all
> times), every risk position carries a stop, and capital is diversified across
> low-correlation return drivers.

---

## Signal-Alpha Research — "can I smartly allocate to beat buy & hold?"

For the basket `QQQ, VEU, STC (7010.SR), Al Rajhi (1120.SR), Bonyan REIT
(4347.SR), Alinma Hospitality REIT (4349.SR)`, the engine first asks whether the
signals actually predict returns (information coefficient), then races the two
opposing theses out-of-sample.

**Information-coefficient finding (2010→2026):** the only signals with positive
predictive pull are 12-1 momentum (IC +0.07) and **1-month reversal (IC +0.06)**
— both weak; trend/MA, RSI, VWAP and volume are non-predictive (IC ≈ 0). The
exploitable short-horizon effect is **mean-reversion**: recent laggards bounce.

**Out-of-sample race (validate era is untouched):**

| Strategy | Validate Sharpe | Validate MaxDD | Full Sharpe | Full CAGR |
|---|---|---|---|---|
| Buy & Hold (1/N) | 0.62 | -15.9% | 0.69 | 9.3% |
| Momentum tilt ("double down on winners") | **0.12** | -19.0% | 0.61 | 8.4% |
| **Contrarian tilt (mean-revert)** | **0.67** | **-10.2%** | 0.67 | **9.1%** |
| **Contrarian + VIX throttle** | **0.76** | **-8.5%** | 0.70 | 8.2% |

**Verdict:** *doubling down on winners over-fits and fails live* (validate
Sharpe ≈ 0.12). The real, repeatable edge is **contrarian / mean-reversion**: a
mild "buy the laggards" tilt matched buy & hold's return with a higher
out-of-sample Sharpe and ~35% shallower drawdowns; adding a VIX risk-off
throttle pushed validate Sharpe to 0.76 with the shallowest drawdowns. This is
*why* 1/N is respectable — monthly rebalancing already harvests mean-reversion;
the contrarian tilt simply leans into it with risk budgeting. Full evidence
(IC table, charts, per-era tables) in `reports/signal_alpha_report.md`.

```bash
python scripts/signal_research.py --config config_portfolio.yaml
```

---

## Portfolio Optimiser (mathematical optimisation)

Answers: *what mix maximises risk-adjusted return while controlling risk?*

- Races `equal_weight`, `inverse_vol`, `min_variance`, `max_sharpe`,
  `risk_parity`, `max_diversification` — each under a **core** (fully-invested)
  and a **defensive** (trend filter + 10% vol target + 20% trailing stops + 20%
  circuit breaker) profile.
- Ledoit-Wolf covariance shrinkage; EWM expected returns shrunk to the
  cross-sectional mean; 35% per-name cap.
- **Walk-forward** with **train / test / validate** over 16+ years; champion
  chosen on the *development* window, judged on the **untouched validate** era.
- **Dynamic universe** (REITs join once they have enough listed history) and a
  **Hampel de-spike** filter that removes vendor bad ticks in Saudi adjusted
  closes. SAR is USD-pegged (3.75) so no FX adjustment.

Reference: risk-parity (core) matched buy & hold's risk-adjusted return with
lower concentration; the defensive variant cut crisis drawdowns. Naive 1/N is
genuinely hard to beat on a small book — the optimiser's edge is robust risk
control. Full tear-sheet: `reports/portfolio_optimization_report.md`.

```bash
python scripts/optimize_portfolio.py --config config_portfolio.yaml
```

---

## Multi-Strategy Signal Book

Four uncorrelated long-only sleeves, each risk-managed and blended with an
inverse-volatility overlay.

| Sleeve | Return driver | Risk exits |
|---|---|---|
| **Trend Following** | Multi-month trends across diversified ETFs | ATR stop + wide ATR trailing stop; MA-cross exit |
| **Mean Reversion** | RSI(2) oversold bounces above the 200d trend | % stop, % take-profit, snap-back exit, time stop |
| **Dual Momentum** | Cross-sectional + absolute momentum rotation with a T-bill safe asset | Protective trailing stop between rebalances |
| **Breakout** | 55-day Donchian breakouts on trendy/vol assets | ATR stop + ATR trailing stop; 20-day Donchian exit |

Reference run (2008→present): blended Sharpe ≈ 0.57, max drawdown -6% vs SPY
-52%, beta ≈ 0.10. See `reports/backtest_report.md`.

```bash
python scripts/run_backtest.py --config config.yaml
```

---

## KPIs reported

Total Return, CAGR, Annualised Volatility, Sharpe, Sortino, Calmar, Max/Average
Drawdown + duration, daily VaR/CVaR (95%), skew, kurtosis, beta/alpha/benchmark
correlation, and trade-level stats (win rate, profit factor, payoff, expectancy,
holding period). Plus correlation matrices, efficient-frontier, allocation and
drawdown charts, an **information-coefficient** table, and monthly / per-era
return tables.

## Project layout

```
config.yaml                    # multi-strategy config
config_portfolio.yaml          # portfolio-optimiser & signal-research config
requirements.txt
scripts/
  run_backtest.py              # multi-strategy runner
  optimize_portfolio.py        # optimisation + walk-forward runner
  signal_research.py           # IC analysis + momentum-vs-contrarian race
src/quant/
  data/loader.py               # yfinance + cache + Hampel de-spike + synthetic fallback
  engine/backtester.py         # long-only, leverage-free, SL/TP/trailing engine
  engine/portfolio.py          # multi-strategy combiner + vol-parity overlay
  optimization/optimizer.py    # mean-variance / risk-parity / etc. + shrinkage
  optimization/allocation.py   # rebalancing + trend/vol/stop overlays, dynamic universe
  optimization/signal_alpha.py # signal components, IC, VIX regime, tilt allocators
  optimization/walkforward.py  # train/test/validate splitting & champion selection
  metrics/kpis.py              # all backtesting KPIs
  strategies/                  # trend / mean-reversion / dual-momentum / breakout
  reporting/report.py          # markdown tear-sheet + charts
tests/                         # engine, metrics, optimiser & signal tests (pytest)
reports/                       # generated reports, figures, CSVs (committed samples)
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/signal_research.py     --config config_portfolio.yaml   # IC + tilt race
python scripts/optimize_portfolio.py  --config config_portfolio.yaml   # optimisation
python scripts/run_backtest.py        --config config.yaml             # multi-strategy
python -m pytest tests/ -q                                             # 18 tests
```

## Backtest hygiene (no look-ahead, honest fills)

- Signals/optimiser inputs use only trailing data; ICs use forward returns only
  for evaluation, never for trading. Breakout highs are shifted one bar.
- Per-position stop / target / trailing exits are intrabar; if a bar could hit
  both, the stop is assumed first; gaps fill at the open.
- Per-side commission and adverse slippage on every fill; idle cash earns the
  risk-free rate; split/dividend-adjusted prices; vendor bad ticks de-spiked.

## Going live — caveats

Backtests are not guarantees. Edges on a small basket are weak (|IC| ≲ 0.07), so
tilts are kept mild and diversified; aggressive concentration increased
drawdowns in testing. Validate against the built-in walk-forward, stress
costs/slippage (Saudi REIT liquidity especially), monitor the IC table for
regime change, and size to your own risk budget. Adding more *uncorrelated*
assets is the surest way to push the frontier out.
