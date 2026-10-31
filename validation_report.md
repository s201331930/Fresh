# Adversarial validation

Skeptical pass over the walk-forward ML trading pipeline. Costs stay at the published setting (2 bps, `COST_BPS = 2.0` in `daily_vs_weekly.py` and `COST_PER_TRADE = 0.0002` in `multi_asset_backtest.py`). The original pipeline files are not edited. Every alternative is a flag in `validation/` with the published behavior as the default (`--macro-lag 0`, `--fill close`).

Seed for the ensembles is the hard-coded `random_state=42` in `daily_vs_weekly.py`. Each rerun records that seed, a hash of the adjusted close and the macro frame, and `git rev-parse HEAD`.

## 1. Repo map

| Stage | Study | Path |
| --- | --- | --- |
| Universe | C, current Tadawul All Share, 252 names, source date 24 Oct 2026 | `tasi_universe.json` |
| Universe | A/B, 102 assets | `study/expanded_universe.json`, `daily_universe.py` |
| Daily prices | B/C. Yahoo `auto_adjust=False`, OHLC scaled by Adj Close/Close, volume 0 dropped | `daily_vs_weekly.py` `fetch_ohlcv` lines 103–129 |
| Weekly prices | A. Same adjustment, then `W-FRI` or `W-THU` | `multi_asset_backtest.py` `fetch_weekly_data` lines 91–120 |
| Macro download | B/C. VIX, TNX, DXY, Gold, Brent. Timezone stripped, date-normalized, no lag | `daily_vs_weekly.py` `fetch_macro_daily` lines 136–167 |
| Macro join | B/C. Exact calendar date, then forward-fill, then percent change | `daily_vs_weekly.py` `add_macro` lines 303–311 |
| Macro join | A. Resample last on `W-THU` (TASI) or `W-FRI`, exact join, forward-fill, then `fillna(0)` | `multi_asset_backtest.py` `fetch_macro_for_asset` lines 122–158 |
| Indicators | All. Rolling windows use `min_periods=period`. No centered window | `technical_indicators.py` |
| Feature frame | B/C | `daily_vs_weekly.py` `build_base_features`, `add_calendar` (line 350), `add_lags`, `feature_columns` (389), `impute` (408–411, forward-fill then 0, no back-fill) |
| Weekly release into daily | B mixed mode. Completed week is visible the next calendar day | `daily_vs_weekly.py` `release_weekly_to_daily` lines 414–419, checked at 529–560 |
| HMM | B/C. Fit on bars strictly before the segment. Forward filter only. States reordered by mean return each refit | `daily_vs_weekly.py` `hmm_walk_forward` lines 217–296 |
| HMM | A. VIX is `ffill().bfill()` at line 71. Last filtered state of `predict` on `obs[:t+1]` is kept | `hmm_regime.py` `fit_hmm_walk_forward` |
| Walk-forward | B/C. Feature selection and early stopping on the training tail only | `daily_vs_weekly.py` `select_features` 581, `_fit_ensemble` 598, `walk_forward` 642–682 |
| Walk-forward | A. Early stopping `eval_set` is the test fold | `multi_asset_backtest.py` lines 312, 319, 326 |
| Threshold / scale | B/C. Fit on the calibration block only, frozen before evaluation | `daily_vs_weekly.py` `fit_threshold` 720–733, `run_mode` 814–816 |
| Threshold / scale | C periods. Refit on the three years before each period; frozen pair from before the first period | `tasi_periods.py` `_calib_before` 174, `run_one` 244–266 |
| Threshold / scale | A. Threshold maximizes Sharpe on the first half of the OOS predictions. The published backtest scores the full OOS series, including that half. Scaled divisor is the 90th percentile of the full prediction series | `multi_asset_backtest.py` lines 509–523 |
| Positions and PnL | B/C. Long-only, 2 bps on turnover, initial position charged | `daily_vs_weekly.py` `positions_*` 689–717, `score_window` 736–782 |
| Positions and PnL | A | `multi_asset_backtest.py` `backtest` line 378 |
| Impute | A. Forward-fill and back-fill | `multi_asset_backtest.py` `impute_nans` line 257 |
| Metrics in the studies | Sharpe `mean/std * sqrt(bars_per_year)` with a zero risk-free rate. CAGR on calendar days/365.25. Drawdown on the bar equity | `score_window`. Study C passes `bars_per_year=252` (`tasi_periods.py` `_score` line 194) |
| Study A runner and report | | `expanded_study.py`, `generate_expanded_report.py`, `study/expanded_characteristics.csv` |
| Study B runner and report | | `daily_universe.py`, `generate_universe_report.py`, `daily_universe/` |
| Study C runner and report | | `tasi_periods.py`, `generate_tasi_report.py`, `tasi_periods/` |
| SPY daily study | | `daily_vs_weekly.py` SPY wrappers, `generate_daily_report.py`, `daily_study/` |
| This audit | Flags, clock, reruns | `validation/` |

## 2. Summary

| Task | Concern | Verdict | Key number | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Same-day US/ICE macro on a 15:00 Riyadh decision | Pending the paired rerun. The join itself is a same-day read on overlap sessions. | VIX: Sunday 0/792 same-day; Thursday 689/710 same-day. 2026-10-01 lag 0 would use the 1 Oct print; lag 1 uses 30 Sep. | `validation/macro_clock.csv`, `validation/macro_same_day_counts.csv` |

The paired Sharpe test (lag 0 vs lag 1 vs no macro, 155 names, every period, Binary t=0, cost unchanged) is the pass rule for this row. It is not filled in until that rerun finishes.

## 3. Task 1 — cross-timezone macro look-ahead

Tadawul is open Sunday–Thursday and closes at 15:00 Asia/Riyadh (12:00 UTC). The five macro series are not known at that clock:

| Series | Yahoo | Source zone | Event | Local time | Riyadh, US on EDT | Riyadh, US on EST | Known by 15:00 Riyadh? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VIX | `^VIX` | America/New_York | CBOE cash close | 16:15 | 23:15 same date | 00:15 next date | No |
| TNX | `^TNX` | America/New_York | Treasury cash close | 16:00 | 23:00 same date | 00:00 next date | No |
| DXY | `DX-Y.NYB` | America/New_York | ICE futures settle | 17:00 | 00:00 next date | 01:00 next date | No |
| Gold | `GC=F` | America/New_York | COMEX settlement | 13:30 | 20:30 same date | 21:30 same date | No |
| Brent | `BZ=F` | Europe/London | ICE Brent settlement | 19:30 | 21:30 same date | 22:30 same date | No |

Yahoo daily bars are date-stamped, not timed. The table is the exchange convention. `fetch_macro_daily` drops the timezone and normalizes to midnight. `add_macro` then joins on that date and forward-fills.

What that does on the Al Rajhi calendar (3,958 sessions, 2010-03-03 to 2026-10-29), counting each series on its own non-null prints:

- Sunday: lag 0 and lag 1 both take Friday. 0 of 792 Sundays see a same-day print. Sunday is not the leak.
- Monday–Thursday: lag 0 takes that session's print whenever the series traded. For VIX that is 743/819 Mondays, 814/820 Tuesdays, 808/817 Wednesdays, 689/710 Thursdays.
- Brent is missing on 1,991 macro rows. Its own prints still collide with the Saudi session on 732 Mondays, 805 Tuesdays, 802 Wednesdays, and 688 Thursdays. A shared calendar would have hidden those holes. Lag 1 is computed per series.

Checked session: Thursday 2026-10-01. Lag 1 attaches the 2026-09-30 print for VIX (16.34), TNX (5.293), DXY (101.45), Gold (4186.7), and Brent (103.53). Lag 0's join uses 2026-10-01 when that print exists. Sunday 2026-09-27 takes Friday 2026-09-25 under both rules.

A one-row shift on the US calendar is the wrong fix. It writes Thursday's close onto Friday's label, and Sunday then forward-fills Thursday and misses Friday. `validation/check_clock.py` locks the correct rule: Sunday sees Friday, Thursday does not see Thursday. The implementation is `merge_asof(..., direction="backward", allow_exact_matches=False)` on each series' non-null dates (`validation/macro_clock.py`). `--macro-lag 0` passes the vendor frame through, so `add_macro` is unchanged. `--macro-lag none` drops the columns.

Al Rajhi lag 0, this download, full window from 2016-01-03: Binary (t=0) Sharpe 1.310, buy-and-hold 0.877. The published Study C file for 1120 is 1.273 and 0.879. Same cost, same code path. The gap is the fresh Yahoo pull (hash `affdaabf32a08d37`), not a scoring change. Pair every later comparison to this audit's lag 0, not to the October headline file alone.

Pass rule, applied to period rows (same name, same period): median (lag 1 Sharpe − lag 0 Sharpe) inside ±0.03, and the share of periods with Sharpe above buy-and-hold falls by fewer than 5 percentage points. Otherwise the leak is confirmed and lag 1 replaces lag 0 as the baseline. Macro removed entirely is the third column.

## 4. Run log

| Config | Seed | Commit at launch | Data | Status |
| --- | --- | --- | --- | --- |
| TASI daily Binary (t=0), macro lag 0, fill close, 2 bps | 42 | this commit, written into each JSON | adjusted OHLC in `daily_study/cache/` (not committed), macro `macro_daily.csv` | full 155-name batch launched after this commit. Smoke on 1120 before the commit: Sharpe 1.310 vs buy-and-hold 0.877, hash `affdaabf32a08d37` |
| macro lag 1 | 42 | same | same prices, each series' last print strictly before the session | same batch |
| macro removed | 42 | same | same prices, no macro columns | same batch |

Original published Study C Binary (t=0), for the side-by-side once the rerun is in: 290/438 period wins (66%), median Sharpe 0.404 vs buy-and-hold 0.244, median gap +0.187, median CAGR 6.40% vs 1.81%. Full window: 114/155, median gap +0.177, median Sharpe 0.361 vs 0.165, median CAGR 5.96% vs −0.06%. Source: `tasi_periods/summary.json`.
