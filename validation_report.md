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
| 1 | Same-day US/ICE macro on a 15:00 Riyadh decision | **CONFIRMED** | Period median Sharpe 0.435 → 0.327 (paired median −0.086). Beat-buy-and-hold 300/438 (68.5%) → 270/438 (61.6%), a drop of 6.8 points. Lag 1 is the new baseline. | `validation/paired_lag1_minus_lag0.csv`, `validation/macro_same_day_counts.csv` |
| 2 | Corporate-action jumps in Yahoo `.SR` | **CONFIRMED** | On the lag-1 tape, neutralizing \|return\| > 15% moves the period median Sharpe gap +0.108 → +0.138 (+28% relative). Beat rate 270/438 → 274/438 (+1.5% relative). Paired median gap change is 0.00 because 221/438 periods are untouched. Maaden and Tawuniya lose the 2022–2026 edge. | `validation/pipeline_jumps.csv`, `validation/paired_clean_minus_lag1.csv` |
| 3 | Close-to-close fill is not a trade you can place at 15:00 after seeing that close | **CONFIRMED** | Same cleaned lag-1 tape. Next-open median period gap +0.138 → +0.017 (12% of the close-fill gap; the bar was 60%). One-bar delay → −0.033. Median close-to-close mean is 99% overnight. | `validation/paired_next_open_minus_close.csv`, `validation/paired_delay1_minus_close.csv`, `validation/overnight_intraday.csv` |
| 4 | Study A's weekly means are not Study B's frozen weekly results | **CONFIRMED** | Mean best Sharpe 0.921 vs Study B mean best weekly 0.521. Mean vs median inside Study A is only 0.921 vs 0.901. t=opt beats buy-and-hold in 100/102 (Study A) vs 35/102 (Study B). Tiers withdrawn. The point split between window and leakage was not isolated by a refit. | `study/expanded_characteristics.csv`, `daily_universe/*.json`, `multi_asset_backtest.py` lines 312–326 and 509–523 |
| 5 | The strategy-picking tree and forest have no skill beyond "always t=opt" | **CONFIRMED** | Majority exact 51/102, top-2 80/102. Tree 48/102 and 74/102, permutation p = 0.225. Forest 50/102 and 77/102. Nested depth selection stays at 48/102. | `validation/meta_model_tree.json` |

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

Pass rule, applied to period rows (same name, same period): median (lag 1 Sharpe − lag 0 Sharpe) inside ±0.03, and the share of periods with Sharpe above buy-and-hold falls by fewer than 5 percentage points. Otherwise the leak is confirmed and lag 1 replaces lag 0 as the baseline.

### Result — CONFIRMED

All 155 names, 438 periods, 0 errors. Same prices (buy-and-hold Sharpe and CAGR match to the stored decimal; max absolute difference 0). Same cost (2 bps), same seed (42), same commit `e3af634d8d9629f149327e74eff313bd7bb2b9a6`. Finished 31 Oct 2026, 13:22 / 13:34 / 13:49 Asia/Riyadh for lag 0, lag 1, and macro removed.

| | Lag 0 (published join) | Lag 1 (new baseline) | Macro removed |
| --- | --- | --- | --- |
| Periods beating own buy-and-hold | 300/438 (68.5%) | 270/438 (61.6%) | 274/438 (62.6%) |
| Median period Sharpe | 0.435 | 0.327 | 0.354 |
| Median period gap vs buy-and-hold | +0.216 | +0.109 | +0.106 |
| Median period CAGR | 7.28% | 4.66% | 5.34% |
| Median period CAGR gap | +6.01 | +3.21 | +3.72 |
| Full-window names beating buy-and-hold | 114/155 (73.5%) | 97/155 (62.6%) | 107/155 (69.0%) |
| Full-window median Sharpe | 0.393 | 0.281 | 0.315 |
| Full-window median gap | +0.185 | +0.078 | +0.117 |
| Full-window median CAGR | 6.74% | 3.56% | — |
| Full-window median CAGR gap | +5.75 | +3.11 | +4.38 |

Paired median (lag 1 − lag 0) Sharpe is **−0.086**. Mean paired change −0.092. The beat rate falls **6.8 percentage points**. Both pass bars are missed. Buy-and-hold did not move, so the whole change is the strategy.

The loss is not uniform. Median paired change by period: 2016–2018 −0.206 (95/126 wins → 74/126), 2019–2021 +0.027 (78 → 80), 2022–2026 −0.124 (114/142 → 102/142). The bull window is untouched. The weak-market windows, where a same-day VIX or oil print is most valuable, give the leak back.

Dropping macro entirely, paired against lag 0, is a median Sharpe change of −0.054 and a beat-rate drop of 5.9 points (300 → 274). That also misses the pass bar, so the macro block is not harmless. Paired against lag 1, dropping macro **raises** the typical period Sharpe by +0.059. A stale macro print is worse than no macro print. Lag 1 is still the required baseline, because it is the join that could have been traded at 15:00 Riyadh. Later tasks use lag 1.

The edge is smaller and it is not gone. Under lag 1 the strategy still beats buy-and-hold in 270 of 438 periods, median gap +0.109 Sharpe and +3.21 CAGR points.

Published Study C (the October tape, same leaky join) was 290/438, median gap +0.187, median Sharpe 0.404 vs 0.244, median CAGR 6.40% vs 1.81%. This download's lag 0 is close to that (300/438, median gap +0.216) and is the right "before" for the paired test. Al Rajhi full window: lag 0 Sharpe 1.310 vs buy-and-hold 0.877; lag 1 Sharpe 0.985 vs the same 0.877. Data hash `affdaabf32a08d37`.

## 4. Headline after accepted fixes

Tasks 1 and 2 are in. The running baseline is lag 1, jumps with \|adjusted return\| > 15% removed from the price path, labels on those sessions dropped from training, fill at the close, 2 bps, seed 42.

| | Published Study C (leaky join, October tape) | This tape, lag 0 | Lag 1 | Lag 1, jumps neutralized |
| --- | --- | --- | --- | --- |
| Periods beating buy-and-hold | 290/438 (66%) | 300/438 (68.5%) | 270/438 (61.6%) | 274/438 (62.6%) |
| Median period Sharpe | 0.404 | 0.435 | 0.327 | 0.327 |
| Median period gap | +0.187 | +0.216 | +0.108 | +0.138 |
| Median period CAGR | 6.40% | 7.28% | 4.66% | 4.94% |
| Median period buy-and-hold CAGR | 1.81% | 1.98% | 1.98% | 1.70% |
| Full-window names beating buy-and-hold | 114/155 | 114/155 | 97/155 | 112/155 |
| Full-window median Sharpe | 0.361 | 0.393 | 0.281 | 0.311 |
| Full-window median gap | +0.177 | +0.185 | +0.078 | +0.118 |
| Full-window median CAGR | 5.96% | 6.74% | 3.56% | 4.44% |

The two data fixes cut the published period gap from +0.187 to +0.138. Most of that cut is the macro lag, not the price jumps. That +0.138 still assumes the position is on at the close that produced the signal. Entering at the next open, on the same cleaned lag-1 tape, leaves a median period gap of **+0.017**, which is 12% of the close-fill gap. A one-bar delay leaves **−0.033**. The close-to-close headline is not a fill that can be traded after seeing the close.

| | Close fill (research upper bound) | Next open | Position delayed one bar |
| --- | --- | --- | --- |
| Periods beating buy-and-hold | 274/438 (62.6%) | 227/438 (51.8%) | 202/438 (46.1%) |
| Median period Sharpe | 0.327 | 0.205 | 0.170 |
| Median period gap | +0.138 | +0.017 | −0.033 |
| Median period CAGR gap | +3.51 | +1.16 | −0.14 |
| Full-window names beating buy-and-hold | 112/155 | 80/155 | 60/155 |
| Full-window median gap | +0.118 | +0.009 | −0.049 |

## 5. Task 2 — corporate-action contamination

Tadawul's daily limit is ±10%. An adjusted close-to-close move larger than 15% is not a normal continuous-trading print. On the tape the model sees (`daily_study/cache/`, adjusted OHLC, volume 0 dropped):

- 225 such sessions, 85 of 155 names, 87 of them on or after 2016-01-01 across 47 names.
- After neutralization, every name's annualized volatility (√250) is between the original clean names and 51%. Median 33%. No remaining \|return\| above 15%. Before the fix, Saudi Fisheries was 95% and Methanol Chemicals 85%.

The raw Yahoo file, before the volume filter and the adjustment ratio, is much worse. Al Rajhi's raw close has 40 days above 15% in 2010–2013, including +270% prints that reverse the next session. The adjusted pipeline series for Al Rajhi and Jarir has **zero** days above 15%, and weekly volatility on the current adjusted tape is 22% for both. Study A's published annualized vols (Jarir 439%, Al Rajhi 199%, Maaden 105%, Tawuniya 97%) do not reproduce on this download. Those four numbers were a property of the tape Study A stored, not of the adjusted series used here. Study A's weekly model was not refit in this pass. The 12 TASI names inside Study B are inside the 155 and are covered by the daily rerun below.

Cross-checks against Tadawul actions, on the adjusted series the model actually uses:

- Maaden (1211), 8 Jun 2023 +51% and 11 Jun 2023 −34%. Shareholders approved a 1-for-2 bonus on 7 Jun 2023; Edaa implemented it on 12 Jun 2023. A 1-for-2 bonus cuts the price by one third. The adjusted series contains a spurious +51% and then the −33% ex-date. Both are above the limit.
- Tawuniya (8010), 6 Sep 2026 +50% and 7 Sep 2026 −33.5%. The 3 Sep 2026 EGM approved a 1-for-2 bonus; the exchange reset the 6 Sep limit off SAR 90.65; Edaa credited the shares on 8 Sep 2026. Same broken pattern.
- Saudi Fisheries (6050). The CMA approved a capital reduction from SAR 400 million to SAR 66.986 million. The adjusted series in Aug–Sep 2026 has −71.5%, +238%, −69%, and +203%. Those are not tradable limit moves.

Neutralization scales the flagged session's OHLC so its close equals the previous close, and scales every later price by the same factor so later returns are unchanged. The label on the previous session is dropped from training. Cost stays 2 bps. Macro lag stays 1. Commit `76280546ba16f4024d1663aba2d3a0823f99093f`. Finished 31 Oct 2026, 14:06 Asia/Riyadh. 155/155 succeeded. 85 names had at least one dropped label (225 labels).

Pass bar: beat rate, median Sharpe gap, and median CAGR gap each move by less than 20% relative versus lag 1.

| Headline | Lag 1 | Lag 1, cleaned | Relative change |
| --- | --- | --- | --- |
| Period beat rate | 270/438 (61.6%) | 274/438 (62.6%) | +1.5% |
| Period median Sharpe gap | +0.108 | +0.138 | **+28%** |
| Period median CAGR gap | +3.21 | +3.51 | +9% |
| Full-window beat rate | 97/155 | 112/155 | +15% |
| Full-window median Sharpe gap | +0.078 | +0.118 | **+51%** |
| Full-window median CAGR gap | +3.11 | +3.39 | +9% |

The Sharpe-gap headlines miss the 20% bar. The paired median of (cleaned − lag 1) is **0.000** on both Sharpe and the gap, because 221 of 438 periods have no flagged session and do not move. The level of the median moves because the contaminated minority shifts the center of the distribution. Buy-and-hold is not held fixed: the max absolute change in a period's buy-and-hold Sharpe is 1.36.

Names, full window, lag 1 → cleaned:

- Tawuniya: Sharpe gap +0.313 → +0.035, CAGR gap +11.63 → +0.76. In 2022–2026 the gap goes from +0.412 to −0.046 and the CAGR gap from +18.37 to −2.62. The recent edge was the botched bonus.
- Maaden: full-window gap +0.066 → −0.042, so it stops beating buy-and-hold. The 2022–2026 CAGR gap goes from +10.34 to −0.04. The June 2023 bonus was the edge in that window. 2016–2021 is unchanged, because the jumps are not in those periods.
- Saudi Fisheries: strategy Sharpe 0.524 → 0.503, but buy-and-hold Sharpe 0.234 → −0.062, so the Sharpe gap **rises** (+0.290 → +0.565). The CAGR gap is cut from +39.9 to +22.6, and in 2022–2026 from +64.9 to +31.2. About half of that CAGR gap was the bad prints. A large remainder is still there, on a series whose cleaned vol is 45%.
- AFG: Sharpe gap +0.579 → +0.272. Still ahead of its own buy-and-hold, which stays negative.
- Amiantit: gap +0.505 → +0.482. One flagged day does not explain it.
- Naseej: gap +0.374 → +0.732. Removing the jumps makes buy-and-hold worse. The edge is not a free ride on a down print.
- Tihama: gap +0.811 → +0.778.
- Jarir and Al Rajhi: no flagged days. Unchanged. Jarir still trails buy-and-hold (gap −0.057).

Three full windows flip from ahead of buy-and-hold to behind: 2120 (gap +0.131 → −0.189), Maaden, and 8020 (+0.090 → −0.062). The median name does not lose its edge. The names whose recent window was a mis-adjusted bonus do.

## 6. Task 3 — execution timing

The label in the published pipeline is close(t) → close(t+1), and the features at t include close(t). A decision that uses the closing print can be filled in the closing auction. It cannot be filled in the continuous session that has already ended. Two flags, both on the cleaned lag-1 tape, cost still 2 bps, seed 42:

- `--fill next_open`: the signal at close t is entered at open t+1 and earns open(t+1) → open(t+2). Buy-and-hold is the same open-to-open return.
- `--fill delay1`: the position is applied one full bar later and earns close(t+1) → close(t+2).

155/155 names, 438 periods, both fills. Paired against the cleaned close fill. Finished 31 Oct 2026, 14:23 and 14:37 Asia/Riyadh.

Pass bar: the edge over buy-and-hold keeps at least 60% of the close-fill median Sharpe gap.

Next open keeps **12%** of the period gap (+0.138 → +0.017) and **8%** of the full-window gap (+0.118 → +0.009). Paired median change in the period gap is −0.121. The beat rate falls from 62.6% to 51.8%. One-bar delay does not keep any of it: the period median gap is −0.033 (202/438 still ahead) and the full-window median gap is −0.049 (60/155). **CONFIRMED.** The close-to-close result does not survive a fill that happens after the close.

Where the close-fill return sits, using the cleaned close-fill positions already saved (not a new model). For each name, from the full-window start, the position at close t is applied separately to close(t)→open(t+1) and to open(t+1)→close(t+1):

- Median strategy mean: overnight +4.5 bps, intraday −1.3 bps, close-to-close +3.7 bps. The overnight piece is 0.99 of the close-to-close mean.
- Median buy-and-hold mean: overnight +5.8 bps, intraday −3.9 bps.
- Median Sharpe of those pieces: strategy overnight 0.87 vs buy-and-hold overnight 0.93; strategy intraday −0.13 vs buy-and-hold intraday −0.36.
- On 111 of 155 names the overnight mean is larger in absolute value than the intraday mean.

The book earns its close-to-close average by being long the overnight gap. The open-to-open rerun, which never collects that gap, gives the gap back. The intraday piece of the close-fill book is a small loss, less bad than buy-and-hold's intraday loss, and it is not large enough to carry a next-open strategy.

The 102-asset daily set was not rerun under the three fills in this pass. The TASI result is the one the pass rule was aimed at, and it already fails.

Phase 2 is not started. The instruction is to run it only after tasks 1–3 pass or are fixed. Task 1 is fixed by lag 1. Task 2 is fixed by dropping the bad prints, and the median edge survived that fix. Task 3 failed and was not repaired: there is no fill, other than the close itself, that keeps 60% of the gap.

## 7. Task 4 — Study A against Study B

Study A (`multi_asset_backtest.py`) and Study B (`daily_vs_weekly.py`) are not the same weekly test.

Where Study A uses the evaluation sample:

- Early stopping for XGBoost, LightGBM, and CatBoost uses the test fold as `eval_set` (lines 312, 319, 326).
- The t=opt threshold maximizes Sharpe on the first half of the out-of-sample predictions (lines 509–516). The published backtest then scores the **full** prediction series, including that half (lines 531–532).
- The Scaled divisor is the 90th percentile of the **full** prediction series (line 523).
- Missing values are forward-filled and back-filled (`impute_nans`, line 257).

Study B fits the threshold and the Scaled divisor on a calibration block that ends before evaluation (`run_mode`, lines 814–816). Early stopping uses the trailing part of the training window only (`walk_forward`, lines 656–664).

Published numbers, same 102 names, weekly:

| | Study A mean | Study A median | Study B mean | Study B median |
| --- | --- | --- | --- | --- |
| Best of the four | 0.921 | 0.901 | 0.521 | 0.532 |
| Binary (t=opt) | 0.864 | 0.836 | 0.379 | 0.342 |
| Binary (t=0) | 0.754 | 0.752 | 0.437 | 0.470 |
| Buy-and-hold | 0.509 | 0.494 | 0.478 | 0.454 |

Mean versus median inside Study A is 0.020 on the best Sharpe (0.921 vs 0.901). That is not the 0.40 gap versus Study B. Reporting a mean instead of a median is rejected as the explanation.

What the two protocols do to the win rate: Study A Binary (t=opt) beats buy-and-hold on **100 of 102** names. Study B's frozen weekly t=opt beats buy-and-hold on **35 of 102**, and its mean Sharpe (0.379) is below its own buy-and-hold mean (0.478). t=opt is in the market less than 20% of the time on 25 of 102 Study B weekly runs.

A refit that holds the model fixed and changes only the calendar, then only the threshold rule, was not run. The 0.40 gap is therefore not split into "window" versus "leakage" in Sharpe points. Both are in the comparison: the samples are not the same window, and Study A's published Sharpe includes the block the threshold was fit on. The code is enough to withdraw the number. It is not enough to say what fraction of 0.40 is each cause.

Study A's Tier 1 (use t=opt, expected mean Sharpe 0.864, beats buy-and-hold for about 95%), Tier 2, and Tier 3 are **withdrawn** as expectations for a live book. They describe a score that was allowed to see part of the sample it reports.

## 8. Task 5 — selection meta-model

Labels are the Study A strategy with the highest saved Sharpe. "Always predict Binary (t=opt)" is the majority class: 51 of 102 exact (50.0%), and that label is one of the two best strategies on 80 of 102 (78.4%).

Leave-one-out, same features and hyperparameters as `generate_expanded_report.py`:

| Rule | Exact | Top-2 |
| --- | --- | --- |
| Always t=opt | 51/102 (50.0%) | 80/102 (78.4%) |
| Tree, depth 3 | 48/102 (47.1%) | 74/102 (72.5%) |
| Tree, depth chosen inside each training fold | 48/102 (47.1%) | 74/102 (72.5%) |
| Random forest, 100 trees, depth 4 | 50/102 (49.0%) | 77/102 (75.5%) |

The nested tree picks depth 3 on 96 of 102 folds, depth 4 on 5, depth 5 on 1. Choosing depth on the same leave-one-out that is then reported does not change the result here.

One thousand label shuffles of the depth-3 tree: 224 shuffles had leave-one-out exact accuracy at least as high as 48. p = (224+1)/(1000+1) = **0.225**. The null mean is 41.6 correct. The tree is not distinguishable from a shuffled label, and it is below the majority class.

The forest's 1,000 shuffles were not run. One leave-one-out takes 7.6 seconds, so a thousand takes about two hours. It is already below the majority class on both exact and top-2 accuracy, so it fails the pass rule whether or not a permutation would call it better than a shuffled label.

Pass rule: beat the majority baseline and have permutation p < 0.05. **CONFIRMED, no skill.** The selection framework does not belong in the conclusions. Picking t=opt because a tree said so is worse than picking t=opt because it was the most common winner in this sample, and that common-winner rule is itself fit on the same Sharpes (section 7).

## 9. Run log

| Config | Seed | Commit | Data hash | Finished (Asia/Riyadh) |
| --- | --- | --- | --- | --- |
| Lag 0, fill close, 2 bps, 155 names | 42 | `e3af634d8d9629f149327e74eff313bd7bb2b9a6` | per name, Al Rajhi `affdaabf32a08d37` | 31 Oct 2026, 13:22 |
| Lag 1, same | 42 | same | same hash, different join | 31 Oct 2026, 13:34 |
| Macro removed, same | 42 | same | same hash | 31 Oct 2026, 13:49 |
| Lag 1, jumps neutralized, 155 names | 42 | `76280546ba16f4024d1663aba2d3a0823f99093f` | same prices before the jump scaling | 31 Oct 2026, 14:06 |
| Same, fill next open | 42 | `844a5ab6228d175dc37e760cae184f0ceaebb7a9` | cleaned prices, open-to-open label | 31 Oct 2026, 14:23 |
| Same, fill delayed one bar | 42 | same | cleaned prices, close(t+1)→close(t+2) label | 31 Oct 2026, 14:37 |
| Study A vs B and the selection tree | 42 for the tree | no new model fit; characteristics and `daily_universe/` already on this branch | `study/expanded_characteristics.csv` | same day, no retrain |

Per-name JSON: `validation/runs/`. Predictions are local and gitignored. Prices are in `daily_study/cache/` and are gitignored.

## 10. Still open

Tasks 6 through 14 were not run. Phase 2 was not started, because task 3 failed the next-open bar and was not repaired. Nothing in those later tasks can put the close-to-close TASI gap back into a fill that happens after the close. The conclusions that are already withdrawn are Study A's tier list and the strategy-selection tree and forest. The TASI daily Binary (t=0) close-to-close edge, after the lag-1 and jump fixes, is a research upper bound. At the next open the median period gap is +0.017.

