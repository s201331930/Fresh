# Supply, demand, and mood

The previous run was one rule scored on the whole history. This one uses the indicators already in the library, fits a single integer on a training window, and leaves the next two windows untouched.

## The rule

Six features, each already computed by `technical_indicators`:

| Feature | What it is asking |
|---|---|
| Close vs SMA(200) | Is the business year of demand still intact? |
| RSI(14) ≤ 40 | Is the crowd afraid? |
| Bollinger %b ≤ 0.35 | Is price pressed into the lower band? |
| DI− above DI+ | Does supply have the directional upper hand? |
| Volume ≥ 1.2× its 20-session average, on a down close | Was today's effort on the sell side? |
| Last five closes down | Is the pullback real, not a one-day mark? |

From cash, buy when demand is intact and at least K of the other five signs are present. Sell when demand breaks, or when RSI reaches 70 (greed). Both orders go to the next session's open. A signal does not earn the gap into that open. Turnover costs 2 bps. Closes that jump more than 15% are removed first. Cash earns nothing.

K is 2, 3, or 4. It is chosen on the training portfolio only, by the Sharpe gap against buy and hold on that window. A tie keeps the stricter K. The RSI levels, the band, and the volume multiple were not searched.

| Window | Dates | Used to choose K |
|---|---|---|
| Train | first tradable bar through 31 Dec 2016 | yes |
| Validate | 1 Jan 2017 through 31 Dec 2019 | no |
| Test | 1 Jan 2020 through the last bar | no |

Sharpe uses a zero risk-free rate and the square root of 252. CAGR is calendar time. The book is an equal weight of the names that have a bar that day.

## Saudi book, 155 names

Training already rejected every K. The least-bad was K=2, with a Sharpe gap of −0.43. K=3 was −0.57 and K=4 was −0.71. Validation and the test agreed with that rejection.

| Window | Strategy Sharpe | Buy and hold | CAGR | Buy and hold CAGR | Max drawdown | Invested |
|---|---:|---:|---:|---:|---:|---:|
| Train, 4 Jan 2011 – 29 Dec 2016 | −0.12 | 0.31 | −1.0% | 4.5% | −20% | 29% |
| Validate, 2017–2019 | 0.04 | 0.21 | 0.1% | 1.9% | −7% | 30% |
| Test, 1 Jan 2020 – 4 Nov 2026 | 0.33 | 0.53 | 1.9% | 8.9% | −14% | 28% |

On the test, 67 of 155 names beat buy and hold and the median name gap is −0.04. The book is the worse figure because it is in cash about 70% of the time while the market is rising. The smaller drawdown is the cash, not a gain in return per unit of risk: the Sharpe is lower too.

## Global book, 102 assets

Training picked the same K=2. The train Sharpe gap was +0.26 (0.74 against 0.48). Wealth was already lower in training: 4.9% a year against 7.6%. That Sharpe gap did not survive the next window.

| Window | Strategy Sharpe | Buy and hold | CAGR | Buy and hold CAGR | Max drawdown | Invested |
|---|---:|---:|---:|---:|---:|---:|
| Train, 17 Oct 2000 – 30 Dec 2016 | 0.74 | 0.48 | 4.9% | 7.6% | −12% | 44% |
| Validate, 2017–2019 | 1.09 | 1.24 | 7.8% | 16.4% | −9% | 47% |
| Test, 1 Jan 2020 – 5 Nov 2026 | 0.41 | 0.48 | 2.9% | 7.7% | −13% | 44% |

Validation Sharpe gap −0.15. Test Sharpe gap −0.08. On the test, 26 of 102 names beat buy and hold, median gap −0.19.

## The one-feature comparison

The same three windows were scored for the plain demand line: long only while the close is above the 200-session average, same next-open fill, nothing fitted. It is the baseline this rule had to beat. It was not a second search.

Saudi baseline: train Sharpe 0.62 against 0.31, validate 0.17 against 0.21, test 0.96 against 0.53. Test CAGR 9.9% against 8.9%, drawdown −19% against −41%. The middle window misses. A test win that the validation window does not support is not a confirmed rule.

Global baseline: train 0.79 against 0.48, validate 1.32 against 1.24, test 0.57 against 0.48. The gap stays positive and thin (+0.08 on the test). CAGR stays lower in every window (test 5.6% against 7.7%). Drawdown on the test is −16% against −37%.

## What the split says

Buying fear inside an uptrend, and selling into greed, lost to owning the asset on every Saudi window and on the two windows that were not used to pick K for the global book. Requiring more of the five signs made the training result worse, not better. The indicators did their job as a description of a dip. The dip was not a place to make money once the order had to wait for the next open.
