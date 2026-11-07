# Monthly relative strength

The daily models failed because they forecast a move that is over before the next open. Buying the first fearful day failed because the selling had not finished. What can still be traded at the next open is slower. People underreact to news for weeks. A name that has been demanded for a year, skipping the last month, tends to keep being demanded. The last month is skipped because that is the horizon on which people overreact.

## What was frozen before the run

Once a month, at the close, rank the names and buy the top 30 percent in equal weight at the next session's open. Hold the shares until the next monthly open. Cash earns nothing. A change in the book costs 2 bps. Daily jumps above 15 percent were removed first, and any open-to-open move still above 15 percent was set to zero for every book (708 Saudi name-days, 277 in the global book).

The model is a boosted tree, depth 3, 200 trees, minimum 40 names in a leaf, L2 penalty 1, seed 42, no early stopping. It is refit every month. Its label is the return from the entry open to the next entry open. A month is allowed into the fit only after that exit open has already happened, so the month just finishing is not in the training set. The first fit needs 24 finished months. Nothing in the settings was chosen on 2017–2019 or on 2020 onward.

The features are the existing scale-free ones: one-month, one-quarter, and twelve-month-skip-one-month returns, distance from the 200-session average, RSI(14), Bollinger %b, DI+ minus DI−, volume against its own year, the share of volume on down closes, drawdown from the year's high, and current volatility against the year's volatility.

The tree has to beat two books that use the same opens, the same cost, and the same 30 percent rule. One is equal weight of every investable name. The other ranks by the twelve-month return, skipping the last month, with no model. The Saudi book is the 155 names. The global book is the 90 non-Saudi names in the 102-asset universe, so the Sunday session is not mixed into a Friday clock.

"Train" here is not an in-sample fit. Every month in it was predicted by a tree that had not seen that month.

## Saudi book

First rebalance is 2 Jun 2013. The tree's rank agrees only weakly with the twelve-month return (correlation 0.13) and a little more with the last month (0.17). It reshuffles the book about 14 times a year. The plain twelve-month rank turns about 5 times a year.

| Window | Tree Sharpe | 12-month rank | Equal weight | Tree CAGR | Rank CAGR | Equal-weight CAGR |
|---|---:|---:|---:|---:|---:|---:|
| Train, Jun 2013 – 2016 | 0.32 | 0.38 | 0.24 | 5.0% | 6.2% | 3.0% |
| Validate, 2017–2019 | 0.76 | 0.47 | 0.15 | 9.6% | 5.7% | 1.0% |
| Test, 2020 – 4 Nov 2026 | 0.47 | 0.59 | 0.27 | 7.3% | 10.0% | 3.2% |

Drawdowns on the test are still large: tree −37%, twelve-month rank −37%, equal weight −42%. The validation gap for the tree is mostly 2018 (the tree +14%, equal weight −4%). One year does not make a rule.

On the test, a block bootstrap of the tree's excess over equal weight (block of 21 sessions, 400 draws, seed 42) has a 95% interval of −0.7% to +8.9% a year. The interval includes zero. The same interval for the twelve-month rank is +1.8% to +12.9%. That one sits above zero.

The test years show where the rank's wealth came from. It lagged equal weight in 2021 (+11% against +15%) and was close in 2020 and 2025. It lost much less in 2022 (−8% against −23%) and made more in 2023 (+39% against +30%) and 2024 (+28% against +12%).

## Global book

Ninety names, first rebalance 3 Mar 2003. The tree's rank lines up a little with the twelve-month return (correlation 0.21) and not with the last month.

| Window | Tree Sharpe | 12-month rank | Equal weight | Tree CAGR | Rank CAGR | Equal-weight CAGR |
|---|---:|---:|---:|---:|---:|---:|
| Train, Mar 2003 – 2016 | 0.70 | 0.92 | 0.80 | 12.0% | 15.3% | 11.2% |
| Validate, 2017–2019 | 1.31 | 1.18 | 1.47 | 20.1% | 18.3% | 14.6% |
| Test, 2020 – 5 Nov 2026 | 0.81 | 0.91 | 0.84 | 15.9% | 16.8% | 11.7% |

Both concentrated books made more money than equal weight and took a deeper or similar drawdown. The tree's test drawdown is −33% against −23% for equal weight. The twelve-month rank's is −26%. The bootstrap interval of test excess includes zero for both: tree −0.3% to +9.6% a year, rank −1.2% to +10.8%. In 2008 the tree lost 41% while equal weight lost 33% and the twelve-month rank lost 27%. Chasing the recent ranking into a crash is the failure mode, and the tree did it.

## What the split supports

The tree was the model under test. On the Saudi holdout its extra return is not distinguishable from noise, and it is behind the ranking it was supposed to improve. On the global holdout it is behind equal weight on Sharpe and ahead on wealth, with a worse drawdown, and the excess interval includes zero.

The twelve-month rank, skipping the last month, was specified beforehand as that benchmark. On Saudi names it is ahead of equal weight in all three windows, and the holdout interval for the excess sits above zero. It does not do that on the global book. It also does not remove the large drawdowns. About 46 Saudi names are held, fully invested, turned about five times a year.

No setting was changed after these numbers. A later model would have to be stopped from chasing the last month, because that is what made the tree reshuffle a monthly book. That constraint is not applied here, and these results are not a reason to pretend it was.
