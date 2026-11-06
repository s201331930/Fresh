# Demand line

One rule. Decide at the close. Trade at the next session’s open.

The old model’s average came from the overnight gap between the signal close and the next open. That gap is not available once the order waits for the open, and the edge disappeared under that fill. This rule is built the other way around: it only pays if a regime lasts many sessions, because the signal-day gap is given away on purpose.

## The rule

At the close, be long if the close is above its 200-session average. Otherwise be in cash. The average uses only past closes, and it needs 200 of them before the first trade.

That average is a business year of demand. Price holds above it only when buyers have been willing to pay more than sellers for a year. A break below it means supply has taken the market, which is also where holders give up a line they have been defending. The position is binary, long only, with no leverage and no fitted threshold. Cash earns nothing in this test.

The order is filled at the next session’s open. A buy signal today does not earn today’s close to tomorrow’s open. A sell signal today is sold at the next open, so the break itself is still held. Each unit of turnover costs 2 bps, the same convention as the earlier studies. The first entry pays that cost too.

Before the signal and the profit are computed, any close-to-close move larger than 15% is taken out of the price path. Later percentage changes are left as they were. That screen caught 225 events on 85 of the 155 Saudi names, the same events the audit found in the adjusted Yahoo tape. A bonus share is not a trade.

Nothing in the rule was chosen after seeing these results. Two hundred sessions, the 15% screen, and 2 bps were fixed first. There is no random seed.

## A real fill

Al Rajhi, 1120.SR, first entry. The close on 2 Feb 2011 was above the average, so the buy was for the next open, Sunday 6 Feb. The open was 0.62% above that Wednesday close. The strategy’s result that day is minus 2 bps of entry cost. It does not include the 0.62%. A synthetic check of the same lag is in `demand_line.py` and passed before the run.

## Saudi book

Equal weight of the 155 names that had enough history. On a day when a name has no bar it is left out of that day’s average. A name that is in cash stays in the average, at zero. The book runs from 4 Jan 2011 to 4 Nov 2026, 15.8 years. About 124 names were investable in 2011 and all 155 by 2025. Sharpe uses a zero risk-free rate and the square root of 252. A Saudi year is nearer 250 sessions, so these Sharpes are slightly high. CAGR is calendar time.

| | Demand line | Buy and hold, same opens |
|---|---:|---:|
| Sharpe | 0.710 | 0.383 |
| CAGR | 6.74% | 5.84% |
| Max drawdown | −30.4% | −59.7% |
| Worst month | −16.8% (May 2012) | −23.4% (same month) |
| Capital invested | 49% | 100% |
| Turnover | 7.2 one-way turns per name per year | — |

The typical name is a much smaller result than the book. Median Sharpe is 0.264 against a buy-and-hold 0.195. The median gap is +0.049. The rule is ahead on 91 of 155 names. The book looks better than the median name because the names fall together, and the rule steps aside on many of them at the same time.

Removing the 10 names that added the most extra wealth (4210, 4003, 2300, 4240, 2001, 2310, 4004, 1202, 7040, 4230) leaves Sharpe 0.663 against 0.384, CAGR 6.28% against 5.86%, and a drawdown of −32.5% against −59.0%. The book is not one lucky name.

The cost of waiting is already in those numbers. Across names, the median entry morning is 18 bps above the signal close. That gap is not earned. At roughly three and a half entries a year it is about 65 bps a year of forgone gap, on top of about 14 bps of the 2 bps commission.

The rule earns the difference in the years supply wins, and gives part of it back when demand is already obvious.

| Year | Invested | Strategy CAGR | Buy and hold CAGR |
|---|---:|---:|---:|
| 2015 | 24% | −6.5% | −13.6% |
| 2016 | 27% | +0.1% | −9.2% |
| 2020 | 66% | +29.9% | +44.7% |
| 2021 | 80% | +24.0% | +17.9% |
| 2022 | 24% | −1.6% | −14.7% |
| 2023 | 50% | +33.8% | +67.7% |
| 2025 | 26% | −8.7% | −24.8% |

May 2012 is the honest limit of a next-open exit. The book was still 76% invested and lost 16.8% while buy and hold lost 23.4%. The average breaks after the damage has started, and the sale waits for the next open.

## Global book

The same rule on the existing 102-asset universe, from 17 Oct 2000 to 5 Nov 2026. Twelve of the names are Saudi. Equal weight, same fill, same cost.

| | Demand line | Buy and hold |
|---|---:|---:|
| Sharpe | 0.780 | 0.541 |
| CAGR | 7.07% | 8.57% |
| Max drawdown | −17.4% | −42.7% |
| Worst month | −8.2% | — |
| Capital invested | 65% | 100% |

The path is smoother and the ending wealth is lower. The typical asset does not beat buy and hold: 42 of 102, median Sharpe gap −0.037. Dropping the 10 largest excess contributors leaves Sharpe 0.770 against 0.583 and CAGR 6.88% against 9.03%.

By class the same split shows up. Drawdowns shrink everywhere. Wealth shrinks where the asset mostly rises. US stocks made 11.4% a year against 18.7% for buy and hold, with a drawdown of −20% against −43%, invested 69% of the time. SPY itself is Sharpe 0.662 against 0.531, CAGR 6.9% against 8.5%, drawdown −24% against −55%. The Saudi sleeve inside this universe is the closer one: CAGR 9.1% against 11.0%, Sharpe 0.91 against 0.67, drawdown −24% against −46%.

## The volume check

A second version was written down before the run and was not allowed to replace the rule. It is long only when the close is above the average and, over the last 20 sessions, more shares changed hands on rising closes than on falling ones. That is the supply-and-demand check: price and effort pointing the same way.

On the Saudi book it is the better path. Sharpe 0.962, CAGR 6.97%, drawdown −20.9%, invested 37% of the time. It turns 14 times a name per year, about twice the plain rule. On the global book it is the worse path. CAGR falls to 4.39% against 8.57% for buy and hold, and only 27 of 102 names finish ahead.

It helps the choppy market and taxes the trending one. Adopting it because the Saudi number is better would be choosing the rule after seeing the result. The traded rule stays the average.

## What this does not say

Cash earned zero. Interest on the idle half would raise the CAGR. It was not added.

Two basis points is the old study convention, not a measured Tadawul spread. A wider spread matters, because a name still turns about seven times a year.

The prices are Yahoo’s adjusted series, then the 15% screen. The screen is required. Without it, bonus shares and capital reductions look like trades.

No US macro series is used. A same-day US close is not known when the Saudi market shuts.

This is one historical path of a rule with no fitted parameter. It can be rerun on a new bar without refitting. It is not a forecast that the next sixteen years will look like these.
