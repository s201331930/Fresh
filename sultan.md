# Sultan

Sultan checks data leakage before any TASI strategy is built or scored. A feature that fails is dropped. It is not lagged into the model under another name.

## The rule

A TASI order for a trading day may use only information that was already public before that day opened. Tadawul trades Sunday through Thursday. The open is 10:00 Asia/Riyadh. Saudi Arabia does not use daylight saving time.

If a series has no print time, or the print time cannot be shown to fall before that open, the series is dropped for that day.

## Why the US close fails a same-day join

The US cash session opens at 9:30 America/New_York. That is 16:30 in Riyadh while the US is on daylight time, and 17:30 in Riyadh otherwise. Either way it is after the Tadawul close. The US close of a calendar date is not known during that date's TASI session.

The last completed US cash session that is legal for each TASI day:

| TASI day | Last US cash close that may be used |
|---|---|
| Sunday | Friday |
| Monday | Friday |
| Tuesday | Monday |
| Wednesday | Tuesday |
| Thursday | Wednesday |

Monday is the easy mistake. There is no Sunday US cash session, and Monday's US session has not opened when Monday's TASI session starts. Sunday and Monday both see Friday.

The same clock covers VIX, the US 10-year yield, the dollar index, and gold when the stored value is a US daily close. SPY, sector ETFs, and US single stocks follow the cash-session row above.

## Oil

A Brent or WTI daily close from a vendor bar is not treated as known at the TASI open of the same calendar date. Those bars settle during the London afternoon or the US session, after Tadawul has opened and, for the US settlement, after Tadawul has closed.

An oil print is usable only when its own timestamp is before the TASI open. A daily bar with no timestamp is dropped for that calendar date and becomes eligible on the next TASI day only after the bar's session is over. An intraday oil move during the TASI day is not eligible for that day.

## What this does to the files we have

`factors/spy_daily_features.csv`, `factors/spy_weekly_features.csv`, and `factors/spy_mixed_features.csv` are stamped on the SPY clock. Joining them to a TASI row on the same calendar date uses the US close of that date. Sultan drops that join.

A weekly bar labeled Friday contains Friday's US close. That close is public before Sunday's open, so the completed week may be used on Sunday. It may not be used on the Thursday before that Friday close.

TASI's own prices and indicators, through the previous TASI close, are legal for the next TASI open.

## Before a strategy is scored

Sultan signs the feature list. Each US or oil column needs the print time used and the TASI day it was first allowed onto. A column without that line is out. A backtest that joins on the raw calendar date does not count.
