"""Synthetic proof of the Saudi/US session rule. No market data."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from validation.macro_clock import lag_macro_to_sessions, prepare_macro, same_day_leak_counts

macro = pd.DataFrame(
    {"VIX_Close": [10.0, 11.0, 12.0, 13.0, 14.0]},
    index=pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-11"]),
)
sessions = pd.to_datetime(["2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"])
values, prints = lag_macro_to_sessions(macro, sessions)
expect = {
    "2024-01-07": 11.0,  # Sunday sees Friday
    "2024-01-08": 11.0,  # Monday does not see Monday
    "2024-01-09": 12.0,  # Tuesday sees Monday
    "2024-01-10": 13.0,  # Wednesday sees Tuesday
    "2024-01-11": 13.0,  # Thursday does not see Thursday
}
for day, level in expect.items():
    got = float(values.loc[pd.Timestamp(day), "VIX_Close"])
    if got != level:
        raise SystemExit(f"{day} expected {level} got {got}")

# Shifting one ROW on the US calendar, then forward-filling onto Tadawul,
# puts Thursday's close on Friday's label. Sunday then sees Thursday and
# misses Friday. That is the calendar-mismatch bug. Do not use it.
row_shift = macro.shift(1)
sunday = row_shift.reindex(sessions, method="ffill").loc[pd.Timestamp("2024-01-07"), "VIX_Close"]
if float(sunday) == 11.0:
    raise SystemExit("row-shift unexpectedly kept Friday's close")
if float(sunday) != 10.0:
    raise SystemExit(f"row-shift Sunday expected Thursday's 10, got {sunday}")

lag0 = prepare_macro(macro, sessions, 0)
if not lag0.index.equals(macro.index):
    raise SystemExit("lag 0 must not reindex the vendor frame")
empty = prepare_macro(macro, sessions, "none")
if empty.shape[1] != 0:
    raise SystemExit("macro removal left columns")

counts = same_day_leak_counts(macro, sessions)
print(counts.to_string(index=False))
print("clock checks passed")
