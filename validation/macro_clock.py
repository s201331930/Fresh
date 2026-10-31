"""Cross-timezone availability of the macro series used by the daily model.

The pipeline joins each macro print on the calendar date and forward-fills
(daily_vs_weekly.add_macro). That is lag 0. Lag 1 keeps only a print whose
date is strictly before the Saudi session, which is the last close a 15:00
Riyadh decision can have seen.

US and ICE settlements are not in the Yahoo daily file. The clock below is the
exchange convention, not a timestamp read from the vendor.
"""

from __future__ import annotations

import pandas as pd

# Tadawul cash close. Sunday–Thursday.
TASI_CLOSE_RIYADH = "15:00 Asia/Riyadh"


# source_tz, event, local clock, Riyadh clock while the US is on EDT (UTC-4)
# and while it is on EST (UTC-5). All of these are after 15:00 Riyadh.
SERIES = [
    {
        "series": "VIX",
        "yahoo": "^VIX",
        "source_tz": "America/New_York",
        "event": "CBOE VIX cash close",
        "local_time": "16:15",
        "riyadh_time_edt": "23:15 same calendar date",
        "riyadh_time_est": "00:15 next calendar date",
        "known_by_tasi_close": "no",
        "lag0_join": "exact calendar date, then ffill",
        "lag1_rule": "last print with date < Saudi session",
    },
    {
        "series": "TNX",
        "yahoo": "^TNX",
        "source_tz": "America/New_York",
        "event": "US Treasury 10-year cash close",
        "local_time": "16:00",
        "riyadh_time_edt": "23:00 same calendar date",
        "riyadh_time_est": "00:00 next calendar date",
        "known_by_tasi_close": "no",
        "lag0_join": "exact calendar date, then ffill",
        "lag1_rule": "last print with date < Saudi session",
    },
    {
        "series": "DXY",
        "yahoo": "DX-Y.NYB",
        "source_tz": "America/New_York",
        "event": "ICE dollar index futures settle",
        "local_time": "17:00",
        "riyadh_time_edt": "00:00 next calendar date",
        "riyadh_time_est": "01:00 next calendar date",
        "known_by_tasi_close": "no",
        "lag0_join": "exact calendar date, then ffill",
        "lag1_rule": "last print with date < Saudi session",
    },
    {
        "series": "Gold",
        "yahoo": "GC=F",
        "source_tz": "America/New_York",
        "event": "COMEX gold settlement",
        "local_time": "13:30",
        "riyadh_time_edt": "20:30 same calendar date",
        "riyadh_time_est": "21:30 same calendar date",
        "known_by_tasi_close": "no",
        "lag0_join": "exact calendar date, then ffill",
        "lag1_rule": "last print with date < Saudi session",
    },
    {
        "series": "Brent",
        "yahoo": "BZ=F",
        "source_tz": "Europe/London",
        "event": "ICE Brent futures settlement",
        "local_time": "19:30",
        "riyadh_time_edt": "21:30 same calendar date",
        "riyadh_time_est": "22:30 same calendar date",
        "known_by_tasi_close": "no",
        "lag0_join": "exact calendar date, then ffill",
        "lag1_rule": "last print with date < Saudi session",
    },
]


def clock_table():
    return pd.DataFrame(SERIES)


def _sessions(session_index):
    sessions = pd.DatetimeIndex(pd.to_datetime(session_index))
    if getattr(sessions, "tz", None) is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def _asof_column(series, sessions, allow_exact):
    """Last non-null print on or before the session, or strictly before it."""
    clean = series.dropna().astype(float)
    clean.index = pd.to_datetime(clean.index).tz_localize(None).normalize()
    clean = clean[~clean.index.duplicated(keep="last")].sort_index()
    left = pd.DataFrame({"session": pd.DatetimeIndex(sessions).unique().sort_values()})
    right = clean.rename("value").reset_index()
    date_col = right.columns[0]
    right = right.rename(columns={date_col: "print_date"}).sort_values("print_date")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="session",
        right_on="print_date",
        direction="backward",
        allow_exact_matches=allow_exact,
    )
    session_index = pd.DatetimeIndex(aligned["session"])
    values = pd.Series(aligned["value"].values, index=session_index).reindex(sessions)
    prints = pd.Series(aligned["print_date"].values, index=session_index).reindex(sessions)
    return values, prints


def lag_macro_to_sessions(macro, session_index):
    """Last non-null print of each series strictly before each session.

    Sunday takes Friday. Thursday takes Wednesday (or the prior print if
    Wednesday is missing). A same-calendar-date print is excluded. A date that
    exists only because another series printed does not count.
    """
    sessions = _sessions(session_index)
    values = pd.DataFrame(index=sessions)
    prints = pd.DataFrame(index=sessions)
    for col in macro.columns:
        values[col], prints[col] = _asof_column(macro[col], sessions, allow_exact=False)
    return values, prints


def lag0_print_dates(macro, session_index):
    """Print date the existing join would use: last non-null print on or before the session."""
    sessions = _sessions(session_index)
    prints = pd.DataFrame(index=sessions)
    for col in macro.columns:
        _, prints[col] = _asof_column(macro[col], sessions, allow_exact=True)
    if prints.shape[1] == 1:
        return prints.iloc[:, 0]
    return prints


def prepare_macro(macro, session_index, macro_lag):
    """Return the frame passed to add_macro.

    macro_lag 0 leaves the vendor calendar untouched, so add_macro's exact
    join and forward-fill stay the published behavior.
    macro_lag 1 pre-aligns onto session dates with the strict-before rule.
    'none' drops every macro column.
    """
    if macro_lag in (0, "0"):
        return macro
    if macro_lag in (1, "1"):
        values, _ = lag_macro_to_sessions(macro, session_index)
        return values
    if macro_lag in ("none", None):
        return macro.iloc[:, 0:0]
    raise ValueError(f"macro_lag must be 0, 1, or none, got {macro_lag!r}")


def same_day_leak_counts(macro, session_index):
    """Sessions where lag 0 uses a print dated on the session itself.

    Pass one column. Counts use that series' own non-null dates.
    """
    if macro.shape[1] != 1:
        raise ValueError("same_day_leak_counts expects a single series")
    lag0 = lag0_print_dates(macro, session_index)
    _, lag1 = lag_macro_to_sessions(macro, session_index)
    lag1 = lag1.iloc[:, 0]
    both = pd.DataFrame({"lag0": lag0, "lag1": lag1}).dropna()
    same = both["lag0"].to_numpy() == pd.to_datetime(both.index).to_numpy()
    same = pd.Series(same, index=both.index)
    dow = pd.DatetimeIndex(both.index).day_name()
    rows = []
    for name in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
        mask = dow == name
        if int(mask.sum()) == 0:
            continue
        rows.append({
            "weekday": name,
            "sessions": int(mask.sum()),
            "lag0_uses_same_calendar_print": int((same & mask).sum()),
            "lag0_differs_from_lag1": int((both.loc[mask, "lag0"] != both.loc[mask, "lag1"]).sum()),
        })
    return pd.DataFrame(rows)
