"""The pharmacy's closed-day calendar: which days nobody can visit, so Track 1
has no Chronic visits to predict on them (see CONTEXT.md "Pharmacy Calendar"
and issue #39).

Inside the extract, a day is closed exactly when nothing was dispensed on it
-- on the v0.3 extract that is every Sunday but four, plus most statutory
public holidays, but not all of them, and not the same ones every year. Past
the extract's last 내방일 nothing is observed yet, so a day is closed if it's
a Sunday or the pharmacist lists it (public holidays and the pharmacy's own
closures). Closures are planned in advance, so reading a past day's closure
from the extract when backtesting it isn't leakage: the pharmacist knew it
the day before.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

# `pipeline.marts.VISIT_DATE_COL`, repeated here because marts imports this
# module.
_VISIT_DATE_COL = "내방일"
_SUNDAY = 6


@dataclass(frozen=True)
class PharmacyCalendar:
    """`closed_days`: every day known to be closed. `observed_through`: the
    last day the extract covers -- later days are also closed on Sundays;
    None for a calendar with no observations (`always_open`)."""

    closed_days: pd.DatetimeIndex
    observed_through: pd.Timestamp | None

    @classmethod
    def from_visits(
        cls, raw_visits: pd.DataFrame, closure_dates: Iterable = ()
    ) -> PharmacyCalendar:
        """The calendar `raw_visits` implies: every day between its first and
        last 내방일 with no dispensing is closed, as is every date in
        `closure_dates` (the pharmacist's list, inside the extract or past
        it); past the last 내방일, so are Sundays.

        `raw_visits` must be the full extract, not a sample: a day missing
        from it reads as a closure."""
        visit_days = pd.to_datetime(raw_visits[_VISIT_DATE_COL]).dt.normalize().dropna()
        if visit_days.empty:
            observed = pd.DatetimeIndex([])
            observed_through = None
        else:
            span = pd.date_range(visit_days.min(), visit_days.max(), freq="D")
            observed = span.difference(pd.DatetimeIndex(visit_days.unique()))
            observed_through = visit_days.max()
        listed = pd.DatetimeIndex(pd.to_datetime(list(closure_dates))).normalize()
        return cls(closed_days=observed.union(listed), observed_through=observed_through)

    @classmethod
    def always_open(cls) -> PharmacyCalendar:
        """No known closures at all -- what Track 1 assumes when it's given
        no calendar."""
        return cls(closed_days=pd.DatetimeIndex([]), observed_through=None)

    def is_closed(self, dates: pd.Series) -> pd.Series:
        """Per date in `dates`, whether the pharmacy is closed that day, as a
        bool Series sharing `dates`' index."""
        days = pd.to_datetime(dates).dt.normalize()
        closed = days.isin(self.closed_days)
        if self.observed_through is not None:
            closed |= (days > self.observed_through) & (days.dt.dayofweek == _SUNDAY)
        return closed.astype(bool)

    def is_closed_on(self, date: pd.Timestamp | str) -> bool:
        """Whether the pharmacy is closed on the single day `date`."""
        return bool(self.is_closed(pd.Series([pd.Timestamp(date)])).iloc[0])
