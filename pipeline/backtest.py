"""Track 1 + Track 2 backtest harness: walks `run_daily_forecast` forward
one calendar day at a time across a test window, comparing pre-buffer
combined per-drug demand to actual per-drug consumption and reporting WAPE.

See CONTEXT.md ("Safety Stock", "Decision Thresholds (provisional)"),
docs/adr/0003-mart3-population-and-backoff-thresholds.md (written before
this harness existed), and issue #18 for the design
this encodes. Consumes `pipeline.inference.run_daily_forecast`'s public seam
unchanged -- no separate reimplementation of the combination/allocation
logic, so this harness can't silently drift from what actually ships.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .inference import (
    TRACK1_DEMAND_COL,
    TRACK2_STAT_COL,
    VISIT_PROB_COL,
    run_daily_forecast,
)
from .marts import (
    CLOSED_TOMORROW_COL,
    CONSUMPTION_COL,
    CUSTOMER_ID_COL,
    DRUG_ID_COL,
    SNAPSHOT_DATE_COL,
    VISIT_DATE_COL,
    build_mart1_training_set,
    mart1_split_windows,
)
from .pharmacy_calendar import PharmacyCalendar

# This module's own output columns.
PREDICTED_COL = "예측수요"
ACTUAL_COL = "실제소모량"
ERROR_COL = "절대오차"
# Per-track split of PREDICTED_COL/ACTUAL_COL (see issue #25): each pair sums
# to its combined column. Track 1's actual is consumption by the customers
# Track 1 scored that day; Track 2's is everyone else's. Once Lapsed Chronic
# customers stop being scored (ADR-0005), a returning Lapsed customer's
# consumption -- forecast by neither track -- lands in Track 2's actual; split
# it out then if it skews Track 2's WAPE.
TRACK1_PREDICTED_COL = "track1_예측수요"
TRACK1_ACTUAL_COL = "track1_실제소모량"
TRACK2_PREDICTED_COL = "track2_예측수요"
TRACK2_ACTUAL_COL = "track2_실제소모량"
BACKTEST_DAILY_COLUMNS = [
    SNAPSHOT_DATE_COL,
    DRUG_ID_COL,
    PREDICTED_COL,
    ACTUAL_COL,
    ERROR_COL,
    TRACK1_PREDICTED_COL,
    TRACK1_ACTUAL_COL,
    TRACK2_PREDICTED_COL,
    TRACK2_ACTUAL_COL,
]

# Per-as-of-date Track 1 visit-probability diagnostics (see issue #24's "Done
# when" and CONTEXT.md "Visit-Probability Calibration").
SUM_VISIT_PROB_COL = "예측방문확률_합"
ACTUAL_CHRONIC_VISITS_COL = "실제_만성_내일방문수"
VISIT_LIST_SIZE_COL = "방문목록_인원"
SCORED_POPULATION_SIZE_COL = "채점_인원"
BACKTEST_PER_DATE_COLUMNS = [
    SNAPSHOT_DATE_COL,
    SUM_VISIT_PROB_COL,
    ACTUAL_CHRONIC_VISITS_COL,
    VISIT_LIST_SIZE_COL,
    SCORED_POPULATION_SIZE_COL,
    # Mart 1's own 내일_휴무: whether the target date is closed, so open- and
    # closed-day figures can be reported apart (see issue #39).
    CLOSED_TOMORROW_COL,
]


class BacktestSummary(NamedTuple):
    wape: float
    total_predicted: float
    total_actual: float
    track1_wape: float
    track2_wape: float


class BacktestResult(NamedTuple):
    daily: pd.DataFrame
    per_date: pd.DataFrame
    summary: BacktestSummary


class _BacktestDay(NamedTuple):
    daily: pd.DataFrame
    per_date: pd.DataFrame


def run_backtest(
    raw_visits: pd.DataFrame,
    model,
    test_dates=None,
    chronic_visit_prob_cutoff: float | None = None,
    pharmacy_calendar: PharmacyCalendar | None = None,
) -> BacktestResult:
    """Walks `run_daily_forecast(raw_visits, as_of_date, model)` forward one
    calendar day at a time across `test_dates`, comparing each day's
    pre-buffer combined demand (`track1_기댓값 + track2_통계값`) to actual
    per-drug consumption on the target date (`as_of_date + 1 day`).

    `test_dates` defaults to every calendar date of the Temporal Split's
    test window (`mart1_split_windows(build_mart1_training_set(raw_visits))`,
    `test_start` to `end`) -- production's own test window -- but is overridable for
    smaller/faster runs and tests. `model` is used as-is across every
    as-of-date; this function never trains or retrains it (see
    `pipeline.model.train_track1_model`), so its probabilities are whatever
    `model.predict_proba` gives -- calibrated, for a `CalibratedTrack1Model`.
    `chronic_visit_prob_cutoff` is forwarded to `run_daily_forecast`; left
    unset, that uses the model's own tuned cutoff. So is
    `pharmacy_calendar`, which zeroes Track 1 on closed target dates.

    Returns a `BacktestResult`:

    - `daily` (`BACKTEST_DAILY_COLUMNS`, one row per as-of-date x drug):
      combined predicted/actual/absolute error, plus the same predicted and
      actual split per track. Track 1's actual is the target date's
      consumption by the customers Track 1 scored as of that as-of-date;
      Track 2's actual is everyone else's, so the two always sum to the
      combined actual.
    - `per_date` (`BACKTEST_PER_DATE_COLUMNS`, one row per as-of-date): the
      sum of Track 1 visit probabilities, how many scored customers actually
      visited on the target date, the Visit List size and the scored
      population size, and whether `pharmacy_calendar` has the target date
      closed (내일_휴무).
    - `summary`: overall WAPE across every row in `daily`, total
      predicted/actual volume for context, and a WAPE per track (each
      track's predicted against its own actual).

    WAPE is `sum(abs(predicted - actual)) /
    sum(actual)`; `nan` if `daily` is empty or nothing was actually consumed
    across the whole window.

    A drug either track predicts but that's never actually dispensed on its
    target date contributes 0 actual, not a dropped row; a drug actually
    dispensed but neither track ever mentions contributes 0 predicted --
    matching `_combine_order_quantities`'s own zero-filled outer union.
    """
    if test_dates is None:
        test_dates = _default_test_dates(raw_visits)

    days = [
        _backtest_one_day(
            raw_visits, as_of_date, model, chronic_visit_prob_cutoff, pharmacy_calendar
        )
        for as_of_date in test_dates
    ]
    if days:
        daily = pd.concat([day.daily for day in days], ignore_index=True)
        per_date = pd.concat([day.per_date for day in days], ignore_index=True)
    else:
        daily, per_date = _empty_daily(), _empty_per_date()
    return BacktestResult(daily=daily, per_date=per_date, summary=_summarize(daily))


def _empty_daily() -> pd.DataFrame:
    daily = pd.DataFrame(columns=BACKTEST_DAILY_COLUMNS)
    daily[SNAPSHOT_DATE_COL] = pd.to_datetime(daily[SNAPSHOT_DATE_COL])
    float_cols = BACKTEST_DAILY_COLUMNS[BACKTEST_DAILY_COLUMNS.index(PREDICTED_COL) :]
    for col in float_cols:
        daily[col] = daily[col].astype(float)
    return daily


def _empty_per_date() -> pd.DataFrame:
    per_date = pd.DataFrame(columns=BACKTEST_PER_DATE_COLUMNS)
    per_date[SNAPSHOT_DATE_COL] = pd.to_datetime(per_date[SNAPSHOT_DATE_COL])
    per_date[SUM_VISIT_PROB_COL] = per_date[SUM_VISIT_PROB_COL].astype(float)
    for col in (ACTUAL_CHRONIC_VISITS_COL, VISIT_LIST_SIZE_COL, SCORED_POPULATION_SIZE_COL):
        per_date[col] = per_date[col].astype(int)
    per_date[CLOSED_TOMORROW_COL] = per_date[CLOSED_TOMORROW_COL].astype(bool)
    return per_date


def _default_test_dates(raw_visits: pd.DataFrame) -> pd.DatetimeIndex:
    windows = mart1_split_windows(build_mart1_training_set(raw_visits))
    if pd.isna(windows.end):
        return pd.DatetimeIndex([])
    return pd.date_range(windows.test_start, windows.end, freq="D")


def _backtest_one_day(
    raw_visits: pd.DataFrame,
    as_of_date,
    model,
    chronic_visit_prob_cutoff: float | None,
    pharmacy_calendar: PharmacyCalendar | None,
) -> _BacktestDay:
    as_of_date = pd.Timestamp(as_of_date)
    target_date = as_of_date + pd.Timedelta(days=1)
    calendar = pharmacy_calendar or PharmacyCalendar.always_open()
    forecast = run_daily_forecast(
        raw_visits,
        as_of_date,
        model,
        chronic_visit_prob_cutoff=chronic_visit_prob_cutoff,
        pharmacy_calendar=pharmacy_calendar,
    )
    scored_customer_ids = set(forecast.scored_population[CUSTOMER_ID_COL])
    target_visits = _visits_on(raw_visits, target_date)

    daily = _daily_per_drug(forecast.order_quantities, target_visits, scored_customer_ids)
    daily.insert(0, SNAPSHOT_DATE_COL, as_of_date)

    visited_customer_ids = set(target_visits[CUSTOMER_ID_COL])
    per_date = pd.DataFrame(
        {
            SNAPSHOT_DATE_COL: [as_of_date],
            SUM_VISIT_PROB_COL: [float(forecast.scored_population[VISIT_PROB_COL].sum())],
            ACTUAL_CHRONIC_VISITS_COL: [len(scored_customer_ids & visited_customer_ids)],
            VISIT_LIST_SIZE_COL: [len(forecast.visit_list)],
            SCORED_POPULATION_SIZE_COL: [len(scored_customer_ids)],
            CLOSED_TOMORROW_COL: [bool(calendar.is_closed(pd.Series([target_date])).iloc[0])],
        }
    )
    return _BacktestDay(daily=daily[BACKTEST_DAILY_COLUMNS], per_date=per_date)


def _daily_per_drug(
    order_quantities: pd.DataFrame, target_visits: pd.DataFrame, scored_customer_ids: set
) -> pd.DataFrame:
    """One as-of-date's per-drug predicted vs actual, per track and combined.
    The combined columns are the per-track sums, so the headline WAPE is the
    same pre-buffer `track1_기댓값 + track2_통계값` against all actual
    consumption it always was."""
    predicted = order_quantities[[DRUG_ID_COL]].assign(
        **{
            TRACK1_PREDICTED_COL: order_quantities[TRACK1_DEMAND_COL],
            TRACK2_PREDICTED_COL: order_quantities[TRACK2_STAT_COL],
        }
    )
    actual = _actual_drug_demand(target_visits, scored_customer_ids)

    daily = predicted.merge(actual, on=DRUG_ID_COL, how="outer")
    for col in (TRACK1_PREDICTED_COL, TRACK2_PREDICTED_COL, TRACK1_ACTUAL_COL, TRACK2_ACTUAL_COL):
        daily[col] = daily[col].astype(float).fillna(0.0)
    daily[PREDICTED_COL] = daily[TRACK1_PREDICTED_COL] + daily[TRACK2_PREDICTED_COL]
    daily[ACTUAL_COL] = daily[TRACK1_ACTUAL_COL] + daily[TRACK2_ACTUAL_COL]
    daily[ERROR_COL] = (daily[PREDICTED_COL] - daily[ACTUAL_COL]).abs()
    return daily


def _visits_on(raw_visits: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    return raw_visits.loc[visit_dates == target_date]


def _actual_drug_demand(target_visits: pd.DataFrame, scored_customer_ids: set) -> pd.DataFrame:
    """Actual per-drug demand on the target date, split by track: sum of
    소모량 across every visit in `target_visits`, attributed to Track 1 if
    the visiting customer is in `scored_customer_ids` and to Track 2
    otherwise (see CONTEXT.md "Point-in-Time Correctness" -- this is ground
    truth, not an as-of-date feature, so no leakage guard applies here)."""
    target_visits = target_visits.dropna(subset=[DRUG_ID_COL])
    is_track1 = target_visits[CUSTOMER_ID_COL].isin(scored_customer_ids)
    consumption = target_visits[CONSUMPTION_COL]
    by_track = pd.DataFrame(
        {
            DRUG_ID_COL: target_visits[DRUG_ID_COL],
            TRACK1_ACTUAL_COL: consumption.where(is_track1, 0.0),
            TRACK2_ACTUAL_COL: consumption.where(~is_track1, 0.0),
        }
    )
    return by_track.groupby(DRUG_ID_COL, as_index=False)[
        [TRACK1_ACTUAL_COL, TRACK2_ACTUAL_COL]
    ].sum()


def _summarize(daily: pd.DataFrame) -> BacktestSummary:
    return BacktestSummary(
        wape=_wape(daily[PREDICTED_COL], daily[ACTUAL_COL]),
        total_predicted=daily[PREDICTED_COL].sum(),
        total_actual=daily[ACTUAL_COL].sum(),
        track1_wape=_wape(daily[TRACK1_PREDICTED_COL], daily[TRACK1_ACTUAL_COL]),
        track2_wape=_wape(daily[TRACK2_PREDICTED_COL], daily[TRACK2_ACTUAL_COL]),
    )


def _wape(predicted: pd.Series, actual: pd.Series) -> float:
    total_actual = actual.sum()
    total_error = (predicted - actual).abs().sum()
    return (total_error / total_actual) if total_actual else float("nan")
