"""Track 1 + Track 2 backtest harness: walks `run_daily_forecast` forward
one calendar day at a time across a test window, comparing pre-buffer
combined per-drug demand to actual per-drug consumption and reporting WAPE.

See CONTEXT.md ("Safety Stock", "Decision Thresholds (provisional)"),
docs/adr/0003-mart3-population-and-backoff-thresholds.md ("no Track 2
evaluation harness exists yet in the repo"), and issue #18 for the design
this encodes. Consumes `pipeline.inference.run_daily_forecast`'s public seam
unchanged -- no separate reimplementation of the combination/allocation
logic, so this harness can't silently drift from what actually ships.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .inference import TRACK1_DEMAND_COL, TRACK2_STAT_COL, run_daily_forecast
from .marts import (
    CONSUMPTION_COL,
    DRUG_ID_COL,
    SNAPSHOT_DATE_COL,
    VISIT_DATE_COL,
    build_mart1_training_set,
    split_mart1_training_set,
)

# This module's own output columns.
PREDICTED_COL = "예측수요"
ACTUAL_COL = "실제소모량"
ERROR_COL = "절대오차"
BACKTEST_DAILY_COLUMNS = [SNAPSHOT_DATE_COL, DRUG_ID_COL, PREDICTED_COL, ACTUAL_COL, ERROR_COL]


class BacktestSummary(NamedTuple):
    wape: float
    total_predicted: float
    total_actual: float


class BacktestResult(NamedTuple):
    daily: pd.DataFrame
    summary: BacktestSummary


def run_backtest(raw_visits: pd.DataFrame, model, test_dates=None) -> BacktestResult:
    """Walks `run_daily_forecast(raw_visits, as_of_date, model)` forward one
    calendar day at a time across `test_dates`, comparing each day's
    pre-buffer combined demand (`track1_기댓값 + track2_통계값`) to actual
    per-drug consumption on the target date (`as_of_date + 1 day`).

    `test_dates` defaults to every calendar date (min to max, daily) in
    `split_mart1_training_set(build_mart1_training_set(raw_visits)).test`'s
    기준일자 range -- production's own test window -- but is overridable for
    smaller/faster runs and tests. `model` is used as-is across every
    as-of-date; this function never trains or retrains it (see
    `pipeline.model.train_track1_model`).

    Returns a `BacktestResult`: `daily` (`BACKTEST_DAILY_COLUMNS`, one row
    per as-of-date x drug -- predicted/actual/absolute error) and `summary`
    (overall WAPE across every row in `daily`, plus total predicted/actual
    volume for context). WAPE is `sum(abs(predicted - actual)) /
    sum(actual)`; `nan` if `daily` is empty or nothing was actually consumed
    across the whole window.

    A drug either track predicts but that's never actually dispensed on its
    target date contributes 0 actual, not a dropped row; a drug actually
    dispensed but neither track ever mentions contributes 0 predicted --
    matching `_combine_order_quantities`'s own zero-filled outer union.
    """
    if test_dates is None:
        test_dates = _default_test_dates(raw_visits)

    daily_frames = [_backtest_one_day(raw_visits, as_of_date, model) for as_of_date in test_dates]
    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else _empty_daily()
    return BacktestResult(daily=daily, summary=_summarize(daily))


def _empty_daily() -> pd.DataFrame:
    daily = pd.DataFrame(columns=BACKTEST_DAILY_COLUMNS)
    daily[SNAPSHOT_DATE_COL] = pd.to_datetime(daily[SNAPSHOT_DATE_COL])
    for col in (PREDICTED_COL, ACTUAL_COL, ERROR_COL):
        daily[col] = daily[col].astype(float)
    return daily


def _default_test_dates(raw_visits: pd.DataFrame) -> pd.DatetimeIndex:
    training_set = build_mart1_training_set(raw_visits)
    test = split_mart1_training_set(training_set).test[SNAPSHOT_DATE_COL]
    if test.empty:
        return pd.DatetimeIndex([])
    return pd.date_range(test.min(), test.max(), freq="D")


def _backtest_one_day(raw_visits: pd.DataFrame, as_of_date, model) -> pd.DataFrame:
    as_of_date = pd.Timestamp(as_of_date)
    order_quantities = run_daily_forecast(raw_visits, as_of_date, model).order_quantities

    predicted = order_quantities[[DRUG_ID_COL]].assign(
        **{PREDICTED_COL: order_quantities[TRACK1_DEMAND_COL] + order_quantities[TRACK2_STAT_COL]}
    )
    actual = _actual_drug_demand(raw_visits, as_of_date + pd.Timedelta(days=1))

    daily = predicted.merge(actual, on=DRUG_ID_COL, how="outer")
    daily[PREDICTED_COL] = daily[PREDICTED_COL].astype(float).fillna(0.0)
    daily[ACTUAL_COL] = daily[ACTUAL_COL].astype(float).fillna(0.0)
    daily[ERROR_COL] = (daily[PREDICTED_COL] - daily[ACTUAL_COL]).abs()
    daily.insert(0, SNAPSHOT_DATE_COL, as_of_date)
    return daily[BACKTEST_DAILY_COLUMNS]


def _actual_drug_demand(raw_visits: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Actual per-drug demand on `target_date`: sum of 소모량 across every
    visit dated exactly `target_date`, regardless of which population
    (Chronic or Acute) the visiting customer belongs to (see CONTEXT.md
    "Point-in-Time Correctness" -- this is ground truth, not a as-of-date
    feature, so no leakage guard applies here)."""
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    target_visits = raw_visits.loc[visit_dates == target_date].dropna(subset=[DRUG_ID_COL])
    return (
        target_visits.groupby(DRUG_ID_COL, as_index=False)[CONSUMPTION_COL]
        .sum()
        .rename(columns={CONSUMPTION_COL: ACTUAL_COL})
    )


def _summarize(daily: pd.DataFrame) -> BacktestSummary:
    total_predicted = daily[PREDICTED_COL].sum()
    total_actual = daily[ACTUAL_COL].sum()
    total_error = daily[ERROR_COL].sum()
    wape = (total_error / total_actual) if total_actual else float("nan")
    return BacktestSummary(wape=wape, total_predicted=total_predicted, total_actual=total_actual)
