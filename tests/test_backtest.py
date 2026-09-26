import numpy as np
import pandas as pd
import pytest

from conftest import (
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.backtest import (
    ACTUAL_COL,
    BACKTEST_DAILY_COLUMNS,
    ERROR_COL,
    PREDICTED_COL,
    run_backtest,
)
from pipeline.marts import (
    DRUG_ID_COL,
    SNAPSHOT_DATE_COL,
    build_mart1_training_set,
    split_mart1_training_set,
)


class StubModel:
    """A minimal `predict_proba`-only stand-in for T1's real LightGBM model
    (see `tests/test_inference.py`'s identical stub): returns a fixed
    probability per row, regardless of which as-of-date is currently being
    scored -- every as-of-date in these fixtures presents the same single
    Chronic customer, so one fixed probability is all any of these tests
    need.
    """

    def __init__(self, probability: float):
        self._probability = probability

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self._probability)
        return np.column_stack([1 - p, p])


def _one_chronic_customer_visiting_daily(drug_id: int, consumptions: dict) -> pd.DataFrame:
    """One Chronic customer (established via an old Revisit Match, well
    before any date these tests exercise) plus one dated visit per
    `consumptions` entry (`{date_str: consumption}`), each consuming
    `drug_id`. No Acute-population noise beyond the frequency filler, so
    Track 2 never contributes and every day's Mart 1 population is exactly
    this one customer -- keeps each as-of-date's forecast reducible to a
    single number by hand.
    """
    rows = make_high_frequency_filler_visits() + make_independent_chronic_match_visits(
        customer_id=1, drug_id=901, visit_id_start=10
    )
    for i, (date, consumption) in enumerate(consumptions.items(), start=100):
        rows.append(
            make_visit_row(
                조제판매ID=i, 고객ID=1, 내방일=date, 약품ID=drug_id, 소모량=consumption
            )
        )
    return make_raw_visits(rows)


def test_one_daily_row_group_per_test_date():
    raw_visits = _one_chronic_customer_visiting_daily(
        drug_id=501, consumptions={"2024-01-15": 40.0, "2024-01-16": 30.0}
    )
    model = StubModel(0.5)

    result = run_backtest(
        raw_visits, model, test_dates=["2024-01-14", "2024-01-15"]
    )

    assert set(result.daily[SNAPSHOT_DATE_COL]) == {
        pd.Timestamp("2024-01-14"),
        pd.Timestamp("2024-01-15"),
    }


def test_actual_demand_comes_from_target_date_not_as_of_date():
    # As-of 2024-01-14, the as-of-date's own visit (consumption 999.0, dated
    # 2024-01-14 itself) must never be read as "actual" -- only the target
    # date's (2024-01-15, consumption 40.0).
    raw_visits = _one_chronic_customer_visiting_daily(
        drug_id=501, consumptions={"2024-01-14": 999.0, "2024-01-15": 40.0}
    )
    model = StubModel(0.0)  # zero predicted demand -- isolates the actual side

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14"])

    row = result.daily.set_index(DRUG_ID_COL).loc[501]
    assert row[ACTUAL_COL] == 40.0


def test_wape_matches_hand_computed_value():
    # Drug 501's only Chronic patient is customer 1, well under the default
    # rare-drug threshold (5) -- Track 1's 100%-allocation rule applies (see
    # docs/adr/0004), not probability-weighted expected value, so its
    # predicted demand is a fixed 40.0 (the customer's reference Mart 2
    # consumption) on every day the model clears the 0.3 cutoff, regardless
    # of the day's actual dispensation -- kept on a drug (777/778) no track
    # ever predicts, so predicted and actual are independently controlled
    # per day.
    #
    # The Chronic-establishing match visits below (drug 901, matching
    # `make_independent_chronic_match_visits`'s own shape) use 소모량=0.0 --
    # otherwise that drug would be *its own* rare-allocated predicted demand
    # every day too, via the same Mart 2 mechanism, complicating this test's
    # hand-computed total for no reason.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=10,
                고객ID=1,
                내방일="2023-11-01",
                다음내방일="2023-12-01",
                약품ID=901,
                소모량=0.0,
            ),
            make_visit_row(
                조제판매ID=11, 고객ID=1, 내방일="2023-11-15", 약품ID=901, 소모량=0.0
            ),
        ]
        + [
            make_visit_row(
                조제판매ID=50, 고객ID=1, 내방일="2023-06-01", 약품ID=501, 소모량=40.0
            ),
            # Day 1's actual: a one-off customer dispensed on 2024-01-15
            # (target date for as-of 2024-01-14).
            make_visit_row(
                조제판매ID=100, 고객ID=2, 내방일="2024-01-15", 약품ID=777, 소모량=25.0
            ),
            # Day 2's actual: a different one-off customer/drug dispensed on
            # 2024-01-16 (target date for as-of 2024-01-15).
            make_visit_row(
                조제판매ID=101, 고객ID=3, 내방일="2024-01-16", 약품ID=778, 소모량=10.0
            ),
        ]
    )
    model = StubModel(0.5)

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14", "2024-01-15"])

    # Day 1: predicted (drug 501) = 40.0, actual (drug 777) = 25.0 -> errors 40.0 + 25.0.
    # Day 2: predicted (drug 501) = 40.0, actual (drug 778) = 10.0 -> errors 40.0 + 10.0.
    # WAPE = sum(abs error) / sum(actual) = (65.0 + 50.0) / (25.0 + 10.0)
    assert result.summary.wape == pytest.approx(115.0 / 35.0)
    assert result.summary.total_predicted == pytest.approx(80.0)
    assert result.summary.total_actual == pytest.approx(35.0)


def test_drug_predicted_but_never_actually_dispensed_contributes_without_crashing():
    # Track 1's 100%-allocation rule (drug 501, a rare drug under the
    # default threshold) predicts positive demand, but nothing is actually
    # dispensed on the target date at all.
    raw_visits = _one_chronic_customer_visiting_daily(drug_id=501, consumptions={"2024-01-15": 40.0})
    model = StubModel(0.5)

    result = run_backtest(raw_visits, model, test_dates=["2024-01-15"])

    row = result.daily.set_index(DRUG_ID_COL).loc[501]
    assert row[PREDICTED_COL] == pytest.approx(40.0)
    assert row[ACTUAL_COL] == 0.0
    assert row[ERROR_COL] == pytest.approx(40.0)


def test_drug_actually_dispensed_but_never_predicted_contributes_without_crashing():
    # Nobody in Mart 1/2/3 has anything to say about drug 777 -- it's
    # consumed only on the target date itself, by a customer with no other
    # history at all (so never scored by Track 1, and too thin/absent for
    # Track 2 either).
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=901, visit_id_start=10)
        + [
            make_visit_row(
                조제판매ID=100, 고객ID=2, 내방일="2024-01-15", 약품ID=777, 소모량=15.0
            )
        ]
    )
    model = StubModel(0.5)

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14"])

    row = result.daily.set_index(DRUG_ID_COL).loc[777]
    assert row[PREDICTED_COL] == 0.0
    assert row[ACTUAL_COL] == 15.0
    assert row[ERROR_COL] == pytest.approx(15.0)


def test_daily_columns():
    raw_visits = _one_chronic_customer_visiting_daily(drug_id=501, consumptions={"2024-01-15": 40.0})
    model = StubModel(0.5)

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14"])

    assert list(result.daily.columns) == BACKTEST_DAILY_COLUMNS


def test_default_test_dates_matches_split_mart1_training_sets_test_window():
    # Leaving test_dates unset must walk forward across exactly the same
    # dates as explicitly passing split_mart1_training_set's own test
    # window range -- checked by running both and comparing output, since a
    # day with nothing to predict or observe drops out of `daily` entirely
    # either way (see `_backtest_one_day`), so the two runs' `daily` frames
    # are only guaranteed identical, not a fixed date count.
    raw_visits = _one_chronic_customer_visiting_daily(
        drug_id=501, consumptions={"2024-01-15": 40.0, "2024-01-16": 30.0}
    )
    model = StubModel(0.5)
    expected_test = split_mart1_training_set(build_mart1_training_set(raw_visits)).test
    expected_dates = list(
        pd.date_range(
            expected_test[SNAPSHOT_DATE_COL].min(), expected_test[SNAPSHOT_DATE_COL].max()
        )
    )

    default_result = run_backtest(raw_visits, model, test_dates=None)
    explicit_result = run_backtest(raw_visits, model, test_dates=expected_dates)

    pd.testing.assert_frame_equal(default_result.daily, explicit_result.daily)


def test_empty_test_dates_returns_nan_wape_and_empty_daily():
    raw_visits = _one_chronic_customer_visiting_daily(drug_id=501, consumptions={"2024-01-15": 40.0})
    model = StubModel(0.5)

    result = run_backtest(raw_visits, model, test_dates=[])

    assert result.daily.empty
    assert list(result.daily.columns) == BACKTEST_DAILY_COLUMNS
    assert np.isnan(result.summary.wape)
    assert result.summary.total_predicted == 0.0
    assert result.summary.total_actual == 0.0
