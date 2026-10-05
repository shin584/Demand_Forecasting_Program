import numpy as np
import pandas as pd
import pytest

from conftest import (
    closed_on,
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.backtest import (
    ACTUAL_CHRONIC_VISITS_COL,
    ACTUAL_COL,
    BACKTEST_DAILY_COLUMNS,
    BACKTEST_PER_DATE_COLUMNS,
    ERROR_COL,
    PREDICTED_COL,
    SCORED_POPULATION_SIZE_COL,
    SUM_VISIT_PROB_COL,
    TRACK1_ACTUAL_COL,
    TRACK1_PREDICTED_COL,
    TRACK2_ACTUAL_COL,
    TRACK2_PREDICTED_COL,
    VISIT_LIST_SIZE_COL,
    run_backtest,
)
from pipeline.marts import (
    CLOSED_TOMORROW_COL,
    DRUG_ID_COL,
    SNAPSHOT_DATE_COL,
    build_mart1_training_set,
    mart1_split_windows,
)
from pipeline.model import CalibratedTrack1Model, PlattCalibrator


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
            # Dispensed on the Anchoring Visit (visit 11) alongside drug 901,
            # so 501 is in customer 1's Current Regimen (issue #30).
            make_visit_row(
                조제판매ID=11, 고객ID=1, 내방일="2023-11-15", 약품ID=501, 소모량=40.0
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


def test_default_test_dates_matches_the_temporal_splits_test_window():
    # Leaving test_dates unset must walk forward across exactly the same
    # dates as explicitly passing mart1_split_windows' own test window --
    # checked by running both and comparing output, since a day with nothing
    # to predict or observe drops out of `daily` entirely either way (see
    # `_backtest_one_day`), so the two runs' `daily` frames are only
    # guaranteed identical, not a fixed date count.
    raw_visits = _one_chronic_customer_visiting_daily(
        drug_id=501, consumptions={"2024-01-15": 40.0, "2024-01-16": 30.0}
    )
    model = StubModel(0.5)
    windows = mart1_split_windows(build_mart1_training_set(raw_visits))
    expected_dates = list(pd.date_range(windows.test_start, windows.end))

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


def _one_chronic_customer_on_drug_501() -> list[dict]:
    """Customer 1: Chronic since an old Revisit Match (drug 901, 소모량 0.0
    so it never adds predicted demand), whose most recent visit (2024-01-01)
    dispenses rare drug 501 at 40.0 -- so Track 1 predicts 40.0 of drug 501
    every day the model clears the cutoff (100%-allocation rule, see
    docs/adr/0004), and Track 2 predicts nothing (the filler's noise drugs
    are too rare for Mart 3)."""
    return make_high_frequency_filler_visits() + [
        make_visit_row(
            조제판매ID=10,
            고객ID=1,
            내방일="2023-11-01",
            다음내방일="2023-12-01",
            약품ID=901,
            소모량=0.0,
        ),
        make_visit_row(조제판매ID=11, 고객ID=1, 내방일="2023-11-15", 약품ID=901, 소모량=0.0),
        make_visit_row(조제판매ID=12, 고객ID=1, 내방일="2024-01-01", 약품ID=501, 소모량=40.0),
    ]


def _per_track_fixture() -> pd.DataFrame:
    # Target date 2024-01-15 (as-of 2024-01-14):
    # - customer 1 (scored by Track 1) consumes 30.0 of drug 501
    # - customer 3 (one-off, Track 2's population) consumes 10.0 of drug 501
    # - customer 2 (one-off, Track 2's population) consumes 25.0 of drug 777
    return make_raw_visits(
        _one_chronic_customer_on_drug_501()
        + [
            make_visit_row(조제판매ID=100, 고객ID=1, 내방일="2024-01-15", 약품ID=501, 소모량=30.0),
            make_visit_row(조제판매ID=101, 고객ID=3, 내방일="2024-01-15", 약품ID=501, 소모량=10.0),
            make_visit_row(조제판매ID=102, 고객ID=2, 내방일="2024-01-15", 약품ID=777, 소모량=25.0),
        ]
    )


def test_daily_splits_predicted_and_actual_per_track():
    result = run_backtest(_per_track_fixture(), StubModel(0.5), test_dates=["2024-01-14"])

    daily = result.daily.set_index(DRUG_ID_COL)
    assert daily.loc[501, TRACK1_PREDICTED_COL] == pytest.approx(40.0)
    assert daily.loc[501, TRACK2_PREDICTED_COL] == 0.0
    assert daily.loc[501, TRACK1_ACTUAL_COL] == pytest.approx(30.0)
    assert daily.loc[501, TRACK2_ACTUAL_COL] == pytest.approx(10.0)
    assert daily.loc[777, TRACK1_ACTUAL_COL] == 0.0
    assert daily.loc[777, TRACK2_ACTUAL_COL] == pytest.approx(25.0)


def test_per_track_columns_sum_to_the_combined_columns():
    result = run_backtest(_per_track_fixture(), StubModel(0.5), test_dates=["2024-01-14"])

    daily = result.daily
    pd.testing.assert_series_equal(
        daily[TRACK1_PREDICTED_COL] + daily[TRACK2_PREDICTED_COL],
        daily[PREDICTED_COL],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        daily[TRACK1_ACTUAL_COL] + daily[TRACK2_ACTUAL_COL],
        daily[ACTUAL_COL],
        check_names=False,
    )


def test_per_track_wape_matches_hand_computed_values():
    result = run_backtest(_per_track_fixture(), StubModel(0.5), test_dates=["2024-01-14"])

    # Track 1: |40 - 30| / 30.  Track 2: (|0 - 10| + |0 - 25|) / 35.
    assert result.summary.track1_wape == pytest.approx(10.0 / 30.0)
    assert result.summary.track2_wape == pytest.approx(1.0)
    # Combined (unchanged formula): drug 501 |40 - 40| + drug 777 |0 - 25|, over 65.
    assert result.summary.wape == pytest.approx(25.0 / 65.0)


def test_per_date_reports_visit_probability_sum_against_actual_chronic_visits():
    # Two scored Chronic customers (1 and 4); only customer 1 visits on the
    # target date, on two drug rows -- one visit, not two. Customer 2's
    # one-off visit isn't a Chronic visit at all.
    raw_visits = make_raw_visits(
        _one_chronic_customer_on_drug_501()
        + make_independent_chronic_match_visits(customer_id=4, drug_id=902, visit_id_start=20)
        + [
            make_visit_row(조제판매ID=100, 고객ID=1, 내방일="2024-01-15", 약품ID=501, 소모량=30.0),
            make_visit_row(조제판매ID=100, 고객ID=1, 내방일="2024-01-15", 약품ID=502, 소모량=5.0),
            make_visit_row(조제판매ID=101, 고객ID=2, 내방일="2024-01-15", 약품ID=777, 소모량=25.0),
        ]
    )

    result = run_backtest(raw_visits, StubModel(0.5), test_dates=["2024-01-14"])

    assert list(result.per_date.columns) == BACKTEST_PER_DATE_COLUMNS
    row = result.per_date.set_index(SNAPSHOT_DATE_COL).loc[pd.Timestamp("2024-01-14")]
    assert row[SUM_VISIT_PROB_COL] == pytest.approx(1.0)
    assert row[ACTUAL_CHRONIC_VISITS_COL] == 1
    assert row[VISIT_LIST_SIZE_COL] == 2
    assert row[SCORED_POPULATION_SIZE_COL] == 2


def test_per_date_visit_list_size_respects_the_cutoff():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())

    result = run_backtest(
        raw_visits, StubModel(0.2), test_dates=["2024-01-14"], chronic_visit_prob_cutoff=0.3
    )

    row = result.per_date.iloc[0]
    assert row[VISIT_LIST_SIZE_COL] == 0
    assert row[SCORED_POPULATION_SIZE_COL] == 1
    assert row[SUM_VISIT_PROB_COL] == pytest.approx(0.2)
    assert row[ACTUAL_CHRONIC_VISITS_COL] == 0


def test_per_date_visit_list_size_uses_the_given_cutoff():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())

    result = run_backtest(
        raw_visits, StubModel(0.2), test_dates=["2024-01-14"], chronic_visit_prob_cutoff=0.1
    )

    assert result.per_date.iloc[0][VISIT_LIST_SIZE_COL] == 1


def test_scores_with_calibrated_probabilities():
    # Slope 0 maps the stub's raw 0.5 to sigmoid(intercept) = 0.05.
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())
    model = CalibratedTrack1Model(
        classifier=StubModel(0.5),
        calibrator=PlattCalibrator(slope=0.0, intercept=float(np.log(0.05 / 0.95))),
    )

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14"])

    assert result.per_date.iloc[0][SUM_VISIT_PROB_COL] == pytest.approx(0.05)


def test_per_date_visit_list_size_uses_the_models_tuned_cutoff_by_default():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())
    model = CalibratedTrack1Model(
        classifier=StubModel(0.2),
        calibrator=PlattCalibrator(slope=1.0, intercept=0.0),
        chronic_visit_prob_cutoff=0.15,
    )

    result = run_backtest(raw_visits, model, test_dates=["2024-01-14"])

    assert result.per_date.iloc[0][VISIT_LIST_SIZE_COL] == 1


def test_per_date_has_one_row_per_test_date():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())

    result = run_backtest(raw_visits, StubModel(0.5), test_dates=["2024-01-14", "2024-01-15"])

    assert list(result.per_date[SNAPSHOT_DATE_COL]) == [
        pd.Timestamp("2024-01-14"),
        pd.Timestamp("2024-01-15"),
    ]


def test_empty_test_dates_returns_empty_per_date_and_nan_track_wapes():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())

    result = run_backtest(raw_visits, StubModel(0.5), test_dates=[])

    assert result.per_date.empty
    assert list(result.per_date.columns) == BACKTEST_PER_DATE_COLUMNS
    assert np.isnan(result.summary.track1_wape)
    assert np.isnan(result.summary.track2_wape)


def test_per_date_flags_closed_target_days_and_scores_them_zero():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())
    calendar = closed_on("2024-01-14")

    result = run_backtest(
        raw_visits,
        StubModel(0.5),
        test_dates=["2024-01-13", "2024-01-14"],
        chronic_visit_prob_cutoff=0.3,
        pharmacy_calendar=calendar,
    )

    per_date = result.per_date.set_index(SNAPSHOT_DATE_COL)
    assert per_date[CLOSED_TOMORROW_COL].tolist() == [True, False]
    closed = per_date.loc[pd.Timestamp("2024-01-13")]
    assert closed[SUM_VISIT_PROB_COL] == 0.0
    assert closed[VISIT_LIST_SIZE_COL] == 0
    assert per_date.loc[pd.Timestamp("2024-01-14"), SUM_VISIT_PROB_COL] == pytest.approx(0.5)
    track1_predicted = result.daily.groupby(SNAPSHOT_DATE_COL)[TRACK1_PREDICTED_COL].sum()
    assert track1_predicted.loc[pd.Timestamp("2024-01-13")] == 0.0


def test_per_date_marks_no_day_closed_without_a_calendar():
    raw_visits = make_raw_visits(_one_chronic_customer_on_drug_501())

    result = run_backtest(raw_visits, StubModel(0.5), test_dates=["2024-01-13", "2024-01-14"])

    assert not result.per_date[CLOSED_TOMORROW_COL].any()
