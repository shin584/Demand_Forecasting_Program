import pandas as pd

from conftest import (
    closed_on,
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.marts import (
    CLOSED_TODAY_COL,
    CLOSED_TOMORROW_COL,
    MART1_BOOLEAN_FEATURE_COLS,
    SNAPSHOT_DATE_COL,
    build_mart1_daily_snapshots,
    build_mart1_training_set,
    build_marts,
    split_mart1_training_set,
)
from pipeline.pharmacy_calendar import PharmacyCalendar


def _one_chronic_customer() -> pd.DataFrame:
    return make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=101, visit_id_start=10)
        + [make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=102)]
    )


def test_closed_day_flags_are_x_features():
    assert CLOSED_TOMORROW_COL in MART1_BOOLEAN_FEATURE_COLS
    assert CLOSED_TODAY_COL in MART1_BOOLEAN_FEATURE_COLS


def test_mart1_flags_a_closed_target_day():
    mart1 = build_marts(
        _one_chronic_customer(), "2024-01-06", pharmacy_calendar=closed_on("2024-01-07")
    ).mart1

    assert mart1[CLOSED_TOMORROW_COL].tolist() == [True]
    assert mart1[CLOSED_TODAY_COL].tolist() == [False]


def test_mart1_flags_a_closed_snapshot_day():
    mart1 = build_marts(
        _one_chronic_customer(), "2024-01-07", pharmacy_calendar=closed_on("2024-01-07")
    ).mart1

    assert mart1[CLOSED_TOMORROW_COL].tolist() == [False]
    assert mart1[CLOSED_TODAY_COL].tolist() == [True]


def test_without_a_calendar_no_day_is_closed():
    mart1 = build_marts(_one_chronic_customer(), "2024-01-06").mart1

    assert mart1[CLOSED_TOMORROW_COL].tolist() == [False]
    assert mart1[CLOSED_TODAY_COL].tolist() == [False]


def test_closed_day_flags_are_lightgbm_ready_booleans():
    mart1 = build_marts(
        _one_chronic_customer(), "2024-01-06", pharmacy_calendar=closed_on("2024-01-07")
    ).mart1

    assert mart1[CLOSED_TOMORROW_COL].dtype == "boolean"
    assert mart1[CLOSED_TODAY_COL].dtype == "boolean"


def _chronic_history() -> pd.DataFrame:
    return make_raw_visits(
        make_high_frequency_filler_visits(6)
        + [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 다음내방일="2024-03-02", 약품ID=1),
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-03-05", 다음내방일="2024-04-04", 약품ID=1),
            make_visit_row(조제판매ID=4, 고객ID=1, 내방일="2024-04-03", 약품ID=1),
        ]
    )


# Every Sunday of the history above.
_SUNDAYS = pd.date_range("2024-01-07", "2024-04-30", freq="W-SUN")
_SUNDAY_CALENDAR = PharmacyCalendar(closed_days=_SUNDAYS, observed_through=None)


def _assert_flags_follow_each_rows_own_snapshot_date(rows: pd.DataFrame) -> None:
    snapshot_dates = rows[SNAPSHOT_DATE_COL]
    expected_tomorrow = (snapshot_dates + pd.Timedelta(days=1)).isin(_SUNDAYS)
    assert rows[CLOSED_TOMORROW_COL].tolist() == expected_tomorrow.tolist()
    assert rows[CLOSED_TODAY_COL].tolist() == snapshot_dates.isin(_SUNDAYS).tolist()


def test_training_rows_flag_closures_as_of_their_own_snapshot_date():
    training_set = build_mart1_training_set(_chronic_history(), pharmacy_calendar=_SUNDAY_CALENDAR)

    assert len(training_set) > 3
    _assert_flags_follow_each_rows_own_snapshot_date(training_set)


def test_daily_snapshot_rows_flag_closures_as_of_their_own_day():
    snapshots = build_mart1_daily_snapshots(
        _chronic_history(), "2024-02-01", "2024-03-31", pharmacy_calendar=_SUNDAY_CALENDAR
    )

    assert snapshots[CLOSED_TOMORROW_COL].any()
    _assert_flags_follow_each_rows_own_snapshot_date(snapshots)


def test_daily_snapshots_match_build_marts_with_a_calendar():
    raw_visits = _chronic_history()
    snapshots = build_mart1_daily_snapshots(
        raw_visits, "2024-02-01", "2024-03-31", pharmacy_calendar=_SUNDAY_CALENDAR
    )

    for day in pd.to_datetime(["2024-02-10", "2024-02-11", "2024-03-02"]):
        expected = build_marts(raw_visits, day, pharmacy_calendar=_SUNDAY_CALENDAR).mart1
        actual = snapshots[snapshots[SNAPSHOT_DATE_COL] == day][expected.columns]
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected)


def test_split_forwards_the_calendar_to_the_validation_and_test_snapshots():
    raw_visits = _chronic_history()
    training_set = build_mart1_training_set(raw_visits, pharmacy_calendar=_SUNDAY_CALENDAR)

    split = split_mart1_training_set(
        training_set, raw_visits, test_months=1, val_months=1, pharmacy_calendar=_SUNDAY_CALENDAR
    )

    eval_rows = pd.concat([split.val, split.test], ignore_index=True)
    assert eval_rows[CLOSED_TOMORROW_COL].any()
    _assert_flags_follow_each_rows_own_snapshot_date(eval_rows)
