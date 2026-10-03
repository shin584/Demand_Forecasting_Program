import pandas as pd
import pytest

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    MART1_COLUMNS,
    MART1_TRAINING_COLUMNS,
    SNAPSHOT_DATE_COL,
    build_mart1_daily_snapshots,
    build_mart1_training_set,
    build_marts,
    mart1_split_windows,
    split_mart1_training_set,
)
from pipeline.model import train_track1_model

# More filler occurrences than any test drug's, so the test drugs stay
# outside the top-2 Revisit Match exclusion.
_FILLER_OCCURRENCES = 6


def _extract():
    """Three Chronic customers who join, visit, turn severe and lapse at
    different times, plus a late Acute visit that sets the extract's end.

    - Customer 1: Chronic from 2024-02-15; a severity flag first appears on
      their 2024-03-20 visit, so their 학습_가중치 steps up from then on.
    - Customer 2: Chronic from 2024-04-15.
    - Customer 3: Chronic from 2024-01-20 and never seen again, so Lapsed
      from 2024-07-19 (180 days later).
    """
    return make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 다음내방일="2024-03-16", 약품ID=1),
            make_visit_row(
                조제판매ID=3, 고객ID=1, 내방일="2024-03-20", 다음내방일="2024-04-19", 약품ID=1,
                중증암등록대상자="Y",
            ),
            make_visit_row(조제판매ID=4, 고객ID=1, 내방일="2024-04-20", 다음내방일="2024-05-20", 약품ID=1),
            make_visit_row(
                조제판매ID=20, 고객ID=2, 가족ID=2, 내방일="2024-03-01", 다음내방일="2024-03-31", 약품ID=2
            ),
            make_visit_row(
                조제판매ID=21, 고객ID=2, 가족ID=2, 내방일="2024-04-15", 다음내방일="2024-05-15", 약품ID=2
            ),
            make_visit_row(
                조제판매ID=22, 고객ID=2, 가족ID=2, 내방일="2024-05-20", 처방조제일수=60, 약품ID=2
            ),
            make_visit_row(
                조제판매ID=30, 고객ID=3, 가족ID=3, 내방일="2023-12-20", 다음내방일="2024-01-19", 약품ID=3
            ),
            make_visit_row(조제판매ID=31, 고객ID=3, 가족ID=3, 내방일="2024-01-20", 약품ID=3),
            make_visit_row(조제판매ID=40, 고객ID=4, 가족ID=4, 내방일="2024-09-01", 약품ID=4),
        ]
    )


# The days each customer's population, label or weight changes on (see
# _extract), give or take a day.
_EVENT_DAYS = pd.to_datetime(
    ["2024-01-20", "2024-02-15", "2024-03-20", "2024-04-15", "2024-04-20", "2024-05-20", "2024-07-18"]
)


def _days_to_compare(start, end):
    """Every 7th day from `start` to `end`, plus the days around each of
    `_EVENT_DAYS` -- one `build_marts` per day is too slow to check them all."""
    one_day = pd.Timedelta(days=1)
    around_events = [day + shift for day in _EVENT_DAYS for shift in (-one_day, pd.Timedelta(0), one_day)]
    days = pd.date_range(start, end, freq="7D").union(pd.DatetimeIndex(around_events))
    return days[(days >= start) & (days <= end)]


def _assert_matches_build_marts(snapshots, raw_visits, days, **build_marts_kwargs):
    for day in days:
        expected = build_marts(raw_visits, day, **build_marts_kwargs).mart1
        actual = snapshots[snapshots[SNAPSHOT_DATE_COL] == day][MART1_COLUMNS]
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True), obj=str(day.date())
        )


def test_each_days_rows_match_that_days_build_marts_snapshot():
    raw_visits = _extract()
    start, end = pd.Timestamp("2023-12-15"), pd.Timestamp("2024-08-31")

    snapshots = build_mart1_daily_snapshots(raw_visits, start, end)

    assert list(snapshots.columns) == MART1_TRAINING_COLUMNS
    # Every population change is inside the range: customers joining,
    # visiting, turning severe and lapsing.
    assert set(snapshots["고객ID"]) == {1, 2, 3}
    assert snapshots["내일_방문"].any()
    assert set(snapshots["학습_가중치"]) == {2.0, 3.0}
    assert set(snapshots[SNAPSHOT_DATE_COL]) <= set(pd.date_range(start, end))
    _assert_matches_build_marts(snapshots, raw_visits, _days_to_compare(start, end))


def test_lapse_horizon_is_overridable():
    raw_visits = _extract()
    start, end = pd.Timestamp("2024-02-01"), pd.Timestamp("2024-07-01")

    snapshots = build_mart1_daily_snapshots(raw_visits, start, end, lapse_horizon_days=20)

    # A 20-day horizon opens gaps inside customers' visit histories too.
    _assert_matches_build_marts(
        snapshots, raw_visits, _days_to_compare(start, end), lapse_horizon_days=20
    )


def test_rows_stop_before_the_extracts_last_visit():
    # The label for 2024-09-01 would be a visit on 2024-09-02, past the data.
    raw_visits = _extract()

    snapshots = build_mart1_daily_snapshots(raw_visits, "2024-08-01", "2024-09-30")

    assert snapshots[SNAPSHOT_DATE_COL].max() == pd.Timestamp("2024-08-31")


def test_returns_empty_frame_when_no_one_is_in_the_population():
    raw_visits = _extract()

    before_anyone_is_chronic = build_mart1_daily_snapshots(raw_visits, "2023-12-01", "2024-01-10")
    no_chronic_patients = build_mart1_daily_snapshots(
        make_raw_visits(make_high_frequency_filler_visits() + [make_visit_row()]),
        "2024-01-01",
        "2024-01-10",
    )

    for snapshots in (before_anyone_is_chronic, no_chronic_patients):
        assert snapshots.empty
        assert list(snapshots.columns) == MART1_TRAINING_COLUMNS


def _training_set_with_dates(dates: list) -> pd.DataFrame:
    return pd.DataFrame({SNAPSHOT_DATE_COL: pd.to_datetime(dates)})


def test_windows_are_cut_from_the_end_by_snapshot_date():
    # Max 기준일자 is 2025-01-16: test = last 6mo (>= 2024-07-16), val = the
    # 6mo before that (>= 2024-01-16, < 2024-07-16), train = everything else.
    training_set = _training_set_with_dates(["2024-01-01", "2024-06-01", "2025-01-16"])

    windows = mart1_split_windows(training_set)

    assert windows.val_start == pd.Timestamp("2024-01-16")
    assert windows.test_start == pd.Timestamp("2024-07-16")
    assert windows.end == pd.Timestamp("2025-01-16")


def test_window_sizes_are_overridable():
    training_set = _training_set_with_dates(["2024-01-01", "2025-01-16"])

    windows = mart1_split_windows(training_set, test_months=1, val_months=2)

    assert windows.val_start == pd.Timestamp("2024-10-16")
    assert windows.test_start == pd.Timestamp("2024-12-16")


def test_split_keeps_train_sampled_and_makes_val_and_test_full_daily_snapshots():
    raw_visits = _extract()
    training_set = build_mart1_training_set(raw_visits)
    windows = mart1_split_windows(training_set, test_months=2, val_months=2)

    split = split_mart1_training_set(training_set, raw_visits, test_months=2, val_months=2)

    pd.testing.assert_frame_equal(
        split.train, training_set[training_set[SNAPSHOT_DATE_COL] < windows.val_start]
    )
    one_day = pd.Timedelta(days=1)
    expected_val = build_mart1_daily_snapshots(raw_visits, windows.val_start, windows.test_start - one_day)
    expected_test = build_mart1_daily_snapshots(raw_visits, windows.test_start, windows.end)
    pd.testing.assert_frame_equal(split.val, expected_val)
    pd.testing.assert_frame_equal(split.test, expected_test)
    # Every day of each window is there, boundary days on the newer side.
    assert split.val[SNAPSHOT_DATE_COL].min() == windows.val_start
    assert split.val[SNAPSHOT_DATE_COL].max() == windows.test_start - one_day
    assert split.test[SNAPSHOT_DATE_COL].min() == windows.test_start
    assert split.test[SNAPSHOT_DATE_COL].max() == windows.end
    assert split.val[SNAPSHOT_DATE_COL].nunique() == (windows.test_start - windows.val_start).days


def test_split_forwards_the_lapse_horizon_to_the_snapshots():
    raw_visits = _extract()
    training_set = build_mart1_training_set(raw_visits)
    windows = mart1_split_windows(training_set, test_months=2, val_months=2)

    split = split_mart1_training_set(
        training_set, raw_visits, test_months=2, val_months=2, lapse_horizon_days=60
    )

    expected_test = build_mart1_daily_snapshots(
        raw_visits, windows.test_start, windows.end, lapse_horizon_days=60
    )
    pd.testing.assert_frame_equal(split.test, expected_test)
    # Customer 1 (last seen 2024-04-20) is Lapsed throughout the test
    # window under a 60-day horizon, but not under the default one.
    assert set(split.test["고객ID"]) == {2}


def test_split_of_empty_training_set_returns_three_empty_frames():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + [make_visit_row()])
    training_set = build_mart1_training_set(raw_visits)

    split = split_mart1_training_set(training_set, raw_visits)

    for part in split:
        assert part.empty
        assert list(part.columns) == MART1_TRAINING_COLUMNS


def test_training_early_stops_against_the_snapshot_validation_rows():
    raw_visits = _extract()
    split = split_mart1_training_set(
        build_mart1_training_set(raw_visits), raw_visits, test_months=2, val_months=2
    )

    trained = train_track1_model(split.train, split.val)

    assert trained.model.classifier.best_iteration_ is not None
    assert trained.metrics["base_rate"] == pytest.approx(split.val["내일_방문"].mean())
