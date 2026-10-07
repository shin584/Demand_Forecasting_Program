import math

import pandas as pd
import pytest

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    MART1_NEGATIVE_SAMPLE_COLUMNS,
    WINDOW_WIDTH_COL,
    negative_sample_windows,
    sample_mart1_negatives,
)


def _days_after(dates, visit_date):
    return list((pd.Series(dates) - pd.Timestamp(visit_date)).dt.days)


@pytest.mark.parametrize("prescription_days", [1, 2, 3, 7, 14, 30, 45, 60, 90, 120, 300, math.nan])
@pytest.mark.parametrize("lapse_horizon_days", [180, 100])
def test_windows_tile_every_day_from_1_to_the_lapse_horizon(prescription_days, lapse_horizon_days):
    windows = negative_sample_windows(prescription_days, lapse_horizon_days)

    # Half-open [lo, hi): together they cover days 1..horizon exactly once.
    assert windows[0][0] == 1
    assert windows[-1][1] == lapse_horizon_days + 1
    assert all(lo < hi for lo, hi in windows)
    assert all(prev_hi == lo for (_, prev_hi), (lo, _) in zip(windows, windows[1:]))


def test_windows_run_between_midpoints_of_the_sample_points():
    # Sample points for a 30-day cycle: 4.5/15/27/27.9/28.8 in-cycle,
    # 37.5/45/60/90/120 post-cycle, and 179 (the day before the horizon).
    # Each window's edges are the rounded midpoints with its neighbours.
    assert negative_sample_windows(30) == [
        (1, 10),
        (10, 21),
        (21, 27),
        (27, 28),
        (28, 33),
        (33, 41),
        (41, 52),
        (52, 75),
        (75, 105),
        (105, 150),
        (150, 181),
    ]


def test_short_cycles_collapse_empty_windows():
    # A 1-day cycle's points 0.15..4 crowd into days 1-4, so most of their
    # windows round to nothing and are dropped.
    assert negative_sample_windows(1) == [(1, 2), (2, 4), (4, 92), (92, 181)]


def test_sample_points_past_the_lapse_horizon_get_no_window():
    # 4 x 60 = 240 is past the horizon; 3 x 60 = 180 is its last day and
    # still a point, so the last window splits at the 179/180 midpoint.
    windows = negative_sample_windows(60)

    assert len(windows) == 10
    assert windows[-2:] == [(150, 180), (180, 181)]


def test_a_missing_prescription_days_leaves_one_window_over_the_whole_horizon():
    assert negative_sample_windows(math.nan) == [(1, 181)]


def test_sample_mart1_negatives_for_a_synthetic_chronic_patient():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Customer 1: Chronic via a genuine Revisit Match, 처방조제일수=90
            # on the anchoring visit.
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                처방조제일수=90,
                다음내방일="2024-03-31",
                약품ID=1,
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-03-31", 약품ID=1),
        ]
    )

    negatives = sample_mart1_negatives(raw_visits)

    assert list(negatives.columns) == MART1_NEGATIVE_SAMPLE_COLUMNS
    # The first visit's samples stop two days before its next visit. The
    # second visit is also the extract's last 내방일, so the samples it
    # anchors would all fall past the extract's end.
    customer_1 = negatives[negatives["고객ID"] == 1]
    assert not customer_1.empty
    assert customer_1["기준일자"].max() < pd.Timestamp("2024-03-30")
    assert (customer_1["내일_방문"] == False).all()  # noqa: E712
    assert (customer_1["조제판매ID"] == 1).all()


def test_sample_mart1_negatives_excludes_acute_patients():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Only ever one visit - no Revisit Match possible, so Acute.
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 처방조제일수=30, 약품ID=1),
        ]
    )

    negatives = sample_mart1_negatives(raw_visits)

    assert negatives.empty
    assert list(negatives.columns) == MART1_NEGATIVE_SAMPLE_COLUMNS


def _two_visit_chronic_customer(prescription_days, next_visit_date):
    # Customer 1: Chronic via a genuine Revisit Match (same drug, next visit
    # inside the first visit's 다음내방일 +/-30 day window), anchoring one
    # cycle from 2024-01-01 to `next_visit_date`.
    return make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                처방조제일수=prescription_days,
                다음내방일=next_visit_date,
                약품ID=1,
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일=next_visit_date, 약품ID=1),
        ]
    )


def _chronic_customer_whose_last_visit_is_followed_by(extract_end):
    # Customer 1 last visits on 2024-03-31; an unrelated customer's visit on
    # `extract_end` sets the extract's last 내방일.
    return make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 처방조제일수=90,
                다음내방일="2024-03-31", 약품ID=1,
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-03-31", 처방조제일수=30, 약품ID=1),
            make_visit_row(조제판매ID=3, 고객ID=2, 내방일=extract_end, 약품ID=2),
        ]
    )


def test_an_unbounded_anchor_draws_one_day_inside_each_of_its_windows():
    # Nothing but the Lapse Horizon bounds a last visit with the extract
    # ending long after it, so every window keeps its one draw.
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2025-12-31")

    negatives = sample_mart1_negatives(raw_visits)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2].sort_values("기준일자")
    offsets = _days_after(anchored_on_last["기준일자"], "2024-03-31")
    windows = negative_sample_windows(30)
    assert len(offsets) == len(windows)
    for offset, width, (lo, hi) in zip(offsets, anchored_on_last[WINDOW_WIDTH_COL], windows):
        assert lo <= offset < hi
        assert width == hi - lo
    assert (anchored_on_last["내일_방문"] == False).all()  # noqa: E712


def test_draws_are_deterministic_under_a_fixed_seed_and_change_with_it():
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2025-12-31")

    first = sample_mart1_negatives(raw_visits, seed=1)
    again = sample_mart1_negatives(raw_visits, seed=1)
    other = sample_mart1_negatives(raw_visits, seed=2)

    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(sample_mart1_negatives(raw_visits), sample_mart1_negatives(raw_visits))
    assert list(first["기준일자"]) != list(other["기준일자"])


@pytest.mark.parametrize("seed", range(30))
def test_no_negative_is_sampled_on_or_the_day_before_the_next_visit(seed):
    # 처방조제일수=100: the next visit on day 94 cuts through the window
    # around the day-93 point, so day 93 -- whose customer visits on
    # 기준일자 + 1 -- is drawable but never kept.
    raw_visits = _two_visit_chronic_customer(100, "2024-04-04")

    negatives = sample_mart1_negatives(raw_visits, seed=seed)

    assert max(_days_after(negatives["기준일자"], "2024-01-01")) <= 92


def test_a_window_cut_by_the_next_visit_is_drawn_whole_then_dropped_not_shrunk():
    # 처방조제일수=100 has a [92, 94) window around its day-93 point; the
    # next visit on day 94 leaves only day 92 a true negative. The draw
    # comes from the whole window, so day 92 is kept on about half the
    # seeds, not on every one.
    assert (92, 94) in negative_sample_windows(100)
    raw_visits = _two_visit_chronic_customer(100, "2024-04-04")

    kept = [
        92 in _days_after(sample_mart1_negatives(raw_visits, seed=seed)["기준일자"], "2024-01-01")
        for seed in range(60)
    ]

    assert 0 < sum(kept) < len(kept)


@pytest.mark.parametrize("seed", range(10))
def test_same_day_visits_are_bounded_by_the_next_later_visit_date(seed):
    # Two visits on 2024-01-01 (the first's next row is same-day, not a
    # later visit): both are bounded by the 2024-02-20 visit, and the
    # same-day visits on the customer's last date anchor nothing.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 처방조제일수=90,
                다음내방일="2024-02-20", 약품ID=1,
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-01", 처방조제일수=90, 약품ID=3),
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-02-20", 처방조제일수=90, 약품ID=1),
            make_visit_row(조제판매ID=4, 고객ID=1, 내방일="2024-02-20", 처방조제일수=90, 약품ID=3),
        ]
    )

    negatives = sample_mart1_negatives(raw_visits, seed=seed)

    assert set(negatives["조제판매ID"]) == {1, 2}
    assert negatives["기준일자"].max() < pd.Timestamp("2024-02-19")


@pytest.mark.parametrize("seed", range(30))
def test_samples_anchored_on_a_last_visit_stop_before_the_extracts_end(seed):
    # The extract ends 61 days after the last visit: day 60's Next-Day label
    # (no visit on day 61) is still inside the data, day 61's isn't.
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2024-05-31")

    negatives = sample_mart1_negatives(raw_visits, seed=seed)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2]
    assert max(_days_after(anchored_on_last["기준일자"], "2024-03-31")) <= 60


@pytest.mark.parametrize("seed", range(30))
def test_lapse_horizon_bounds_the_sampled_days(seed):
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2025-12-31")

    negatives = sample_mart1_negatives(raw_visits, lapse_horizon_days=100, seed=seed)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2]
    offsets = _days_after(anchored_on_last["기준일자"], "2024-03-31")
    assert len(offsets) == len(negative_sample_windows(30, lapse_horizon_days=100))
    assert 1 <= min(offsets) and max(offsets) <= 100
