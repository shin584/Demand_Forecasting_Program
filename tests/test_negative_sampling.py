import pandas as pd

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    MART1_NEGATIVE_SAMPLE_COLUMNS,
    negative_sample_dates,
    negative_sample_offsets,
    sample_mart1_negatives,
)


def test_offsets_fall_into_early_mid_and_late_bands_of_the_cycle():
    # A 90-day cycle keeps every fraction's rounded offset distinct, so the
    # early/1-mid/late(2-3) shape from CONTEXT.md's "Negative Sampling
    # Windows" is directly countable. Only the in-cycle offsets (< 90) are
    # checked here; post-cycle ones are covered below.
    offsets = [o for o in negative_sample_offsets(90) if o < 90]

    early = [o for o in offsets if o <= 0.25 * 90]
    mid = [o for o in offsets if 0.4 * 90 < o <= 0.6 * 90]
    late = [o for o in offsets if o > 0.85 * 90]

    assert len(early) == 1
    assert len(mid) == 1
    assert 2 <= len(late) <= 3
    assert len(offsets) == len(early) + len(mid) + len(late)


def test_offsets_continue_past_the_cycle_up_to_just_before_the_lapse_horizon():
    # In-cycle 15/50/90/93/96%, post-cycle 1.25/1.5/2/3/4x, and the day
    # before the 180-day Lapse Horizon.
    assert negative_sample_offsets(30) == [4, 15, 27, 28, 29, 38, 45, 60, 90, 120, 179]


def test_offsets_beyond_the_lapse_horizon_are_dropped():
    # 4 x 60 = 240 days is past the horizon; 3 x 60 = 180 is the horizon's
    # last day and still within it.
    assert negative_sample_offsets(60) == [9, 30, 54, 56, 58, 75, 90, 120, 179, 180]
    assert max(negative_sample_offsets(30, lapse_horizon_days=100)) == 99


def test_duplicate_offsets_are_collapsed():
    assert negative_sample_offsets(1) == [1, 2, 3, 4, 179]


def test_offsets_scale_with_the_patient_own_prescription_days_not_fixed_absolute_days():
    # Two synthetic patients with very different 처방조제일수 cycle lengths
    # (7 vs 60 days, per the acceptance criteria) must get offsets that scale
    # with their own cycle, not the same absolute day numbers.
    short_cycle = negative_sample_offsets(7)
    long_cycle = negative_sample_offsets(60)

    assert short_cycle != long_cycle
    assert max(short_cycle) < max(long_cycle)
    # Each cycle's own early sample is still ~15% of that cycle's length.
    assert short_cycle[0] == round(7 * 0.15)
    assert long_cycle[0] == round(60 * 0.15)


def test_no_offset_is_backward_looking():
    for prescription_days in (1, 2, 7, 30, 60, 120):
        offsets = negative_sample_offsets(prescription_days)
        assert all(offset >= 1 for offset in offsets)


def test_negative_sample_dates_are_strictly_after_the_anchoring_visit():
    dates = negative_sample_dates("2024-01-01", 30)

    assert all(date > pd.Timestamp("2024-01-01") for date in dates)


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
    expected_dates = {
        date
        for date in negative_sample_dates("2024-01-01", 90)
        if date < pd.Timestamp("2024-03-30")
    }
    assert set(customer_1["기준일자"]) == expected_dates
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


def test_in_cycle_negatives_on_or_after_the_next_visit_are_dropped():
    # 처방조제일수=90 puts samples at days 14/45/81/84/86, but the customer
    # comes back early on day 50: the late samples describe a cycle that has
    # already ended.
    raw_visits = _two_visit_chronic_customer(90, "2024-02-20")

    negatives = sample_mart1_negatives(raw_visits)

    assert set(negatives["기준일자"]) == {
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-02-15"),
    }


def test_no_negative_is_sampled_the_day_before_the_next_visit():
    # 처방조제일수=100 puts samples at days 15/50/90/93/96. The next visit is
    # on day 94, so the day-93 sample's customer actually visits on
    # 기준일자 + 1 -- its true Next-Day label is Y=1, not a negative.
    raw_visits = _two_visit_chronic_customer(100, "2024-04-04")

    negatives = sample_mart1_negatives(raw_visits)

    assert pd.Timestamp("2024-04-03") not in set(negatives["기준일자"])
    assert set(negatives["기준일자"]) == {
        pd.Timestamp("2024-01-16"),
        pd.Timestamp("2024-02-20"),
        pd.Timestamp("2024-03-31"),
    }


def test_same_day_visits_are_bounded_by_the_next_later_visit_date():
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

    negatives = sample_mart1_negatives(raw_visits)

    assert set(negatives["조제판매ID"]) == {1, 2}
    assert negatives["기준일자"].max() < pd.Timestamp("2024-02-19")


def test_post_cycle_negatives_stop_two_days_before_the_next_visit():
    # 처방조제일수=30 samples days 4/15/27/28/29 in-cycle and 38/45/60/90/120
    # and 179 after it. The customer comes back on day 61, so the day-60
    # sample's customer visits on 기준일자 + 1 -- not a negative.
    raw_visits = _two_visit_chronic_customer(30, "2024-03-02")

    negatives = sample_mart1_negatives(raw_visits)

    anchored_on_first = negatives[negatives["조제판매ID"] == 1]
    offsets = (anchored_on_first["기준일자"] - pd.Timestamp("2024-01-01")).dt.days
    assert list(offsets) == [4, 15, 27, 28, 29, 38, 45]


def _chronic_customer_whose_last_visit_is_followed_by(extract_end):
    # Customer 1 last visits on 2024-03-31 (처방조제일수=30); an unrelated
    # customer's visit on `extract_end` sets the extract's last 내방일.
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


def test_a_customers_last_visit_anchors_samples_up_to_the_lapse_horizon():
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2024-12-31")

    negatives = sample_mart1_negatives(raw_visits)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2]
    assert set(anchored_on_last["기준일자"]) == set(negative_sample_dates("2024-03-31", 30))
    assert (anchored_on_last["내일_방문"] == False).all()  # noqa: E712


def test_samples_anchored_on_a_last_visit_stop_before_the_extracts_end():
    # The extract ends 61 days after the last visit: the day-60 sample's
    # Next-Day label (no visit on day 61) is still inside the data, the
    # day-90 one's isn't.
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2024-05-31")

    negatives = sample_mart1_negatives(raw_visits)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2]
    offsets = (anchored_on_last["기준일자"] - pd.Timestamp("2024-03-31")).dt.days
    assert list(offsets) == [4, 15, 27, 28, 29, 38, 45, 60]


def test_lapse_horizon_bounds_the_sampled_offsets():
    raw_visits = _chronic_customer_whose_last_visit_is_followed_by("2024-12-31")

    negatives = sample_mart1_negatives(raw_visits, lapse_horizon_days=100)

    anchored_on_last = negatives[negatives["조제판매ID"] == 2]
    offsets = (anchored_on_last["기준일자"] - pd.Timestamp("2024-03-31")).dt.days
    assert list(offsets) == [4, 15, 27, 28, 29, 38, 45, 60, 90, 99]
