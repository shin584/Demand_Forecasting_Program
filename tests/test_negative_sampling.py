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
    # Windows" is directly countable.
    offsets = negative_sample_offsets(90)

    early = [o for o in offsets if o <= 0.25 * 90]
    mid = [o for o in offsets if 0.4 * 90 < o <= 0.6 * 90]
    late = [o for o in offsets if o > 0.85 * 90]

    assert len(early) == 1
    assert len(mid) == 1
    assert 2 <= len(late) <= 3
    assert len(offsets) == len(early) + len(mid) + len(late)


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
    # Only the first visit anchors a cycle - it has a later visit; the
    # second visit is customer 1's last known visit, so it anchors nothing.
    customer_1 = negatives[negatives["고객ID"] == 1]
    expected_dates = set(negative_sample_dates("2024-01-01", 90))
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


def test_negative_to_positive_ratio_is_approximately_1_to_5_across_a_dataset():
    # Four independent chronic patient-cycles, each with a distinct
    # 처방조제일수 long enough to avoid rounding collisions between the
    # scheme's fractions. The scheme's job is to produce ~5 negatives per
    # cycle against the 1 positive each cycle's actual next visit supplies
    # elsewhere (Next-Day Visit label), i.e. the accepted ~1:5 ratio - not
    # the superseded "1:3" figure.
    cycles = [
        (1, "2024-01-01", 90, "2024-04-15"),
        (2, "2024-01-01", 60, "2024-03-15"),
        (3, "2024-01-01", 45, "2024-03-01"),
        (4, "2024-01-01", 120, "2024-05-15"),
    ]
    rows = make_high_frequency_filler_visits()
    visit_id = 1
    for customer_id, visit_date, prescription_days, next_expected in cycles:
        rows.append(
            make_visit_row(
                조제판매ID=visit_id,
                고객ID=customer_id,
                내방일=visit_date,
                처방조제일수=prescription_days,
                다음내방일=next_expected,
                약품ID=customer_id,
            )
        )
        visit_id += 1
        rows.append(
            make_visit_row(
                조제판매ID=visit_id, 고객ID=customer_id, 내방일=next_expected, 약품ID=customer_id
            )
        )
        visit_id += 1
    raw_visits = make_raw_visits(rows)

    negatives = sample_mart1_negatives(raw_visits)

    positives_per_cycle = 1
    total_positives = positives_per_cycle * len(cycles)
    assert len(negatives) == 5 * len(cycles)
    ratio = total_positives / len(negatives)
    assert ratio == 1 / 5
