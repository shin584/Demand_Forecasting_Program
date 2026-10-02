import pandas as pd
import pytest

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    LAPSE_HORIZON_DAYS,
    SNAPSHOT_DATE_COL,
    build_mart1_training_set,
    build_marts,
)

# Customer 1's last visit: the later visit of their only Revisit-Matched
# pair, so they're Chronic from this date on.
_LAST_VISIT = pd.Timestamp("2024-02-15")

# More filler occurrences than any test drug's, so drug 1 stays outside the
# top-2 Revisit Match exclusion.
_FILLER_OCCURRENCES = 6


def _chronic_customer(소모량=60.0, last_visit_prescription_days=30):
    return [
        make_visit_row(
            조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1, 소모량=소모량
        ),
        make_visit_row(
            조제판매ID=2,
            고객ID=1,
            내방일=str(_LAST_VISIT.date()),
            처방조제일수=last_visit_prescription_days,
            약품ID=1,
            소모량=소모량,
        ),
    ]


def test_default_lapse_horizon_is_180_days():
    assert LAPSE_HORIZON_DAYS == 180


@pytest.mark.parametrize(
    ("days_since_last_visit", "in_mart1"),
    [(LAPSE_HORIZON_DAYS - 1, True), (LAPSE_HORIZON_DAYS, True), (LAPSE_HORIZON_DAYS + 1, False)],
)
def test_mart1_excludes_chronic_customers_past_the_lapse_horizon(days_since_last_visit, in_mart1):
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer())
    as_of_date = _LAST_VISIT + pd.Timedelta(days=days_since_last_visit)

    mart1 = build_marts(raw_visits, as_of_date=as_of_date).mart1

    assert (1 in set(mart1["고객ID"])) == in_mart1


def test_lapse_horizon_is_overridable():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer())
    as_of_date = _LAST_VISIT + pd.Timedelta(days=31)

    kept = build_marts(raw_visits, as_of_date=as_of_date, lapse_horizon_days=31).mart1
    lapsed = build_marts(raw_visits, as_of_date=as_of_date, lapse_horizon_days=30).mart1

    assert 1 in set(kept["고객ID"])
    assert 1 not in set(lapsed["고객ID"])


def test_lapsed_chronic_customer_stays_out_of_mart3():
    # Customer 1 is Lapsed as of the snapshot but still Chronic, so their
    # drug-1 consumption must not route to Mart 3; acute customer 2's does.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer(소모량=999.0)
        + [make_visit_row(조제판매ID=3, 고객ID=2, 가족ID=2, 내방일="2024-01-15", 약품ID=1, 소모량=50.0)]
    )
    as_of_date = _LAST_VISIT + pd.Timedelta(days=LAPSE_HORIZON_DAYS + 1)

    result = build_marts(
        raw_visits,
        as_of_date=as_of_date,
        mart3_bucket_min_observations=1,
        rare_drug_patient_threshold=1,
    )

    assert 1 not in set(result.mart1["고객ID"])
    drug1_values = result.mart3.loc[result.mart3["약품ID"] == 1, "소모량"]
    assert not drug1_values.empty
    assert (drug1_values == 50.0).all()


@pytest.mark.parametrize(
    ("gap_days", "kept"),
    [(LAPSE_HORIZON_DAYS + 1, True), (LAPSE_HORIZON_DAYS + 2, False)],
)
def test_training_set_drops_a_positive_whose_customer_is_lapsed_as_of_it(gap_days, kept):
    # The return visit's positive is dated the day before it, so a return
    # gap_days after the last visit puts that row gap_days - 1 days out.
    return_visit = _LAST_VISIT + pd.Timedelta(days=gap_days)
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer()
        + [make_visit_row(조제판매ID=3, 고객ID=1, 내방일=str(return_visit.date()), 약품ID=1)]
    )

    training_set = build_mart1_training_set(raw_visits)

    positives = training_set[training_set["내일_방문"] == True]  # noqa: E712
    return_row = return_visit - pd.Timedelta(days=1)
    assert (return_row in set(positives[SNAPSHOT_DATE_COL])) == kept


def test_training_set_drops_negatives_past_the_lapse_horizon():
    # A 300-day cycle anchored on the visit that makes customer 1 Chronic:
    # its early/mid negatives (+45, +150 days) and the day-before-horizon
    # one (+179) are within the horizon, its late and post-cycle ones
    # (+270 and later) are not. The first visit's own negatives all predate
    # the Chronic-since Date.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer(last_visit_prescription_days=300)
        + [make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2025-01-01", 약품ID=1)]
    )

    training_set = build_mart1_training_set(raw_visits)

    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    assert sorted(negatives[SNAPSHOT_DATE_COL]) == [
        _LAST_VISIT + pd.Timedelta(days=45),
        _LAST_VISIT + pd.Timedelta(days=150),
        _LAST_VISIT + pd.Timedelta(days=179),
    ]
