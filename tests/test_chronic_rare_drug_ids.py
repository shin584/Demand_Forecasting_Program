import pandas as pd

from conftest import make_raw_visits, make_visit_row
from pipeline.marts import chronic_rare_drug_ids

AS_OF_DATE = pd.Timestamp("2024-02-01")


def test_excludes_a_drug_below_the_chronic_population_threshold():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1),
        ]
    )

    rare_ids = chronic_rare_drug_ids(
        raw_visits, AS_OF_DATE, chronic_customer_ids={1, 2}, rare_drug_patient_threshold=3
    )

    # Only 2 distinct Chronic patients -- strictly below the threshold of 3.
    assert 1 in rare_ids


def test_includes_a_drug_at_exactly_the_chronic_population_threshold():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1),
            make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2024-01-15", 약품ID=1),
        ]
    )

    rare_ids = chronic_rare_drug_ids(
        raw_visits, AS_OF_DATE, chronic_customer_ids={1, 2, 3}, rare_drug_patient_threshold=3
    )

    # Exactly 3 distinct patients -- CONTEXT.md's rare-drug cutoff is
    # strictly-less-than, so a drug at the threshold is still not rare.
    assert 1 not in rare_ids


def test_counts_only_the_chronic_population_not_acute():
    # Customer 3 is Acute (excluded from chronic_customer_ids) -- must not
    # count toward drug 1's Chronic-population patient total, even though
    # Mart 3's Acute-scoped `_rare_drug_ids` would count them for its own,
    # independent Acute-population question.
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1),
            make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2024-01-15", 약품ID=1),
        ]
    )

    rare_ids = chronic_rare_drug_ids(
        raw_visits, AS_OF_DATE, chronic_customer_ids={1, 2}, rare_drug_patient_threshold=3
    )

    assert 1 in rare_ids


def test_excludes_patients_outside_the_trailing_12_months():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2022-01-05", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2022-01-10", 약품ID=1),
            make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2022-01-15", 약품ID=1),
        ]
    )

    rare_ids = chronic_rare_drug_ids(
        raw_visits, AS_OF_DATE, chronic_customer_ids={1, 2, 3}, rare_drug_patient_threshold=3
    )

    # All three visits are well over 12 months before as_of_date, so none of
    # them count toward the snapshot's total despite being at-or-before
    # as_of_date.
    assert 1 in rare_ids


def test_respects_as_of_date():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1),
            make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1),
            # This third patient only shows up after as_of_date -- the drug
            # only crosses the threshold later, which this snapshot must not
            # already know about.
            make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2024-03-01", 약품ID=1),
        ]
    )

    rare_ids = chronic_rare_drug_ids(
        raw_visits, AS_OF_DATE, chronic_customer_ids={1, 2, 3}, rare_drug_patient_threshold=3
    )

    assert 1 in rare_ids


def test_default_threshold_matches_rare_drug_patient_threshold():
    from pipeline.marts import RARE_DRUG_PATIENT_THRESHOLD

    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=i, 고객ID=i, 내방일="2024-01-05", 약품ID=1)
            for i in range(1, RARE_DRUG_PATIENT_THRESHOLD)
        ]
    )
    chronic_customer_ids = set(range(1, RARE_DRUG_PATIENT_THRESHOLD))

    rare_ids = chronic_rare_drug_ids(raw_visits, AS_OF_DATE, chronic_customer_ids)

    # One patient short of the default threshold.
    assert 1 in rare_ids
