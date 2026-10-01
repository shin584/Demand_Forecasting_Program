import pandas as pd

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import build_marts, chronic_since_dates


def test_chronic_since_date_is_the_later_visit_of_the_first_matched_pair():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(occurrences=5)
        + [
            # Pair 1: visit 1 is matched by visit 2 (2024-02-15).
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 다음내방일="2024-03-15", 약품ID=1
            ),
            # Pair 2: visit 2 is matched by visit 3 -- observed later, so it
            # doesn't move the Chronic-since Date.
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-03-20", 약품ID=1),
        ]
    )

    since = chronic_since_dates(raw_visits)

    assert since.loc[1] == pd.Timestamp("2024-02-15")


def test_chronic_since_date_uses_the_first_in_window_revisit_not_the_first_revisit():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(occurrences=5)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-03-01", 약품ID=1
            ),
            # Shares the drug but lands before 다음내방일 - 30 days: not a match.
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-10", 약품ID=1),
            # First later visit inside the +/-30-day window.
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-02-10", 약품ID=1),
        ]
    )

    since = chronic_since_dates(raw_visits)

    assert since.loc[1] == pd.Timestamp("2024-02-10")


def test_chronic_since_date_takes_the_earliest_across_pairs_and_drugs():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(occurrences=5)
        + [
            # Drug 1's match is observed 2024-03-10 ...
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-02-01", 다음내방일="2024-03-01", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-03-10", 약품ID=1),
            # ... but an earlier pair on drug 2 is observed 2024-02-01.
            make_visit_row(
                조제판매ID=3, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=2
            ),
            make_visit_row(조제판매ID=4, 고객ID=1, 내방일="2024-02-01", 약품ID=2),
        ]
    )

    since = chronic_since_dates(raw_visits)

    assert since.loc[1] == pd.Timestamp("2024-02-01")


def test_customers_who_never_revisit_match_have_no_chronic_since_date():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(occurrences=5)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
            # Customer 2's revisit falls outside the window.
            make_visit_row(
                조제판매ID=3, 고객ID=2, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=2
            ),
            make_visit_row(조제판매ID=4, 고객ID=2, 내방일="2024-04-01", 약품ID=2),
        ]
    )

    since = chronic_since_dates(raw_visits)

    assert set(since.index) == {1}
    assert since.index.name == "고객ID"


def test_chronic_since_dates_is_empty_for_empty_input():
    raw_visits = make_raw_visits([])

    since = chronic_since_dates(raw_visits)

    assert since.empty


def _single_pair_visits():
    return make_raw_visits(
        make_high_frequency_filler_visits(occurrences=5)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )


def test_build_marts_not_chronic_the_day_before_the_chronic_since_date():
    raw_visits = _single_pair_visits()

    result = build_marts(raw_visits, as_of_date="2024-02-14", rare_drug_patient_threshold=1)

    assert 1 not in set(result.mart1["고객ID"])
    assert 1 in set(result.mart3["약품ID"])


def test_build_marts_chronic_on_and_after_the_chronic_since_date():
    raw_visits = _single_pair_visits()

    for as_of_date in ["2024-02-15", "2024-02-16", "2024-12-31"]:
        result = build_marts(raw_visits, as_of_date=as_of_date, rare_drug_patient_threshold=1)

        assert 1 in set(result.mart1["고객ID"]), as_of_date
        assert result.mart1.set_index("고객ID")["만성질환여부"].loc[1] == True, as_of_date  # noqa: E712
        assert 1 not in set(result.mart3["약품ID"]), as_of_date
