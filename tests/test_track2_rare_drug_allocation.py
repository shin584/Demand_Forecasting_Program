from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    TRACK2_RARE_DRUG_COLUMNS,
    DRUG_ID_COL,
    RARE_STOCK_FLOOR_COL,
    track2_rare_drug_allocation,
    build_marts,
)

# More filler occurrences than any test drug gets below, so test drugs stay
# outside the Revisit Match top-2 exclusion (see ADR-0001). The filler drugs
# themselves are Acute and rare under the thresholds used here; tests only
# look at their own drugs' rows.
_FILLER_OCCURRENCES = 6


def _profile(rows, as_of_date, threshold=3):
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + rows)
    return track2_rare_drug_allocation(raw_visits, as_of_date, rare_drug_patient_threshold=threshold)


def _two_patient_drug_1():
    return [
        make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=10.0),
        make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-05", 약품ID=1, 소모량=20.0),
    ]


def test_profile_has_the_track2_rare_drug_columns():
    profile = _profile(_two_patient_drug_1(), "2024-01-10")

    assert list(profile.columns) == TRACK2_RARE_DRUG_COLUMNS


def test_stock_floor_is_the_latest_single_acute_dispensing():
    profile = _profile(_two_patient_drug_1(), "2024-01-10").set_index(DRUG_ID_COL)

    assert profile.loc[1, RARE_STOCK_FLOOR_COL] == 20.0


def test_stock_floor_takes_the_largest_dispensing_on_the_latest_date():
    rows = _two_patient_drug_1() + [
        make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2024-01-05", 약품ID=1, 소모량=35.0)
    ]

    profile = _profile(rows, "2024-01-10", threshold=4).set_index(DRUG_ID_COL)

    assert profile.loc[1, RARE_STOCK_FLOOR_COL] == 35.0


def test_stock_floor_skips_a_latest_dispensing_with_missing_consumption():
    rows = [
        make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=10.0),
        make_visit_row(조제판매ID=2, 고객ID=2, 내방일="2024-01-05", 약품ID=1, 소모량=float("nan")),
    ]

    profile = _profile(rows, "2024-01-10").set_index(DRUG_ID_COL)

    assert profile.loc[1, RARE_STOCK_FLOOR_COL] == 10.0


def test_drug_at_the_threshold_is_not_rare():
    profile = _profile(_two_patient_drug_1(), "2024-01-10", threshold=2)

    assert 1 not in set(profile[DRUG_ID_COL])


def test_drug_with_no_acute_patient_in_the_trailing_12_months_is_left_out():
    profile = _profile(_two_patient_drug_1(), "2025-03-01")

    assert 1 not in set(profile[DRUG_ID_COL])


def test_visits_after_as_of_date_are_ignored():
    rows = _two_patient_drug_1() + [
        make_visit_row(조제판매ID=3, 고객ID=3, 내방일="2024-02-01", 약품ID=1, 소모량=999.0)
    ]

    profile = _profile(rows, "2024-01-10", threshold=4).set_index(DRUG_ID_COL)

    assert profile.loc[1, RARE_STOCK_FLOOR_COL] == 20.0


def test_only_acute_visits_count():
    # Customer 1 becomes Chronic on 2024-02-01 (Revisit Match on drug 1);
    # their drug-2 visit on 2024-02-12 comes after that, so drug 2 has no
    # Acute dispensing at all. Their drug-1 visits up to 2024-02-01 stay
    # Acute, as in Mart 3 (see docs/adr/0003, "Update (implementing #36)").
    rows = [
        make_visit_row(
            조제판매ID=1, 고객ID=1, 내방일="2024-01-15", 다음내방일="2024-02-14", 약품ID=1, 소모량=7.0
        ),
        make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=1, 소모량=9.0),
        make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-02-12", 약품ID=2, 소모량=50.0),
    ]

    profile = _profile(rows, "2024-03-01").set_index(DRUG_ID_COL)

    assert 2 not in profile.index
    assert profile.loc[1, RARE_STOCK_FLOOR_COL] == 9.0


def test_rare_drugs_are_exactly_those_mart3_leaves_out():
    rows = _two_patient_drug_1() + [
        make_visit_row(조제판매ID=10 + i, 고객ID=10 + i, 내방일="2024-01-03", 약품ID=5)
        for i in range(3)
    ]
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + rows)

    profile = track2_rare_drug_allocation(raw_visits, "2024-01-10", rare_drug_patient_threshold=3)
    mart3 = build_marts(raw_visits, "2024-01-10", rare_drug_patient_threshold=3).mart3

    assert 1 in set(profile[DRUG_ID_COL]) and 1 not in set(mart3[DRUG_ID_COL])
    assert 5 not in set(profile[DRUG_ID_COL]) and 5 in set(mart3[DRUG_ID_COL])


def test_no_rare_drugs_gives_an_empty_profile():
    raw_visits = make_raw_visits([make_visit_row()])

    profile = track2_rare_drug_allocation(raw_visits, "2024-01-10", rare_drug_patient_threshold=1)

    assert profile.empty
    assert list(profile.columns) == TRACK2_RARE_DRUG_COLUMNS
