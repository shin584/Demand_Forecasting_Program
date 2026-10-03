import pandas as pd

from conftest import make_raw_visits, make_visit_row
from pipeline.marts import family_totals_as_of


def test_family_visit_count_excludes_visits_on_or_after_as_of_date():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 가족ID=1, 내방일="2024-01-01"),
            make_visit_row(조제판매ID=2, 고객ID=2, 가족ID=1, 내방일="2024-02-01"),
            make_visit_row(조제판매ID=3, 고객ID=1, 가족ID=1, 내방일="2024-03-01"),
        ]
    )

    totals = family_totals_as_of(raw_visits, as_of_date="2024-01-01").set_index("가족ID")
    assert totals.loc[1, "가족_총내방"] == 0

    totals = family_totals_as_of(raw_visits, as_of_date="2024-03-01").set_index("가족ID")
    assert totals.loc[1, "가족_총내방"] == 2  # the 조제판매ID=3 visit itself is excluded


def test_family_visit_count_differs_across_as_of_dates_despite_constant_live_snapshot():
    # 가족총내방/가족총매출 (the raw live-snapshot columns) are held constant
    # across every row here, exactly like the real leaky extract ADR-0002
    # describes - family_totals_as_of must not just read them back.
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 가족ID=1, 내방일="2024-01-01", 가족총내방=12, 가족총매출=150000
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 가족ID=1, 내방일="2024-06-01", 가족총내방=12, 가족총매출=150000
            ),
        ]
    )

    early = family_totals_as_of(raw_visits, as_of_date="2024-02-01").set_index("가족ID")
    late = family_totals_as_of(raw_visits, as_of_date="2024-07-01").set_index("가족ID")

    assert early.loc[1, "가족_총내방"] != late.loc[1, "가족_총내방"]
    assert early.loc[1, "가족_총내방"] == 1
    assert late.loc[1, "가족_총내방"] == 2


def test_family_visit_count_pools_visits_across_family_members():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 가족ID=1, 내방일="2024-01-01"),
            make_visit_row(조제판매ID=2, 고객ID=2, 가족ID=1, 내방일="2024-01-15"),
            # Different family - must not be counted for 가족ID=1.
            make_visit_row(조제판매ID=3, 고객ID=3, 가족ID=2, 내방일="2024-01-10"),
        ]
    )

    totals = family_totals_as_of(raw_visits, as_of_date="2024-02-01").set_index("가족ID")

    assert totals.loc[1, "가족_총내방"] == 2
    assert totals.loc[2, "가족_총내방"] == 1


def test_family_visit_count_returns_one_row_per_family():
    raw_visits = make_raw_visits(
        [
            make_visit_row(조제판매ID=1, 고객ID=1, 가족ID=1, 내방일="2024-01-01"),
            # Same visit, second drug line - must not double-count the visit.
            make_visit_row(조제판매ID=1, 고객ID=1, 가족ID=1, 내방일="2024-01-01", 약품ID=2),
            make_visit_row(조제판매ID=2, 고객ID=1, 가족ID=1, 내방일="2024-02-01"),
        ]
    )

    totals = family_totals_as_of(raw_visits, as_of_date="2024-02-01")

    assert list(totals["가족ID"]) == [1]
    assert totals.set_index("가족ID").loc[1, "가족_총내방"] == 1
