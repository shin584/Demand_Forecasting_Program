import pandas as pd

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import MART2_COLUMNS, MART3_COLUMNS, build_marts


def test_build_marts_returns_three_well_defined_marts():
    raw_visits = make_raw_visits([make_visit_row()])

    result = build_marts(raw_visits, as_of_date="2024-01-01")
    mart1, mart2, mart3 = result

    assert isinstance(mart1, pd.DataFrame)
    assert isinstance(mart2, pd.DataFrame)
    assert isinstance(mart3, pd.DataFrame)
    assert list(mart1.columns) == ["고객ID", "내일_방문", "만성질환여부"]
    assert list(mart2.columns) == MART2_COLUMNS
    assert list(mart3.columns) == MART3_COLUMNS
    # Named-tuple access works alongside positional unpacking.
    assert result.mart1 is mart1
    assert result.mart2 is mart2
    assert result.mart3 is mart3


def test_next_day_visit_label_true_positive_and_true_negative():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Customer 1: actually visits the day right after the snapshot date,
            # and is Chronic (Revisit Match on 약품ID=1, within the window of
            # its 다음내방일) so it remains in Mart 1 to check the Y label.
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-02", 약품ID=1),
            # Customer 2: has a later visit, but not the immediate next day.
            # Also Chronic (Revisit Match on 약품ID=2) for the same reason.
            make_visit_row(
                조제판매ID=3, 고객ID=2, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=2
            ),
            make_visit_row(조제판매ID=4, 고객ID=2, 내방일="2024-01-10", 약품ID=2),
        ]
    )

    mart1, _, _ = build_marts(raw_visits, as_of_date="2024-01-01")

    labels = mart1.set_index("고객ID")["내일_방문"]
    assert labels.loc[1] == True  # noqa: E712 (readable as a boolean truth table)
    assert labels.loc[2] == False  # noqa: E712


def test_build_marts_is_deterministic_and_side_effect_free():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-02", 약품ID=1),
        ]
    )
    raw_visits_before = raw_visits.copy(deep=True)

    result_a = build_marts(raw_visits, as_of_date="2024-01-01")
    result_b = build_marts(raw_visits, as_of_date="2024-01-01")

    pd.testing.assert_frame_equal(result_a.mart1, result_b.mart1)
    pd.testing.assert_frame_equal(result_a.mart2, result_b.mart2)
    pd.testing.assert_frame_equal(result_a.mart3, result_b.mart3)
    pd.testing.assert_frame_equal(raw_visits, raw_visits_before)


def test_mart2_uses_latest_consumption_at_or_before_as_of_date():
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=30.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=1, 소모량=60.0
            ),
        ]
    )

    mart2 = build_marts(raw_visits, as_of_date="2024-02-01").mart2

    value = mart2.set_index(["고객ID", "약품ID"])["최근소모량"]
    assert value.loc[(1, 1)] == 60.0


def test_mart2_excludes_visits_after_as_of_date():
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=30.0
            ),
            # Later than as_of_date - must not leak into the mart.
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-03-01", 약품ID=1, 소모량=999.0
            ),
        ]
    )

    mart2 = build_marts(raw_visits, as_of_date="2024-01-15").mart2

    value = mart2.set_index(["고객ID", "약품ID"])["최근소모량"]
    assert value.loc[(1, 1)] == 30.0


def test_mart2_differs_across_as_of_dates_on_the_same_raw_dataset():
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=30.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=1, 소모량=60.0
            ),
        ]
    )

    early = build_marts(raw_visits, as_of_date="2024-01-01").mart2
    late = build_marts(raw_visits, as_of_date="2024-02-01").mart2

    early_value = early.set_index(["고객ID", "약품ID"])["최근소모량"].loc[(1, 1)]
    late_value = late.set_index(["고객ID", "약품ID"])["최근소모량"].loc[(1, 1)]
    assert early_value != late_value
    assert early_value == 30.0
    assert late_value == 60.0


def test_mart2_is_per_customer_and_drug():
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=1, 소모량=30.0
            ),
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 약품ID=2, 소모량=15.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-01", 약품ID=1, 소모량=99.0
            ),
        ]
    )

    mart2 = build_marts(raw_visits, as_of_date="2024-01-01").mart2

    value = mart2.set_index(["고객ID", "약품ID"])["최근소모량"]
    assert value.loc[(1, 1)] == 30.0
    assert value.loc[(1, 2)] == 15.0
    assert value.loc[(2, 1)] == 99.0


def test_mart1_includes_only_chronic_patients_and_derives_chronic_flag():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            # Customer 1: Chronic - has a genuine Revisit Match.
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=1
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
            # Customer 2: Acute - only ever has one visit, so no Revisit
            # Match is possible; excluded from Mart 1.
            make_visit_row(
                조제판매ID=3, 고객ID=2, 내방일="2024-01-01", 다음내방일="2024-01-31", 약품ID=2
            ),
        ]
    )

    mart1, _, _ = build_marts(raw_visits, as_of_date="2024-01-01")

    assert set(mart1["고객ID"]) == {1}
    assert mart1.set_index("고객ID")["만성질환여부"].loc[1] == True  # noqa: E712
