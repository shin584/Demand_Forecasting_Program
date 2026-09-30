import pandas as pd

from conftest import (
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.marts import (
    MART1_BOOLEAN_FEATURE_COLS,
    MART1_COLUMNS,
    MART1_NON_FEATURE_COLS,
    MART1_NUMERIC_FEATURE_COLS,
    MART1_TRAINING_COLUMNS,
    SNAPSHOT_DATE_COL,
    build_mart1_training_set,
    build_marts,
    sample_mart1_negatives,
    split_mart1_training_set,
)


def _chronic_customer_history():
    # Customer 1: Chronic via a genuine Revisit Match (V2 falls within V1's
    # 다음내방일 +/-30 day window and shares a drug).
    return [
        make_visit_row(
            조제판매ID=1,
            고객ID=1,
            내방일="2024-01-01",
            다음내방일="2024-01-31",
            처방조제일수=30,
            약품ID=1,
        ),
        make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
    ]


def test_returns_empty_frame_when_no_chronic_patients():
    # Only ever one visit each - no Revisit Match possible, so Acute.
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + [make_visit_row()])

    training_set = build_mart1_training_set(raw_visits)

    assert training_set.empty
    assert list(training_set.columns) == MART1_TRAINING_COLUMNS


def test_columns_match_mart1_training_columns():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    assert list(training_set.columns) == MART1_TRAINING_COLUMNS
    assert not training_set.empty


def test_one_positive_row_per_actual_visit():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    positives = training_set[training_set["내일_방문"] == True]  # noqa: E712
    # One positive per real visit, dated the calendar day right before it.
    assert set(positives["기준일자"]) == {
        pd.Timestamp("2023-12-31"),
        pd.Timestamp("2024-02-14"),
    }


def test_negative_rows_match_sample_mart1_negatives():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)
    expected_negative_dates = set(sample_mart1_negatives(raw_visits, {1})["기준일자"])

    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    assert set(negatives["기준일자"]) == expected_negative_dates
    assert len(negatives) == len(expected_negative_dates)


def test_weight_uses_the_full_raw_visits_not_as_of_filtered():
    # Severity flag sits on the customer's only-ever visit, so this is really
    # exercising that build_mart1_training_set's weight wiring reaches
    # sample_mart1_weights at all - not that it's as-of filtered (unlike
    # build_marts, it deliberately isn't, see the function's docstring).
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                다음내방일="2024-01-31",
                처방조제일수=30,
                약품ID=1,
                중증암등록대상자="Y",
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 약품ID=1),
        ]
    )

    training_set = build_mart1_training_set(raw_visits)

    assert (training_set["학습_가중치"] == 3.0).all()


def test_x_features_are_computed_as_of_each_rows_own_snapshot_date():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits).set_index("기준일자")

    # As of 2023-12-31 (the day before V1), customer 1 has no anchoring
    # visit yet - must not see V1 itself, dated the very next day.
    assert pd.isna(training_set.loc[pd.Timestamp("2023-12-31"), "마지막방문_경과일"])
    # As of 2024-02-14 (the day before V2), only V1 is knowable - 44 days
    # elapsed since it, and V2 itself must not leak in.
    assert training_set.loc[pd.Timestamp("2024-02-14"), "마지막방문_경과일"] == 44



def test_x_features_match_build_marts_snapshot_for_the_same_date():
    # Training (many snapshot dates at once) and daily inference (one
    # build_marts snapshot) must compute every X-feature identically.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=101, visit_id_start=10)
        + [
            make_visit_row(조제판매ID=1, 고객ID=1, 내방일="2024-01-01", 처방조제일수=20, 투약일수=20),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-25", 차상위대상자="Y", 보험구분=None),
            make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-03-10", 다음내방일="2024-04-09"),
        ]
    )
    feature_cols = [col for col in MART1_COLUMNS if col not in MART1_NON_FEATURE_COLS]

    training_set = build_mart1_training_set(raw_visits)

    # Before the high-frequency filler visits (dated 2024-01-01) exist,
    # build_marts' as-of top-2 exclusion can swallow drug 101 and customer 1
    # isn't Chronic yet, so there's no snapshot row to compare against.
    comparable = training_set[training_set[SNAPSHOT_DATE_COL] >= pd.Timestamp("2024-01-01")]
    assert comparable[SNAPSHOT_DATE_COL].nunique() > 5
    for snapshot_date, rows in comparable.groupby(SNAPSHOT_DATE_COL):
        mart1, _, _ = build_marts(raw_visits, snapshot_date)
        expected = mart1.set_index("고객ID").loc[rows["고객ID"], feature_cols]
        pd.testing.assert_frame_equal(rows.set_index("고객ID")[feature_cols], expected)


def test_x_features_have_lightgbm_ready_dtypes():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    for col in MART1_NUMERIC_FEATURE_COLS:
        assert training_set[col].dtype == "float64", col
    for col in MART1_BOOLEAN_FEATURE_COLS:
        assert training_set[col].dtype == "boolean", col

def _training_set_with_dates(dates: list) -> pd.DataFrame:
    return pd.DataFrame({SNAPSHOT_DATE_COL: pd.to_datetime(dates)})


def test_split_cuts_test_and_val_from_the_end_by_snapshot_date():
    # Max 기준일자 is 2025-01-16: test = last 6mo (>= 2024-07-16), val = the
    # 6mo before that (>= 2024-01-16, < 2024-07-16), train = everything else.
    training_set = _training_set_with_dates(
        ["2024-01-01", "2024-01-16", "2024-06-01", "2024-07-16", "2024-07-17", "2025-01-16"]
    )

    split = split_mart1_training_set(training_set)

    assert list(split.train[SNAPSHOT_DATE_COL]) == [pd.Timestamp("2024-01-01")]
    assert list(split.val[SNAPSHOT_DATE_COL]) == [
        pd.Timestamp("2024-01-16"),
        pd.Timestamp("2024-06-01"),
    ]
    assert list(split.test[SNAPSHOT_DATE_COL]) == [
        pd.Timestamp("2024-07-16"),
        pd.Timestamp("2024-07-17"),
        pd.Timestamp("2025-01-16"),
    ]


def test_split_boundary_date_belongs_to_the_newer_window():
    # A row dated exactly on a cutoff is inclusive on the more-recent side -
    # exercised in isolation from test_split_cuts_test_and_val_from_the_end_by_snapshot_date's
    # broader fixture.
    training_set = _training_set_with_dates(["2024-07-16", "2025-01-16"])

    split = split_mart1_training_set(training_set)

    assert list(split.test[SNAPSHOT_DATE_COL]) == [
        pd.Timestamp("2024-07-16"),
        pd.Timestamp("2025-01-16"),
    ]
    assert split.train.empty
    assert split.val.empty


def test_split_of_empty_training_set_returns_three_empty_frames():
    training_set = _training_set_with_dates([])

    split = split_mart1_training_set(training_set)

    assert split.train.empty
    assert split.val.empty
    assert split.test.empty


def test_split_columns_match_input_columns():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + _chronic_customer_history())
    training_set = build_mart1_training_set(raw_visits)

    split = split_mart1_training_set(training_set)

    assert list(split.train.columns) == MART1_TRAINING_COLUMNS
    assert list(split.val.columns) == MART1_TRAINING_COLUMNS
    assert list(split.test.columns) == MART1_TRAINING_COLUMNS
