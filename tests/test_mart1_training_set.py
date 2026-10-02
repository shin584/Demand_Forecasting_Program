import pandas as pd
import pytest

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
    LAPSE_HORIZON_DAYS,
    MART1_TRAINING_COLUMNS,
    SNAPSHOT_DATE_COL,
    WINDOW_WIDTH_COL,
    build_mart1_training_set,
    build_marts,
    sample_mart1_negatives,
    split_mart1_training_set,
)


# _chronic_customer_history uses drug 1 on 4 visits; the filler drugs need
# more occurrences than that to stay the top-2 Revisit Match exclusion.
_FILLER_OCCURRENCES = 5


def _chronic_customer_history():
    # Customer 1: Chronic via a genuine Revisit Match (V2 falls within V1's
    # 다음내방일 +/-30 day window and shares a drug), so Chronic since V2's
    # 2024-02-15. V3 and V4 give the training set rows dated after that.
    return [
        make_visit_row(
            조제판매ID=1,
            고객ID=1,
            내방일="2024-01-01",
            다음내방일="2024-01-31",
            처방조제일수=30,
            약품ID=1,
        ),
        make_visit_row(
            조제판매ID=2, 고객ID=1, 내방일="2024-02-15", 다음내방일="2024-03-16", 약품ID=1
        ),
        make_visit_row(
            조제판매ID=3, 고객ID=1, 내방일="2024-03-20", 다음내방일="2024-04-19", 약품ID=1
        ),
        make_visit_row(조제판매ID=4, 고객ID=1, 내방일="2024-04-20", 약품ID=1),
    ]


def test_returns_empty_frame_when_no_chronic_patients():
    # Only ever one visit each - no Revisit Match possible, so Acute.
    raw_visits = make_raw_visits(make_high_frequency_filler_visits() + [make_visit_row()])

    training_set = build_mart1_training_set(raw_visits)

    assert training_set.empty
    assert list(training_set.columns) == MART1_TRAINING_COLUMNS


def test_columns_match_mart1_training_columns():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    assert list(training_set.columns) == MART1_TRAINING_COLUMNS
    assert not training_set.empty


def test_one_positive_row_per_actual_visit_from_the_chronic_since_date():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    positives = training_set[training_set["내일_방문"] == True]  # noqa: E712
    # One positive per real visit, dated the calendar day right before it --
    # except V1's (2023-12-31) and V2's (2024-02-14), which fall before the
    # 2024-02-15 Chronic-since Date.
    assert set(positives["기준일자"]) == {
        pd.Timestamp("2024-03-19"),
        pd.Timestamp("2024-04-19"),
    }


def test_negative_rows_match_sample_mart1_negatives_from_the_chronic_since_date():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)
    sampled_dates = sample_mart1_negatives(raw_visits, {1})["기준일자"]
    expected_negative_dates = set(sampled_dates[sampled_dates >= pd.Timestamp("2024-02-15")])

    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    assert set(negatives["기준일자"]) == expected_negative_dates
    assert len(negatives) == len(expected_negative_dates)


def test_rows_before_the_chronic_since_date_are_dropped():
    # Customer 1 visits in October (drug 2, never recurs) and January before
    # V2's 2024-02-15 match makes them Chronic. Not Chronic yet as of those
    # earlier dates, so no row of either kind dated before it belongs.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + [
            make_visit_row(
                조제판매ID=10, 고객ID=1, 내방일="2023-10-01", 다음내방일="2023-10-31", 약품ID=2
            )
        ]
        + _chronic_customer_history()
    )
    sampled = sample_mart1_negatives(raw_visits, {1})

    training_set = build_mart1_training_set(raw_visits)

    assert (training_set[SNAPSHOT_DATE_COL] >= pd.Timestamp("2024-02-15")).all()
    # Negatives anchored on the October and January visits are gone; those
    # anchored on V2 and V3 are all kept.
    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    kept_anchors = sampled[sampled["조제판매ID"].isin([2, 3])]
    assert sorted(negatives[SNAPSHOT_DATE_COL]) == sorted(kept_anchors[SNAPSHOT_DATE_COL])
    positives = training_set[training_set["내일_방문"] == True]  # noqa: E712
    assert set(positives[SNAPSHOT_DATE_COL]) == {
        pd.Timestamp("2024-03-19"),
        pd.Timestamp("2024-04-19"),
    }


def test_returns_empty_frame_when_every_row_predates_the_chronic_since_date():
    # The match is only observed on the customer's last visit: every positive
    # is dated the day before a visit, and the last visit is also the
    # extract's last 내방일, so every negative it anchors would fall past the
    # extract's end. Nothing is left on or after 2024-02-15.
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history()[:2])

    training_set = build_mart1_training_set(raw_visits)

    assert training_set.empty
    assert list(training_set.columns) == MART1_TRAINING_COLUMNS


def test_each_customer_is_filtered_by_their_own_chronic_since_date():
    # Customer 2 is Chronic from 2024-04-15, two months after customer 1.
    customer_2 = [
        make_visit_row(
            조제판매ID=20, 고객ID=2, 가족ID=2, 내방일="2024-03-01", 다음내방일="2024-03-31", 약품ID=3
        ),
        make_visit_row(
            조제판매ID=21, 고객ID=2, 가족ID=2, 내방일="2024-04-15", 다음내방일="2024-05-15", 약품ID=3
        ),
        make_visit_row(조제판매ID=22, 고객ID=2, 가족ID=2, 내방일="2024-05-20", 약품ID=3),
    ]
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history() + customer_2
    )

    training_set = build_mart1_training_set(raw_visits)

    earliest = training_set.groupby("고객ID")[SNAPSHOT_DATE_COL].min()
    assert earliest.loc[1] < pd.Timestamp("2024-04-15")
    assert earliest.loc[1] >= pd.Timestamp("2024-02-15")
    assert earliest.loc[2] >= pd.Timestamp("2024-04-15")


def test_weight_uses_the_full_raw_visits_not_as_of_filtered():
    # Severity flag sits only on the customer's first visit, so this is
    # really exercising that build_mart1_training_set's weight wiring reaches
    # sample_mart1_weights at all - not that it's as-of filtered (unlike
    # build_marts, it deliberately isn't, see the function's docstring).
    history = _chronic_customer_history()
    history[0]["중증암등록대상자"] = "Y"
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + history)

    training_set = build_mart1_training_set(raw_visits)

    positives = training_set[training_set["내일_방문"] == True]  # noqa: E712
    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    assert not positives.empty
    assert (positives["학습_가중치"] == 3.0).all()
    # Negatives scale the same tier by their mean-1 window weight.
    assert negatives["학습_가중치"].mean() == pytest.approx(3.0)


def test_negative_weights_are_proportional_to_window_width_with_mean_1():
    # Customer 2 is a second Chronic customer with a different cycle length,
    # so the mean is taken across every negative, not per customer.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer_history()
        + [
            make_visit_row(
                조제판매ID=20, 고객ID=2, 가족ID=2, 내방일="2024-01-05",
                다음내방일="2024-02-04", 처방조제일수=7, 약품ID=5,
            ),
            make_visit_row(
                조제판매ID=21, 고객ID=2, 가족ID=2, 내방일="2024-02-01",
                처방조제일수=7, 약품ID=5,
            ),
        ]
    )
    sampled = sample_mart1_negatives(raw_visits)

    training_set = build_mart1_training_set(raw_visits)

    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    widths = negatives.merge(
        sampled[["고객ID", SNAPSHOT_DATE_COL, WINDOW_WIDTH_COL]],
        on=["고객ID", SNAPSHOT_DATE_COL],
        validate="one_to_one",
    )
    assert set(widths["고객ID"]) == {1, 2}
    assert len(widths) == len(negatives)
    expected_scale = widths[WINDOW_WIDTH_COL] / widths[WINDOW_WIDTH_COL].mean()
    # Both customers sit in the same chronic tier (2.0), which positives keep.
    assert (training_set.loc[training_set["내일_방문"] == True, "학습_가중치"] == 2.0).all()  # noqa: E712
    assert list(widths["학습_가중치"]) == pytest.approx(list(2.0 * expected_scale))
    assert (widths["학습_가중치"] / 2.0).mean() == pytest.approx(1.0)
    assert widths[WINDOW_WIDTH_COL].nunique() > 1


def test_x_features_are_computed_as_of_each_rows_own_snapshot_date():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits).set_index("기준일자")

    # As of 2024-03-19 (the day before V3), V2 anchors - 33 days elapsed
    # since it, and V3 itself must not leak in.
    assert training_set.loc[pd.Timestamp("2024-03-19"), "마지막방문_경과일"] == 33
    # As of 2024-04-19 (the day before V4), V3 anchors - 30 days elapsed.
    assert training_set.loc[pd.Timestamp("2024-04-19"), "마지막방문_경과일"] == 30



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
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

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
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())
    training_set = build_mart1_training_set(raw_visits)

    split = split_mart1_training_set(training_set)

    assert list(split.train.columns) == MART1_TRAINING_COLUMNS
    assert list(split.val.columns) == MART1_TRAINING_COLUMNS
    assert list(split.test.columns) == MART1_TRAINING_COLUMNS


def test_no_negative_row_has_a_visit_on_the_day_after_its_snapshot_date():
    # 처방조제일수=33 on V2/V3 samples days 30/31/32 of each cycle, and V4
    # (2024-04-20) comes 31 days after V3 -- so V3's day-30 sample is the day
    # before V4, a mislabelled negative without a next-visit bound.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + [
            {**row, "처방조제일수": 33} if row["조제판매ID"] in (2, 3) else row
            for row in _chronic_customer_history()
        ]
    )

    training_set = build_mart1_training_set(raw_visits)

    visit_dates = set(raw_visits.loc[raw_visits["고객ID"] == 1, "내방일"])
    negatives = training_set[training_set["내일_방문"] == False]  # noqa: E712
    next_days = negatives["기준일자"] + pd.Timedelta(days=1)
    assert not negatives.empty
    assert not next_days.isin(visit_dates).any()


def test_no_row_is_dated_on_or_after_the_extracts_last_visit():
    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())

    training_set = build_mart1_training_set(raw_visits)

    assert training_set["기준일자"].max() < raw_visits["내방일"].max()


def test_a_sample_on_or_after_the_extracts_last_visit_is_dropped(monkeypatch):
    # Negative Sampling applies the same bound itself; this pins the
    # training set's own guard for a future sampling change that wouldn't (a row dated on the last
    # 내방일 has its label on the day after the extract ends).
    import pipeline.marts as marts

    raw_visits = make_raw_visits(make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_history())
    real_sampler = marts.sample_mart1_negatives

    def sampler_with_late_rows(raw_visits, chronic_customer_ids=None, lapse_horizon_days=LAPSE_HORIZON_DAYS):
        negatives = real_sampler(raw_visits, chronic_customer_ids, lapse_horizon_days)
        late = negatives.iloc[[0, 0]].assign(
            기준일자=[pd.Timestamp("2024-04-20"), pd.Timestamp("2024-04-25")]
        )
        return pd.concat([negatives, late], ignore_index=True)

    monkeypatch.setattr(marts, "sample_mart1_negatives", sampler_with_late_rows)

    training_set = build_mart1_training_set(raw_visits)

    assert training_set["기준일자"].max() < pd.Timestamp("2024-04-20")
