import pandas as pd
import pytest

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    CONSUMPTION_COL,
    DRUG_ID_COL,
    MART3_BUCKET_MIN_OBSERVATIONS,
    MART3_COLUMNS,
    MART3_SEASON_MIN_OBSERVATIONS,
    MART3_SEASONS,
    MART3_WEEKDAYS,
    RARE_DRUG_PATIENT_THRESHOLD,
    SEASON_COL,
    WEEKDAY_COL,
    build_marts,
    resolve_mart3_backoff,
)


def _observation(drug_id, season, weekday, consumption):
    return {
        DRUG_ID_COL: drug_id,
        SEASON_COL: season,
        WEEKDAY_COL: weekday,
        CONSUMPTION_COL: consumption,
    }


def test_default_bucket_min_observations_is_five():
    assert MART3_BUCKET_MIN_OBSERVATIONS == 5


def test_default_season_min_observations_is_fifteen():
    assert MART3_SEASON_MIN_OBSERVATIONS == 15


def test_default_rare_drug_patient_threshold_is_five():
    assert RARE_DRUG_PATIENT_THRESHOLD == 5


def test_well_populated_bucket_returns_its_own_average():
    observations = pd.DataFrame(
        [
            _observation(1, "겨울", "월요일", 10.0),
            _observation(1, "겨울", "월요일", 20.0),
            _observation(1, "겨울", "월요일", 30.0),
            # A different bucket for the same drug -- proves the resolved
            # value comes from the bucket itself, not some wider average.
            _observation(1, "여름", "화요일", 999.0),
        ]
    )

    mart3 = resolve_mart3_backoff(observations, bucket_min_observations=3)

    bucket = mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
    assert bucket.loc[(1, "겨울", "월요일")] == 20.0


def test_sparse_bucket_falls_back_to_season_average():
    observations = pd.DataFrame(
        [
            # Bucket (여름, 화요일) has only 1 observation -- below threshold.
            _observation(2, "여름", "화요일", 100.0),
            # More observations elsewhere in the same season, pushing the
            # season-level count over threshold.
            _observation(2, "여름", "수요일", 20.0),
            _observation(2, "여름", "수요일", 40.0),
        ]
    )

    mart3 = resolve_mart3_backoff(
        observations, bucket_min_observations=3, season_min_observations=3
    )

    bucket = mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
    expected_season_avg = (100.0 + 20.0 + 40.0) / 3
    assert bucket.loc[(2, "여름", "화요일")] == pytest.approx(expected_season_avg)


def test_sparse_at_both_levels_falls_back_to_drug_overall_average():
    observations = pd.DataFrame(
        [
            _observation(3, "가을", "목요일", 50.0),
            # A single observation in a different season -- contributes only
            # to the drug's overall average, not to a season-level rescue.
            _observation(3, "봄", "월요일", 150.0),
        ]
    )

    mart3 = resolve_mart3_backoff(
        observations, bucket_min_observations=3, season_min_observations=3
    )

    bucket = mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
    expected_overall_avg = (50.0 + 150.0) / 2
    assert bucket.loc[(3, "가을", "목요일")] == pytest.approx(expected_overall_avg)


def test_bucket_min_observations_threshold_is_overridable():
    observations = pd.DataFrame(
        [
            _observation(4, "봄", "금요일", 10.0),
            _observation(4, "봄", "금요일", 30.0),
            _observation(4, "여름", "토요일", 1000.0),
            _observation(4, "여름", "토요일", 1000.0),
            _observation(4, "여름", "토요일", 1000.0),
            _observation(4, "여름", "토요일", 1000.0),
            _observation(4, "여름", "토요일", 1000.0),
        ]
    )

    strict = resolve_mart3_backoff(
        observations, bucket_min_observations=3, season_min_observations=3
    )
    lenient = resolve_mart3_backoff(
        observations, bucket_min_observations=2, season_min_observations=2
    )

    strict_value = strict.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[
        CONSUMPTION_COL
    ].loc[(4, "봄", "금요일")]
    lenient_value = lenient.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[
        CONSUMPTION_COL
    ].loc[(4, "봄", "금요일")]

    # threshold=2: the bucket's own 2 observations meet the bar.
    assert lenient_value == 20.0
    # threshold=3: the bucket (and its season, which pools the same 2
    # observations) fall short, so it backs off all the way to the drug's
    # overall average across all 7 observations.
    assert strict_value == pytest.approx((10 + 30 + 1000 * 5) / 7)


def test_season_sufficient_under_old_shared_threshold_but_not_new_falls_through_to_drug_average():
    # Season count is 8 -- would have cleared the old shared threshold (5),
    # but falls short of a materially higher season_min_observations (10).
    # The bucket itself (2 observations) is insufficient either way, so with
    # both tiers now insufficient, the result must fall all the way through
    # to the drug's overall average rather than resting on the season.
    observations = pd.DataFrame(
        [
            _observation(6, "봄", "월요일", 10.0),
            _observation(6, "봄", "월요일", 20.0),
        ]
        + [_observation(6, "봄", "화요일", 100.0) for _ in range(6)]
    )

    mart3 = resolve_mart3_backoff(
        observations, bucket_min_observations=5, season_min_observations=10
    )

    bucket = mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
    expected_overall_avg = (10.0 + 20.0 + 100.0 * 6) / 8
    assert bucket.loc[(6, "봄", "월요일")] == pytest.approx(expected_overall_avg)


def test_season_clears_materially_higher_threshold_still_falls_back_to_season_average():
    # Season count is 15 -- clears a materially higher season_min_observations
    # bar than the old shared threshold ever required. The bucket itself is
    # insufficient, so the season average is still used.
    observations = pd.DataFrame(
        [
            _observation(7, "여름", "월요일", 10.0),
            _observation(7, "여름", "월요일", 20.0),
        ]
        + [_observation(7, "여름", "화요일", 100.0) for _ in range(13)]
    )

    mart3 = resolve_mart3_backoff(
        observations, bucket_min_observations=5, season_min_observations=15
    )

    bucket = mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
    expected_season_avg = (10.0 + 20.0 + 100.0 * 13) / 15
    assert bucket.loc[(7, "여름", "월요일")] == pytest.approx(expected_season_avg)


def test_every_drug_gets_a_full_season_by_weekday_grid():
    observations = pd.DataFrame([_observation(5, "봄", "월요일", 42.0)])

    mart3 = resolve_mart3_backoff(observations, bucket_min_observations=1)

    combos = set(zip(mart3[DRUG_ID_COL], mart3[SEASON_COL], mart3[WEEKDAY_COL]))
    expected = {
        (5, season, weekday) for season in MART3_SEASONS for weekday in MART3_WEEKDAYS
    }
    assert combos == expected
    assert list(mart3.columns) == MART3_COLUMNS


def test_mart3_built_only_from_acute_patient_visits():
    raw_visits = make_raw_visits(
        # occurrences=4 keeps 90001/90002 strictly more frequent than 약품ID=1
        # (3 occurrences below), so 약품ID=1 isn't itself swept into the
        # Revisit Match top-2 exclusion (see ADR-0001).
        make_high_frequency_filler_visits(4)
        + [
            # Customer 1: Chronic via Revisit Match on 약품ID=1 within the
            # +/-30 day window of its 다음내방일 -- must not contribute.
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-15",
                다음내방일="2024-02-14",
                약품ID=1,
                소모량=999.0,
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=1, 소모량=999.0
            ),
            # Customer 2: Acute -- single visit, no Revisit Match possible --
            # does contribute.
            make_visit_row(
                조제판매ID=3, 고객ID=2, 내방일="2024-01-15", 약품ID=1, 소모량=50.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits,
        as_of_date="2024-03-01",
        mart3_bucket_min_observations=1,
        rare_drug_patient_threshold=1,
    ).mart3

    drug1_values = mart3.loc[mart3[DRUG_ID_COL] == 1, CONSUMPTION_COL]
    assert not drug1_values.empty
    assert (drug1_values == 50.0).all()


def test_mart3_excludes_visits_after_as_of_date():
    raw_visits = make_raw_visits(
        [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-15", 약품ID=1, 소모량=50.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-06-01", 약품ID=1, 소모량=999.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits,
        as_of_date="2024-02-01",
        mart3_bucket_min_observations=1,
        rare_drug_patient_threshold=1,
    ).mart3

    drug1_values = mart3.loc[mart3[DRUG_ID_COL] == 1, CONSUMPTION_COL]
    assert not drug1_values.empty
    assert (drug1_values == 50.0).all()


def test_build_marts_mart3_bucket_min_observations_is_overridable():
    raw_visits = make_raw_visits(
        [
            # Winter Mondays -- same bucket, 2 observations.
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-15", 약품ID=1, 소모량=10.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-22", 약품ID=1, 소모량=30.0
            ),
            # A third, differently-bucketed observation, so the bucket
            # average (20.0) differs from the drug's overall average.
            make_visit_row(
                조제판매ID=3, 고객ID=3, 내방일="2024-07-03", 약품ID=1, 소모량=400.0
            ),
        ]
    )

    default_mart3 = build_marts(
        raw_visits, as_of_date="2024-08-01", rare_drug_patient_threshold=1
    ).mart3
    lenient_mart3 = build_marts(
        raw_visits,
        as_of_date="2024-08-01",
        mart3_bucket_min_observations=2,
        rare_drug_patient_threshold=1,
    ).mart3

    bucket = (1, "겨울", "월요일")
    default_value = default_mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[
        CONSUMPTION_COL
    ].loc[bucket]
    lenient_value = lenient_mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[
        CONSUMPTION_COL
    ].loc[bucket]

    # Default threshold (5) isn't met by 2 observations, so it backs off all
    # the way to the drug's overall average across all 3 observations.
    assert default_value == pytest.approx((10 + 30 + 400) / 3)
    # threshold=2: the bucket's own 2 observations meet the bar.
    assert lenient_value == 20.0


def test_mart3_excludes_a_chronic_only_drug_entirely():
    # Customer 1 is Chronic (a genuine Revisit Match on 약품ID=1), so 약품ID=1
    # contributes nothing to Mart 3 -- only the filler customers' visits
    # (Acute, single-visit) do.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(4)
        + [
            make_visit_row(
                조제판매ID=1,
                고객ID=1,
                내방일="2024-01-01",
                다음내방일="2024-01-31",
                약품ID=1,
            ),
            make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-01-15", 약품ID=1),
        ]
    )

    # as_of_date is at (not before) customer 1's revisit, so the match is
    # already knowable at this snapshot (see issue #11).
    mart3 = build_marts(raw_visits, as_of_date="2024-01-15").mart3

    assert list(mart3.columns) == MART3_COLUMNS
    assert 1 not in set(mart3[DRUG_ID_COL])


def test_mart3_excludes_a_drug_below_the_rare_drug_patient_threshold():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(4)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1, 소모량=10.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1, 소모량=20.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits, as_of_date="2024-02-01", rare_drug_patient_threshold=3
    ).mart3

    # Only 2 distinct patients -- strictly below the threshold of 3 -- so
    # the drug gets no Mart 3 row at all.
    assert 1 not in set(mart3[DRUG_ID_COL])


def test_mart3_includes_a_drug_at_exactly_the_rare_drug_patient_threshold():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(4)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1, 소모량=10.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1, 소모량=20.0
            ),
            make_visit_row(
                조제판매ID=3, 고객ID=3, 내방일="2024-01-15", 약품ID=1, 소모량=30.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits, as_of_date="2024-02-01", rare_drug_patient_threshold=3
    ).mart3

    # Exactly 3 distinct patients -- CONTEXT.md's rare-drug cutoff is
    # strictly-less-than, so a drug at the threshold is still included.
    assert 1 in set(mart3[DRUG_ID_COL])


def test_mart3_rare_drug_filter_respects_as_of_date():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(4)
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-05", 약품ID=1, 소모량=10.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-01-10", 약품ID=1, 소모량=20.0
            ),
            # This third patient only shows up after as_of_date -- the drug
            # only crosses the threshold later, which this snapshot must not
            # already know about.
            make_visit_row(
                조제판매ID=3, 고객ID=3, 내방일="2024-03-01", 약품ID=1, 소모량=30.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits, as_of_date="2024-02-01", rare_drug_patient_threshold=3
    ).mart3

    assert 1 not in set(mart3[DRUG_ID_COL])


def test_mart3_rare_drug_filter_excludes_patients_outside_trailing_12_months():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(4)
        + [
            # All three patients visited well over 12 months before
            # as_of_date -- outside the trailing-12-month window, so none of
            # them count toward the snapshot's patient total despite being
            # at-or-before as_of_date.
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2022-01-05", 약품ID=1, 소모량=10.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2022-01-10", 약품ID=1, 소모량=20.0
            ),
            make_visit_row(
                조제판매ID=3, 고객ID=3, 내방일="2022-01-15", 약품ID=1, 소모량=30.0
            ),
        ]
    )

    mart3 = build_marts(
        raw_visits, as_of_date="2024-02-01", rare_drug_patient_threshold=3
    ).mart3

    assert 1 not in set(mart3[DRUG_ID_COL])
