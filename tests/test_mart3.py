import pandas as pd
import pytest

from conftest import make_high_frequency_filler_visits, make_raw_visits, make_visit_row
from pipeline.marts import (
    CONSUMPTION_COL,
    DRUG_ID_COL,
    MART3_BUCKET_MIN_DAYS,
    MART3_COLUMNS,
    MART3_SEASON_MIN_DAYS,
    MART3_SEASONS,
    MART3_VALUE_COL,
    MART3_WEEKDAYS,
    RARE_DRUG_PATIENT_THRESHOLD,
    SEASON_COL,
    VISIT_DATE_COL,
    WEEKDAY_COL,
    build_marts,
    resolve_mart3_backoff,
    season_and_weekday_for,
)


def _observation(drug_id, visit_date, consumption):
    return {
        DRUG_ID_COL: drug_id,
        VISIT_DATE_COL: pd.Timestamp(visit_date),
        CONSUMPTION_COL: consumption,
    }


def _daily_value(mart3, drug_id, season, weekday):
    return mart3.set_index([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[MART3_VALUE_COL].loc[
        (drug_id, season, weekday)
    ]


def test_default_bucket_min_days_is_five():
    assert MART3_BUCKET_MIN_DAYS == 5


def test_default_season_min_days_is_fifteen():
    assert MART3_SEASON_MIN_DAYS == 15


def test_default_rare_drug_patient_threshold_is_five():
    assert RARE_DRUG_PATIENT_THRESHOLD == 5


def test_bucket_value_is_its_total_consumption_over_its_calendar_days():
    # Drug 1 first dispensed Monday 2024-01-01; as of Sunday 2024-01-28 its
    # 겨울/월요일 bucket spans 4 calendar Mondays (1, 8, 15, 22), two of
    # them with no dispensing at all -- those count as zero days, so the
    # bucket holds expected daily consumption, not the mean per dispensing.
    observations = pd.DataFrame(
        [
            _observation(1, "2024-01-01", 10.0),
            _observation(1, "2024-01-08", 20.0),
            # A different bucket for the same drug -- proves the resolved
            # value comes from the bucket itself, not some wider average.
            _observation(1, "2024-01-02", 999.0),
        ]
    )

    mart3 = resolve_mart3_backoff(observations, "2024-01-28", bucket_min_days=4)

    assert _daily_value(mart3, 1, "겨울", "월요일") == pytest.approx((10.0 + 20.0) / 4)


def test_bucket_with_days_but_no_dispensing_resolves_to_zero():
    # Sundays (the pharmacy's closed day) have 4 calendar days in the
    # window but no dispensing -- a genuine zero, not a gap to back off from.
    observations = pd.DataFrame([_observation(1, "2024-01-01", 10.0)])

    mart3 = resolve_mart3_backoff(observations, "2024-01-28", bucket_min_days=4)

    assert _daily_value(mart3, 1, "겨울", "일요일") == 0.0


def test_bucket_days_are_counted_from_the_drugs_own_first_dispensing():
    # Drug 2 is first dispensed Monday 2024-01-15, so as of 2024-01-28 its
    # 겨울/월요일 bucket spans only 2 Mondays (15, 22) -- not the 4 drug 1's
    # earlier first dispensing gives it.
    observations = pd.DataFrame(
        [
            _observation(1, "2024-01-01", 10.0),
            _observation(2, "2024-01-15", 30.0),
        ]
    )

    mart3 = resolve_mart3_backoff(observations, "2024-01-28", bucket_min_days=2)

    assert _daily_value(mart3, 2, "겨울", "월요일") == pytest.approx(30.0 / 2)
    assert _daily_value(mart3, 1, "겨울", "월요일") == pytest.approx(10.0 / 4)


def test_bucket_with_too_few_days_falls_back_to_season_daily_average():
    # First dispensed Monday 2024-01-01, as of Wednesday 2024-01-10: 10
    # winter days, only 2 of them Mondays.
    observations = pd.DataFrame(
        [
            _observation(2, "2024-01-01", 100.0),
            _observation(2, "2024-01-03", 20.0),
            _observation(2, "2024-01-10", 40.0),
        ]
    )

    mart3 = resolve_mart3_backoff(
        observations, "2024-01-10", bucket_min_days=3, season_min_days=10
    )

    assert _daily_value(mart3, 2, "겨울", "월요일") == pytest.approx((100.0 + 20.0 + 40.0) / 10)


@pytest.mark.parametrize(
    ("season_min_days", "expected"),
    # 11 winter days (2024-02-19 to 02-29) and 3 spring days (03-01 to
    # 03-03): the 겨울/월요일 bucket (2 Mondays) is short of bucket_min_days
    # either way, so season_min_days alone decides between the winter
    # season's daily average and the drug's overall one.
    [(11, 22.0 / 11), (12, (22.0 + 70.0) / 14)],
)
def test_season_min_days_decides_between_season_and_overall_daily_average(
    season_min_days, expected
):
    observations = pd.DataFrame(
        [
            _observation(3, "2024-02-19", 22.0),
            _observation(3, "2024-03-01", 70.0),
        ]
    )

    mart3 = resolve_mart3_backoff(
        observations, "2024-03-03", bucket_min_days=3, season_min_days=season_min_days
    )

    assert _daily_value(mart3, 3, "겨울", "월요일") == pytest.approx(expected)


def test_season_with_no_days_yet_falls_back_to_overall_daily_average():
    # First dispensed 2024-01-01, as of 2024-01-10: no summer day has
    # happened since, so every 여름 bucket rests on the overall average.
    observations = pd.DataFrame([_observation(4, "2024-01-01", 50.0)])

    mart3 = resolve_mart3_backoff(observations, "2024-01-10")

    assert _daily_value(mart3, 4, "여름", "화요일") == pytest.approx(50.0 / 10)


def test_missing_consumption_counts_as_zero():
    observations = pd.DataFrame(
        [_observation(1, "2024-01-01", 10.0), _observation(1, "2024-01-08", float("nan"))]
    )

    mart3 = resolve_mart3_backoff(observations, "2024-01-28", bucket_min_days=4)

    assert _daily_value(mart3, 1, "겨울", "월요일") == pytest.approx(10.0 / 4)


def test_every_drug_gets_a_full_season_by_weekday_grid():
    observations = pd.DataFrame([_observation(5, "2024-01-01", 42.0)])

    mart3 = resolve_mart3_backoff(observations, "2024-01-01", bucket_min_days=1)

    combos = set(zip(mart3[DRUG_ID_COL], mart3[SEASON_COL], mart3[WEEKDAY_COL]))
    expected = {
        (5, season, weekday) for season in MART3_SEASONS for weekday in MART3_WEEKDAYS
    }
    assert combos == expected
    assert list(mart3.columns) == MART3_COLUMNS


def test_no_observations_gives_an_empty_mart3():
    observations = pd.DataFrame(columns=[DRUG_ID_COL, VISIT_DATE_COL, CONSUMPTION_COL])

    mart3 = resolve_mart3_backoff(observations, "2024-01-01")

    assert mart3.empty
    assert list(mart3.columns) == MART3_COLUMNS


# More filler occurrences than 약품ID=1 gets in any fixture below, so it
# stays outside the Revisit Match top-2 exclusion (see ADR-0001).
_FILLER_OCCURRENCES = 6


def _chronic_customer_1_visits(post_chronic_consumption=999.0):
    """Customer 1 becomes Chronic on 2024-02-01 -- the later visit of a
    Revisit Match on 약품ID=1 within +/-30 days of its 다음내방일 -- then
    gets 약품ID=1 once more afterwards, on 2024-02-12."""
    return [
        make_visit_row(
            조제판매ID=1, 고객ID=1, 내방일="2024-01-15", 다음내방일="2024-02-14", 약품ID=1, 소모량=7.0
        ),
        make_visit_row(조제판매ID=2, 고객ID=1, 내방일="2024-02-01", 약품ID=1, 소모량=7.0),
        make_visit_row(
            조제판매ID=3, 고객ID=1, 내방일="2024-02-12", 약품ID=1, 소모량=post_chronic_consumption
        ),
    ]


def test_mart3_excludes_visits_made_after_the_customer_became_chronic():
    filler = make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
    acute = [make_visit_row(조제판매ID=4, 고객ID=2, 내방일="2024-01-15", 약품ID=1, 소모량=50.0)]
    kwargs = dict(as_of_date="2024-03-01", mart3_bucket_min_days=1, rare_drug_patient_threshold=1)

    with_post_chronic = build_marts(
        make_raw_visits(filler + _chronic_customer_1_visits(999.0) + acute), **kwargs
    ).mart3
    without_post_chronic = build_marts(
        make_raw_visits(filler + _chronic_customer_1_visits(0.0) + acute), **kwargs
    ).mart3

    pd.testing.assert_frame_equal(with_post_chronic, without_post_chronic)


def test_mart3_counts_a_chronic_customers_visits_up_to_their_chronic_since_date():
    # Customer 1's visits on 2024-01-15 and on their Chronic-since Date
    # (2024-02-01, a Thursday) came while they were still Acute -- Track 1
    # didn't score them the day before -- so Mart 3 counts both, even
    # though customer 1 is Chronic by as_of_date.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES) + _chronic_customer_1_visits()
    )

    mart3 = build_marts(
        raw_visits, as_of_date="2024-03-01", mart3_bucket_min_days=1, rare_drug_patient_threshold=1
    ).mart3

    # Drug 1 was first dispensed 2024-01-15; winter runs to 2024-02-29.
    winter_thursdays = pd.date_range("2024-01-15", "2024-02-29", freq="W-THU")
    assert _daily_value(mart3, 1, "겨울", "목요일") == pytest.approx(7.0 / len(winter_thursdays))
    winter_mondays = pd.date_range("2024-01-15", "2024-02-29", freq="W-MON")
    assert _daily_value(mart3, 1, "겨울", "월요일") == pytest.approx(7.0 / len(winter_mondays))


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
        mart3_bucket_min_days=1,
        rare_drug_patient_threshold=1,
    ).mart3

    # Only the 2024-01-15 Monday dispensing, over the 3 Mondays (01-15,
    # 01-22, 01-29) since.
    assert _daily_value(mart3, 1, "겨울", "월요일") == pytest.approx(50.0 / 3)
    assert mart3[MART3_VALUE_COL].max() < 999.0


def test_build_marts_mart3_bucket_min_days_is_overridable():
    raw_visits = make_raw_visits(
        [
            # Monday 2024-01-15, then nothing else until a summer dispensing,
            # so the drug's 겨울/월요일 bucket spans 7 Mondays (01-15 to
            # 02-26).
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일="2024-01-15", 약품ID=1, 소모량=70.0
            ),
            make_visit_row(
                조제판매ID=2, 고객ID=2, 내방일="2024-07-03", 약품ID=1, 소모량=400.0
            ),
        ]
    )

    default_mart3 = build_marts(
        raw_visits, as_of_date="2024-08-01", rare_drug_patient_threshold=1
    ).mart3
    strict_mart3 = build_marts(
        raw_visits,
        as_of_date="2024-08-01",
        mart3_bucket_min_days=8,
        rare_drug_patient_threshold=1,
    ).mart3

    # Default (5): the bucket's own 7 days suffice.
    assert _daily_value(default_mart3, 1, "겨울", "월요일") == pytest.approx(70.0 / 7)
    # 8: the bucket falls short, so the winter season's daily average
    # (46 days from 01-15 to 02-29, which clears the default season bar).
    assert _daily_value(strict_mart3, 1, "겨울", "월요일") == pytest.approx(70.0 / 46)


def test_mart3_excludes_a_drug_only_dispensed_after_customers_became_chronic():
    # 약품ID=3 is only ever dispensed on customer 1's post-Chronic-since
    # visit, so it gets no Mart 3 row at all.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer_1_visits()
        + [make_visit_row(조제판매ID=3, 고객ID=1, 내방일="2024-02-12", 약품ID=3)]
    )

    mart3 = build_marts(raw_visits, as_of_date="2024-03-01", rare_drug_patient_threshold=1).mart3

    assert list(mart3.columns) == MART3_COLUMNS
    assert 3 not in set(mart3[DRUG_ID_COL])
    assert 1 in set(mart3[DRUG_ID_COL])


def test_mart3_rare_drug_count_includes_patients_from_before_they_became_chronic():
    # Customer 1 is Chronic by as_of_date, but got 약품ID=1 while still
    # Acute -- so together with Acute customer 2 the drug has 2 patients.
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits(_FILLER_OCCURRENCES)
        + _chronic_customer_1_visits()
        + [make_visit_row(조제판매ID=4, 고객ID=2, 내방일="2024-01-20", 약품ID=1)]
    )

    mart3 = build_marts(raw_visits, as_of_date="2024-03-01", rare_drug_patient_threshold=2).mart3

    assert 1 in set(mart3[DRUG_ID_COL])


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


def test_season_and_weekday_for_matches_the_grid_axes():
    # 2024-01-15 is a Monday in January -- 겨울/월요일, matching the same
    # month->season and weekday->name mappings the grid itself is built from.
    assert season_and_weekday_for("2024-01-15") == ("겨울", "월요일")
    assert season_and_weekday_for("2024-07-04") == ("여름", "목요일")


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
