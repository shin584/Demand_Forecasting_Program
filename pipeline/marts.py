"""Mart-construction pipeline: turns raw extracted visit-level rows into
Mart 1 (Visit-Probability), Mart 2 (Customer Drug Profile) and Mart 3
(Acute Drug Statistics).

See CONTEXT.md and docs/adr/ for the design decisions this pipeline encodes.
This module currently implements the pipeline scaffolding, Mart 1's Y label,
Revisit Match-based Chronic/Acute routing (as of each `build_marts` call's
own as_of_date via each customer's Chronic-since Date, per
docs/adr/0005-lapse-aware-track1-population-and-evaluation.md),
Mart 1's sample-weight tiers (wired into both `build_marts`'s single-snapshot
mart1 output and `build_mart1_training_set`'s historical rows),
Mart 1's remaining X-features (demographics, as-of family loyalty,
visit-timing/medication and insurance/차상위 features -- 가족_총매출 is
permanently out of scope, see below -- plus MPR adherence score and
per-patient no-show rate), Mart 2's as-of-date consumption values,
as-of-date family visit totals, Mart 3's drug x season x weekday
aggregation with sparse-bucket backoff (split bucket/season thresholds, and
a rare-drug population filter excluding low-patient drugs entirely -- see
docs/adr/0003-mart3-population-and-backoff-thresholds.md),
`build_mart1_training_set`'s assembly of Mart 1's full historical training
set (Next-Day Visit positives plus early/mid/late negative samples, see that
function) as a separate path from `build_marts`'s own single as_of_date
snapshot, and `split_mart1_training_set`'s train/validation/test temporal
split of that training set.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

CUSTOMER_ID_COL = "고객ID"
VISIT_ID_COL = "조제판매ID"
VISIT_DATE_COL = "내방일"
NEXT_EXPECTED_VISIT_COL = "다음내방일"
PRESCRIPTION_DAYS_COL = "처방조제일수"
FAMILY_ID_COL = "가족ID"
DRUG_ID_COL = "약품ID"
DRUG_NAME_COL = "약품명"
CONSUMPTION_COL = "소모량"
NEXT_DAY_VISIT_COL = "내일_방문"
CHRONIC_COL = "만성질환여부"
FAMILY_VISIT_COUNT_COL = "가족_총내방"
MART2_VALUE_COL = "최근소모량"
SNAPSHOT_DATE_COL = "기준일자"
WEIGHT_COL = "학습_가중치"
CHRONIC_SINCE_COL = "만성_시작일"

# Mart 1 X-features added by issue #7.
BIRTH_DATE_COL = "생년월일"
GENDER_COL = "성별"
AGE_COL = "나이"
DAYS_SINCE_LAST_VISIT_COL = "마지막방문_경과일"
REMAINING_MED_DAYS_COL = "남은_약_일수"
TOMORROW_IS_EXPECTED_VISIT_COL = "내일이_예약일"
LONG_TERM_MED_DAYS_COL = "장기투약_일수"
PRIMARY_INGREDIENT_COL = "주요_약품속명"
INSURANCE_TYPE_COL = "보험구분"
NEAR_POVERTY_COL = "차상위대상자"
MEDICATION_DAYS_COL = "투약일수"
INGREDIENT_COL = "속명"

# Mart 1 X-features added by issue #8: MPR adherence score and per-patient
# no-show rate, both derived purely from a customer's own 조제판매ID history
# (see CONTEXT.md's Mart 1 entry and docs/Research-Log.md's "복약 순응도 지표"
# formula) rather than from any anchoring single visit. Travel together as a
# pair everywhere (the constants below, MPR_NO_SHOW_FEATURE_COLS, and
# MART1_COLUMNS), mirroring how ANCHORING_FEATURE_COLS groups its own
# anchoring-visit features.
MPR_COL = "복약_순응도"
NO_SHOW_RATE_COL = "노쇼_비율"
MPR_NO_SHOW_FEATURE_COLS = [MPR_COL, NO_SHOW_RATE_COL]

# Severity/특례 eligibility flags (see CONTEXT.md "Severe/특례 Weight Tier").
# 차상위대상자 is deliberately excluded here -- it's a copay-assistance
# signal, not a severity signal, and belongs as an X-feature instead (see
# issue #6 and ADR-0001's sibling discussion in CONTEXT.md).
SEVERITY_FLAG_COLS = ["중증암등록대상자", "산전산모대상자", "희귀난치대상자"]

# Internal-only columns, not part of the raw_visits schema.
_DRUG_SET_COL = "_drug_set"
_CLEAN_DRUG_SET_COL = "_clean_drug_set"
_BUCKET_MEAN_COL = "_bucket_mean"
_BUCKET_COUNT_COL = "_bucket_count"
_SEASON_MEAN_COL = "_season_mean"
_SEASON_COUNT_COL = "_season_count"
_DRUG_MEAN_COL = "_drug_mean"
_MATCHED_ON_COL = "_matched_on"

# 가족_총매출 (as-of) is permanently out of scope, not merely deferred: the
# raw extract has no per-visit monetary amount to recompute it from
# point-in-time, and the source-DB extraction work to add one (#10) was
# judged not worth it given 가족_총내방 already covers family loyalty.
# Every other Mart 1 X-feature issue #7 calls for is here.
MART1_COLUMNS = [
    CUSTOMER_ID_COL,
    NEXT_DAY_VISIT_COL,
    CHRONIC_COL,
    AGE_COL,
    GENDER_COL,
    FAMILY_VISIT_COUNT_COL,
    DAYS_SINCE_LAST_VISIT_COL,
    REMAINING_MED_DAYS_COL,
    TOMORROW_IS_EXPECTED_VISIT_COL,
    LONG_TERM_MED_DAYS_COL,
    PRIMARY_INGREDIENT_COL,
    INSURANCE_TYPE_COL,
    NEAR_POVERTY_COL,
    *MPR_NO_SHOW_FEATURE_COLS,
    WEIGHT_COL,
]
# MART1_COLUMNS entries that are an identifier, the Y label, the sample
# weight, or the Chronic-only-population constant -- never a model X-feature.
# Shared by `build_mart1_training_set` (below) and `pipeline.model.FEATURE_COLS`.
MART1_NON_FEATURE_COLS = (CUSTOMER_ID_COL, NEXT_DAY_VISIT_COL, CHRONIC_COL, WEIGHT_COL)

# Mart 1 X-features `_mart1_x_features` pins to float64 / nullable `boolean`
# dtype -- LightGBM rejects object-dtype numeric columns at fit and predict
# time. 주요_약품속명/보험구분/성별 are deliberately left out:
# `pipeline.model.CATEGORICAL_FEATURE_COLS` casts those to pandas `category`
# dtype instead.
MART1_NUMERIC_FEATURE_COLS = [
    AGE_COL,
    FAMILY_VISIT_COUNT_COL,
    DAYS_SINCE_LAST_VISIT_COL,
    REMAINING_MED_DAYS_COL,
    LONG_TERM_MED_DAYS_COL,
    *MPR_NO_SHOW_FEATURE_COLS,
]
MART1_BOOLEAN_FEATURE_COLS = [TOMORROW_IS_EXPECTED_VISIT_COL, NEAR_POVERTY_COL]

MART2_COLUMNS = [CUSTOMER_ID_COL, DRUG_ID_COL, MART2_VALUE_COL]
SEASON_COL = "계절"
WEEKDAY_COL = "요일"
MART3_COLUMNS = [DRUG_ID_COL, SEASON_COL, WEEKDAY_COL, CONSUMPTION_COL]

# Mart 3's fixed bucket axes (see CONTEXT.md "Mart 3 (Acute Drug Statistics
# Mart)" and "Track 2 Sparse-Bucket Backoff"): standard meteorological
# seasons, and Korean weekday names (matching dataset/D's weekday-named
# folders). Every drug present in Mart 3's population gets a row for every
# one of these 4x7 = 28 combinations, regardless of which buckets that drug
# actually has observations in -- a drug with zero winter sales still gets a
# 겨울 row, backed off rather than omitted, since CONTEXT.md's ~72%-sparse
# figure describes the typical case, not an edge case to skip.
MART3_SEASONS = ["봄", "여름", "가을", "겨울"]
MART3_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

_MONTH_TO_SEASON = {
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
    12: "겨울", 1: "겨울", 2: "겨울",
}
_WEEKDAY_INDEX_TO_NAME = dict(enumerate(MART3_WEEKDAYS))

# Mart 3's sparse-bucket backoff thresholds (see CONTEXT.md "Track 2
# Sparse-Bucket Backoff" and docs/adr/0003-mart3-population-and-backoff-thresholds.md):
# separate bucket-level and season-level sufficiency bars, since reusing one
# shared value made the season tier nearly vestigial (a season pools up to 7
# weekdays of bucket data, so it clears any bucket-sized bar almost
# automatically). Placeholders pending empirical tuning, not fixed business
# requirements (see CONTEXT.md "Decision Thresholds (provisional)").
MART3_BUCKET_MIN_OBSERVATIONS = 5
MART3_SEASON_MIN_OBSERVATIONS = 15

# Mart 3's rare-drug population filter (see docs/adr/0003-...): a drug with
# fewer than this many distinct patients (trailing 12 months, as of the
# snapshot date) gets no Mart 3 row at all -- routed to the rare-drug
# 100%-allocation rule instead of a shaky statistical estimate. Reuses
# CONTEXT.md's existing rare-drug special-handling cutoff ("Decision
# Thresholds (provisional)") rather than maintaining a second, independently-
# tunable "rare" definition.
RARE_DRUG_PATIENT_THRESHOLD = 5

# The Lapse Horizon (see CONTEXT.md "Lapsed Chronic Patient" and
# docs/adr/0005-lapse-aware-track1-population-and-evaluation.md): a Chronic
# customer whose latest visit is more than this many days before a snapshot
# date is Lapsed as of it -- still Chronic (so still out of Mart 3), but
# outside Track 1's population, both in `build_marts`'s Mart 1 snapshot and
# in every Mart 1 Training Set row. Provisional, like the other Decision
# Thresholds.
LAPSE_HORIZON_DAYS = 180

# Mart 1's X-features anchored to each customer's most recent visit at or
# before as_of_date (see _anchoring_visit_features) -- shared by that
# function's output columns and _attach_mart1_x_features' join loop.
ANCHORING_FEATURE_COLS = [
    DAYS_SINCE_LAST_VISIT_COL,
    REMAINING_MED_DAYS_COL,
    TOMORROW_IS_EXPECTED_VISIT_COL,
    LONG_TERM_MED_DAYS_COL,
    PRIMARY_INGREDIENT_COL,
    INSURANCE_TYPE_COL,
    NEAR_POVERTY_COL,
]

# Revisit Match criterion H+T2 (see docs/adr/0001-chronic-patient-behavioral-definition.md).
REVISIT_MATCH_WINDOW_DAYS = 30
TOP_FREQUENT_DRUG_EXCLUDE_COUNT = 2

# Mart 1's 학습_가중치 sample-weight tiers (see CONTEXT.md "Severe/특례 Weight
# Tier"): severity takes priority over chronic, which takes priority over the
# baseline. Placeholder values pending empirical tuning against
# validation-set precision/recall -- not fixed business requirements (see
# CONTEXT.md "Decision Thresholds (provisional)").
SEVERITY_TIER_WEIGHT = 3.0
CHRONIC_TIER_WEIGHT = 2.0
BASELINE_WEIGHT = 1.0

# Mart 1's Y=0 negative-sampling scheme (see CONTEXT.md "Negative Sampling
# Windows" and docs/Research-Log.md's absolute-day 초기(1~5일차)/중기(15일차)/
# 말기(27~29일차) scheme this generalizes into proportions of each patient's
# own 처방조제일수 cycle length): ~15% early, ~50% mid, ~90-96% late (up to 3
# samples). 1 early + 1 mid + 3 late, against 1 assumed positive per cycle,
# is what produces the accepted ~1:5 ratio -- not the superseded "1:3" figure.
NEGATIVE_SAMPLE_FRACTIONS = (0.15, 0.50, 0.90, 0.93, 0.96)

MART1_NEGATIVE_SAMPLE_COLUMNS = [
    CUSTOMER_ID_COL,
    VISIT_ID_COL,
    SNAPSHOT_DATE_COL,
    NEXT_DAY_VISIT_COL,
]

# build_mart1_training_set's output columns: MART1_COLUMNS (the same X/Y/weight
# shape build_marts's single-snapshot mart1 uses) plus 기준일자, since a
# historical training row's own snapshot date -- unlike build_marts's single
# as_of_date, already known to that call's caller -- has to travel with the
# row itself (needed for the train/validation/test temporal split, see
# docs/Plan.md "시계열 분할").
MART1_TRAINING_COLUMNS = [SNAPSHOT_DATE_COL, *MART1_COLUMNS]

# split_mart1_training_set's window sizes (see docs/Plan.md "시계열 분할").
# Plan.md's original 2yr/6mo/6mo split assumed ~3 years of history; only the
# two eval windows are pinned to a fixed size here -- train absorbs whatever
# span is left before val_months + test_months, computed from the training
# set's own max 기준일자, never a hardcoded calendar date (so the split stays
# correct as more history accumulates in future extracts).
TEST_WINDOW_MONTHS = 6
VAL_WINDOW_MONTHS = 6


class MartResult(NamedTuple):
    mart1: pd.DataFrame
    mart2: pd.DataFrame
    mart3: pd.DataFrame


class Mart1Split(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def build_marts(
    raw_visits: pd.DataFrame,
    as_of_date,
    mart3_bucket_min_observations: int = MART3_BUCKET_MIN_OBSERVATIONS,
    mart3_season_min_observations: int = MART3_SEASON_MIN_OBSERVATIONS,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
) -> MartResult:
    """Build Mart 1/2/3 from raw visit-level rows, as of a given snapshot date.

    Pure and deterministic: reads only `as_of_date` (never wall-clock "today")
    and never mutates `raw_visits`, so the same inputs always produce the same
    outputs and the same entry point can be reused for backtesting and live
    inference, one as_of_date snapshot at a time (see
    docs/adr/0002-point-in-time-correctness.md). For Mart 1's full historical
    training set instead -- many rows per customer, across many past
    snapshot dates -- see `build_mart1_training_set`.

    A customer is Chronic as of `as_of_date` exactly when their Chronic-since
    Date (see `chronic_since_dates`) is <= `as_of_date` -- a match first
    observable after `as_of_date` can't make an earlier snapshot's
    classification "Chronic" (see docs/adr/0002-point-in-time-correctness.md
    and docs/adr/0005-lapse-aware-track1-population-and-evaluation.md).
    Mart 1 inclusion, 만성질환여부, and Mart 3's population routing all
    derive from this one as-of-correct result. The top-2 high-frequency drug
    exclusion is taken from all of `raw_visits` (see ADR-0005, "Update
    (implementing #26)").

    `mart3_bucket_min_observations`/`mart3_season_min_observations` (defaults
    `MART3_BUCKET_MIN_OBSERVATIONS`/`MART3_SEASON_MIN_OBSERVATIONS`) are Mart
    3's sparse-bucket backoff thresholds (see `resolve_mart3_backoff`).
    `rare_drug_patient_threshold` (default `RARE_DRUG_PATIENT_THRESHOLD`)
    excludes drugs with too few patients from Mart 3 entirely (see
    `_build_mart3`). `lapse_horizon_days` (default `LAPSE_HORIZON_DAYS`)
    drops Lapsed Chronic customers -- latest visit more than that many days
    before `as_of_date` -- from Mart 1 only; they stay Chronic, so Mart 3
    still excludes them.
    """
    as_of_date = pd.Timestamp(as_of_date)
    chronic_customer_ids = _chronic_customer_ids_as_of(chronic_since_dates(raw_visits), as_of_date)
    mart1 = _build_mart1(raw_visits, as_of_date, chronic_customer_ids, lapse_horizon_days)
    mart2 = _build_mart2(raw_visits, as_of_date)
    mart3 = _build_mart3(
        raw_visits,
        as_of_date,
        chronic_customer_ids,
        mart3_bucket_min_observations,
        mart3_season_min_observations,
        rare_drug_patient_threshold,
    )
    return MartResult(mart1=mart1, mart2=mart2, mart3=mart3)


def build_mart1_training_set(
    raw_visits: pd.DataFrame, lapse_horizon_days: int = LAPSE_HORIZON_DAYS
) -> pd.DataFrame:
    """Mart 1's full historical training set: one row per Next-Day Visit
    positive (see `_mart1_positive_samples`) plus one row per
    `sample_mart1_negatives` Y=0 sample, each with X-features and
    학습_가중치 attached as of that row's own snapshot date -- unlike
    `build_marts`, which builds a single as_of_date snapshot meant for daily
    inference/backtesting one day at a time.

    Chronic membership is as-of-correct per row, the same rule `build_marts`
    applies: a row, positive or negative, is kept only if its snapshot date
    is on or after its customer's Chronic-since Date (see
    `chronic_since_dates` and
    docs/adr/0005-lapse-aware-track1-population-and-evaluation.md, which
    supersedes the full-history shortcut docs/adr/0002 once allowed here),
    and only if its customer isn't Lapsed as of that date -- their latest
    visit on or before it is at most `lapse_horizon_days` earlier (see
    `_is_within_lapse_horizon`), the same population `build_marts` scores.
    학습_가중치 tiers are still derived once from the full, unfiltered
    `raw_visits`: the weight only scales a row's loss and is never a model
    input, so it can't leak into predictions. The point, anchoring,
    family-total, and MPR/no-show X-features are computed as of each row's
    own snapshot date, by the same `_mart1_x_features` `build_marts` uses, in
    one vectorized pass over every row rather than once per distinct
    snapshot date.

    Returns an empty `MART1_TRAINING_COLUMNS`-shaped frame if `raw_visits`
    has no Chronic Patients at all, or if no sampled row is both on or after
    its customer's Chronic-since Date and within the Lapse Horizon.
    """
    chronic_since = chronic_since_dates(raw_visits)
    chronic_customer_ids = set(chronic_since.index)
    if not chronic_customer_ids:
        return pd.DataFrame(columns=MART1_TRAINING_COLUMNS)

    samples = pd.concat(
        [
            _mart1_positive_samples(raw_visits, chronic_customer_ids),
            sample_mart1_negatives(raw_visits, chronic_customer_ids),
        ],
        ignore_index=True,
    )
    is_chronic_as_of_row = (
        samples[SNAPSHOT_DATE_COL] >= samples[CUSTOMER_ID_COL].map(chronic_since)
    )
    samples = samples[is_chronic_as_of_row].reset_index(drop=True)

    result = samples.join(
        _mart1_x_features(raw_visits, samples[CUSTOMER_ID_COL], samples[SNAPSHOT_DATE_COL])
    )
    result = result[_is_within_lapse_horizon(result, lapse_horizon_days)].reset_index(drop=True)
    # Derived solely from Revisit Match, same as _build_mart1 -- every row
    # here is already restricted to chronic_customer_ids.
    result[CHRONIC_COL] = True
    result = _attach_mart1_weight(result, raw_visits, chronic_customer_ids)
    return result[MART1_TRAINING_COLUMNS]


def split_mart1_training_set(
    training_set: pd.DataFrame,
    test_months: int = TEST_WINDOW_MONTHS,
    val_months: int = VAL_WINDOW_MONTHS,
) -> Mart1Split:
    """Splits `build_mart1_training_set`'s output into train/val/test by
    기준일자 (see docs/Plan.md "시계열 분할"): the most recent `test_months`
    become test, the `val_months` immediately before that become validation,
    and everything earlier becomes train.

    Both window boundaries are computed from `training_set`'s own max 기준일자
    -- never a hardcoded calendar date -- so test and val always land on the
    same two fixed-size, most-recent windows regardless of how much history
    `training_set` covers; train simply absorbs whatever's left before them.
    This is "cut from the end, unrounded": the two eval windows are protected
    at their planned size, and train isn't padded or trimmed to hit a target
    proportion.

    Each boundary is inclusive on its more-recent side: a row dated exactly
    on a cutoff belongs to the newer of the two windows it separates.
    """
    snapshot_dates = training_set[SNAPSHOT_DATE_COL]
    test_start = snapshot_dates.max() - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)

    train = training_set[snapshot_dates < val_start]
    val = training_set[(snapshot_dates >= val_start) & (snapshot_dates < test_start)]
    test = training_set[snapshot_dates >= test_start]
    return Mart1Split(train=train, val=val, test=test)


def revisit_match(raw_visits: pd.DataFrame) -> pd.Series:
    """The Revisit Match matcher (criterion H+T2, see ADR-0001).

    Takes `raw_visits` as-is and has no `as_of_date` parameter: as-of-date
    filtering is the caller's responsibility (pre-filter `raw_visits` to
    visits with 내방일 <= as_of_date before calling), not something this
    function does itself (see docs/adr/0002-point-in-time-correctness.md,
    "Update (implementing #11)"). `build_marts` instead gets its as-of
    Chronic population from `chronic_since_dates`.

    Returns a boolean Series indexed by 조제판매ID: True if some later visit
    by the same customer shares >=1 drug with this visit - excluding the
    dataset's top-2 highest-frequency drugs, to avoid false matches on
    near-universal OTC drugs - within +/-30 days of this visit's 다음내방일
    (expected next-visit date).
    """
    visit_ids = _one_row_per_visit(raw_visits, [CUSTOMER_ID_COL]).index
    matched_visit_ids = _matched_drug_rows(raw_visits)[VISIT_ID_COL]
    is_match = pd.Series(visit_ids.isin(matched_visit_ids), index=visit_ids)
    is_match.index.name = VISIT_ID_COL
    return is_match


def _matched_drug_rows(raw_visits: pd.DataFrame) -> pd.DataFrame:
    """The distinct (조제판매ID, 약품ID) rows that make their visit a Revisit
    Match (see `revisit_match`), with each visit's 고객ID alongside and, as
    `_MATCHED_ON_COL`, the 내방일 of the earliest later visit that matches it.

    Vectorized rather than comparing every pair of a customer's visits: for
    each (visit, drug) row, the only later visit worth checking is the
    earliest one by the same customer dispensing the same drug on or after
    max(내방일 + 1 day, 다음내방일 - 30 days) -- if that one isn't within
    다음내방일 + 30 days, no later one is either. Found with one sorted
    as-of lookup per bound (see `_as_of_positions`).
    """
    visits = _one_row_per_visit(
        raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL, NEXT_EXPECTED_VISIT_COL]
    )
    # A shared drug only counts if it's outside the top-N most frequent -- the
    # same drug sits on both sides of a match, so dropping those rows up front
    # removes them from both.
    excluded_drug_ids = _top_frequent_drug_ids(raw_visits, TOP_FREQUENT_DRUG_EXCLUDE_COUNT)
    drug_rows = raw_visits[[VISIT_ID_COL, DRUG_ID_COL]].dropna().drop_duplicates()
    drug_rows = (
        drug_rows[~drug_rows[DRUG_ID_COL].isin(excluded_drug_ids)]
        .join(visits, on=VISIT_ID_COL)
        .reset_index(drop=True)
    )
    visit_dates = pd.to_datetime(drug_rows[VISIT_DATE_COL])
    next_dates = pd.to_datetime(drug_rows[NEXT_EXPECTED_VISIT_COL])
    window = pd.Timedelta(days=REVISIT_MATCH_WINDOW_DAYS)

    candidates = drug_rows.assign(**{VISIT_DATE_COL: visit_dates})
    candidates = candidates[visit_dates.notna()].sort_values(
        [CUSTOMER_ID_COL, DRUG_ID_COL, VISIT_DATE_COL], kind="stable"
    )
    candidate_groups = pd.MultiIndex.from_frame(candidates[[CUSTOMER_ID_COL, DRUG_ID_COL]])
    query_groups = pd.MultiIndex.from_frame(drug_rows[[CUSTOMER_ID_COL, DRUG_ID_COL]])
    candidate_dates = candidates[VISIT_DATE_COL]

    def positions(query_dates, side):
        return _as_of_positions(
            candidate_groups, candidate_dates, query_groups, query_dates, side
        )[1]

    # First candidate strictly later than this visit, first candidate inside
    # the window's lower bound, and one past the last inside its upper bound.
    after_visit = positions(visit_dates, "right")
    in_window_from = positions(next_dates - window, "left")
    in_window_to = positions(next_dates + window, "right")
    first_candidate = np.maximum(after_visit, in_window_from)

    matched = (
        visit_dates.notna().to_numpy()
        & next_dates.notna().to_numpy()
        & (first_candidate < in_window_to)
    )
    result = drug_rows.loc[matched, [VISIT_ID_COL, DRUG_ID_COL, CUSTOMER_ID_COL]]
    # The earliest in-window revisit is the day this match first becomes
    # observable (see `chronic_since_dates`).
    result[_MATCHED_ON_COL] = candidate_dates.to_numpy()[first_candidate[matched]]
    return result


def _as_of_positions(
    table_groups: pd.Index,
    table_times: pd.Series,
    query_groups: pd.Index,
    query_times,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-group as-of lookup, the shared engine behind every
    "what's knowable as of this snapshot date" feature (see
    docs/adr/0002-point-in-time-correctness.md) when many (group, date)
    queries must be answered at once.

    `table_groups`/`table_times` must already be sorted by (group, time),
    with no NaT times. For each query (group, time), returns (start, end)
    row positions into that table such that rows start..end-1 are exactly
    that group's rows with time <= the query time (`side="right"`) or
    strictly < it (`side="left"`). A group absent from the table, or a
    NaT query time, gets an empty range (start == end).
    """
    group_index = table_groups.unique()
    table_codes = group_index.get_indexer(table_groups).astype(np.int64)
    query_codes = group_index.get_indexer(query_groups).astype(np.int64)

    table_ns = pd.to_datetime(pd.Series(table_times)).to_numpy("datetime64[ns]").view(np.int64)
    query_datetimes = pd.to_datetime(pd.Series(query_times))
    query_ns = query_datetimes.to_numpy("datetime64[ns]").view(np.int64)
    valid = (query_codes >= 0) & query_datetimes.notna().to_numpy()

    # Rank-compress every time involved so (group, time) packs into one
    # sortable int64 key without any assumption about time resolution.
    all_times = np.unique(np.concatenate([table_ns, query_ns[valid]]))
    width = len(all_times) + 1
    table_keys = table_codes * width + np.searchsorted(all_times, table_ns)
    query_base = np.where(valid, query_codes, 0) * width
    query_keys = query_base + np.searchsorted(all_times, query_ns)

    start = np.searchsorted(table_keys, query_base, side="left")
    end = np.searchsorted(table_keys, query_keys, side=side)
    return np.where(valid, start, 0), np.where(valid, end, 0)


def _visits_at_or_before(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """`raw_visits` restricted to rows with 내방일 <= `as_of_date` -- what's
    actually knowable as of that snapshot. Shared by every as-of-date
    computation that must not see visits from after `as_of_date` (Mart 2's
    latest-consumption lookup, Mart 3's observation population, Mart 1's
    severity tier), per docs/adr/0002-point-in-time-correctness.md.
    """
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    return raw_visits.loc[visit_dates <= as_of_date]


def _one_row_per_visit(
    raw_visits: pd.DataFrame, cols: list, group_col: str = VISIT_ID_COL
) -> pd.DataFrame:
    """One row per `group_col` value (default 조제판매ID -- a visit may
    dispense several drugs, one raw row each), keeping the first value of
    each of `cols`. Pass `group_col=고객ID` for attributes that are constant
    per customer instead (e.g. 성별/생년월일/가족ID)."""
    return raw_visits.groupby(group_col, sort=False).agg(
        **{col: (col, "first") for col in cols}
    )


def _top_frequent_drug_ids(raw_visits: pd.DataFrame, top_n: int) -> set:
    # "Frequency" is the number of distinct visits a drug appears in, matching
    # the 등장조제판매ID수 definition test/Sensitivity analysis.py established
    # for the 고빈도 (high-frequency) drug set this criterion excludes.
    visit_counts = (
        raw_visits.dropna(subset=[DRUG_ID_COL])
        .groupby(DRUG_ID_COL)[VISIT_ID_COL]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    return set(visit_counts.head(top_n).index)


def chronic_since_dates(raw_visits: pd.DataFrame) -> pd.Series:
    """Each Chronic Patient's Chronic-since Date (see CONTEXT.md and
    docs/adr/0005-lapse-aware-track1-population-and-evaluation.md): the
    내방일 of the later visit in their first Revisit-Matched pair -- the
    earliest day that match could actually have been observed. A customer is
    Chronic as of *d* exactly when this date is <= *d*.

    Returns a datetime Series indexed by 고객ID (named `CHRONIC_SINCE_COL`),
    one entry per customer with >=1 Revisit Match anywhere in `raw_visits`;
    customers who never Revisit-Match have no entry. Pass the full extract:
    the top-2 high-frequency drug exclusion is computed from whatever
    `raw_visits` holds, and a per-date population then comes from comparing
    against *d*, not from re-running Revisit Match on as-of-filtered visits.
    """
    if raw_visits.empty:
        return pd.Series(
            pd.to_datetime([]), index=pd.Index([], name=CUSTOMER_ID_COL), name=CHRONIC_SINCE_COL
        )
    matched = _matched_drug_rows(raw_visits)
    return matched.groupby(CUSTOMER_ID_COL)[_MATCHED_ON_COL].min().rename(CHRONIC_SINCE_COL)


def _chronic_customer_ids_as_of(chronic_since: pd.Series, as_of_date: pd.Timestamp) -> set:
    """The Chronic population as of `as_of_date`, from `chronic_since_dates`."""
    return set(chronic_since.index[chronic_since <= as_of_date])


def _chronic_customer_ids(raw_visits: pd.DataFrame) -> set:
    """Customers with >=1 Revisit Match visit - Chronic Patients (see ADR-0001).

    Classifies from exactly the visits in `raw_visits` - it does no as-of-date
    filtering itself. Called with the full, unfiltered table (e.g.
    `sample_mart1_weights`, `sample_mart1_negatives`), where the current
    live-snapshot classification is exactly what's wanted; for the
    population as of an earlier date, see `chronic_since_dates`.
    """
    return set(chronic_since_dates(raw_visits).index)


def sample_mart1_weights(
    raw_visits: pd.DataFrame,
    chronic_customer_ids: set | None = None,
    severity_weight: float = SEVERITY_TIER_WEIGHT,
    chronic_weight: float = CHRONIC_TIER_WEIGHT,
    baseline_weight: float = BASELINE_WEIGHT,
) -> pd.Series:
    """Mart 1's 학습_가중치 (sample weight), per 고객ID (see CONTEXT.md
    "Severe/특례 Weight Tier").

    The severity tier (`severity_weight`, default 3.0) applies to any
    customer with >=1 visit where 중증암등록대상자, 산전산모대상자 or
    희귀난치대상자 is true, regardless of chronic status. Otherwise the
    chronic tier (`chronic_weight`, default 2.0) applies to Chronic Patients
    (per Revisit Match, see ADR-0001). Otherwise the baseline
    (`baseline_weight`, default 1.0) applies. 차상위대상자 never affects this
    weight -- it's a copay-assistance signal, not a severity signal.

    `chronic_customer_ids` can be passed in to reuse a result already
    computed by `build_marts`; otherwise it's derived here via
    `revisit_match`.

    Wired into Mart 1's 학습_가중치 column via `_attach_mart1_weight`, used by
    both `build_marts`'s single-snapshot mart1 output (given as-of-filtered
    `raw_visits`, so severity status is as-of-correct, and its as-of
    Chronic population) and `build_mart1_training_set`'s historical rows (given the full,
    unfiltered `raw_visits` for the weight tiers only -- its rows themselves
    are already filtered by Chronic-since Date; see that function's
    docstring for why).
    """
    if chronic_customer_ids is None:
        chronic_customer_ids = _chronic_customer_ids(raw_visits)
    severity_customer_ids = _severity_customer_ids(raw_visits)

    customers = _distinct_customer_ids(raw_visits)
    weights = pd.Series(
        baseline_weight, index=customers.index, dtype=float, name=WEIGHT_COL
    )
    # Severity assigned after chronic so it wins where a customer is both.
    weights[customers.isin(chronic_customer_ids)] = chronic_weight
    weights[customers.isin(severity_customer_ids)] = severity_weight
    weights.index = pd.Index(customers, name=CUSTOMER_ID_COL)
    return weights


def _distinct_customer_ids(raw_visits: pd.DataFrame, mask=None) -> pd.Series:
    """Distinct 고객ID values (optionally restricted to `mask`), sorted
    ascending with a fresh 0..n-1 index -- the customer-listing shape shared
    by Mart 1's population and its sample-weight lookup."""
    customer_ids = raw_visits[CUSTOMER_ID_COL]
    if mask is not None:
        customer_ids = customer_ids[mask]
    return customer_ids.drop_duplicates().sort_values().reset_index(drop=True)


def _severity_customer_ids(raw_visits: pd.DataFrame) -> set:
    """Customers with >=1 visit where 중증암등록대상자, 산전산모대상자 or
    희귀난치대상자 is true -- the severity sample-weight tier (see
    CONTEXT.md "Severe/특례 Weight Tier")."""
    if raw_visits.empty:
        return set()
    mask = pd.Series(False, index=raw_visits.index)
    for col in SEVERITY_FLAG_COLS:
        mask = mask | _eligibility_flag_mask(raw_visits[col])
    return set(raw_visits.loc[mask, CUSTOMER_ID_COL])


def _eligibility_flag_mask(column: pd.Series) -> pd.Series:
    """True where a raw 대상자 eligibility flag column value represents
    'yes'.

    Source columns store 'Y' (case-insensitive) for membership and
    None/NaN/blank otherwise (see docs/Research-Log.md "학습 가중치").
    """
    return column.astype("string").str.strip().str.upper().eq("Y").fillna(False)


def negative_sample_offsets(prescription_days: float) -> list[int]:
    """Day-offsets (relative to an anchoring visit) for Mart 1's Y=0
    negative-sampling scheme (see CONTEXT.md "Negative Sampling Windows").

    Offsets are proportions of `prescription_days` - that visit's own
    처방조제일수 cycle length - rather than fixed absolute days, so sampling
    stays meaningful whether the observed cycle is 7 days or 60+. Every
    offset is >=1 day so no negative sample can land on or before the
    anchoring visit itself; offsets a short cycle rounds onto the same day
    are collapsed, so a very short cycle yields fewer than 5 samples.
    """
    offsets = {
        max(1, round(prescription_days * fraction))
        for fraction in NEGATIVE_SAMPLE_FRACTIONS
    }
    return sorted(offsets)


def negative_sample_dates(visit_date, prescription_days: float) -> list[pd.Timestamp]:
    """Candidate Mart 1 Y=0 snapshot dates for one anchoring visit, via the
    early/mid/late window scheme (see negative_sample_offsets)."""
    visit_date = pd.Timestamp(visit_date)
    return [
        visit_date + pd.Timedelta(days=offset)
        for offset in negative_sample_offsets(prescription_days)
    ]


def sample_mart1_negatives(
    raw_visits: pd.DataFrame, chronic_customer_ids: set | None = None
) -> pd.DataFrame:
    """Mart 1's Y=0 negative-sample rows, via the early/mid/late window
    scheme, for every Chronic Patient (see ADR-0001) patient-cycle - a visit
    that has a later visit by the same customer, bounding the cycle the
    offsets are sampled within.

    One row per (anchoring visit, sampled offset), dated that visit's own
    내방일 plus the offset. A customer's chronologically last known visit
    never anchors a cycle - it has no later visit, so there's nothing to
    bound the sampling window with, and it's what supplies the "1 assumed
    positive per cycle" that the accepted ~1:5 ratio is measured against.
    `chronic_customer_ids` can be passed in to reuse a result already
    computed by `build_marts`; otherwise it's derived here via
    `revisit_match`.

    Wired into `build_mart1_training_set`, which assembles these Y=0 rows
    alongside `_mart1_positive_samples`'s Next-Day Visit positives into
    Mart 1's full historical training set (see that function) -- kept
    separate from `build_marts`'s own mart1 output, which stays a single
    as_of_date snapshot for daily inference/backtesting.
    """
    if chronic_customer_ids is None:
        chronic_customer_ids = _chronic_customer_ids(raw_visits)
    if not chronic_customer_ids:
        return pd.DataFrame(columns=MART1_NEGATIVE_SAMPLE_COLUMNS)

    visits = _one_row_per_visit(
        raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL, PRESCRIPTION_DAYS_COL]
    ).reset_index()
    visits = visits[visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]

    # Customers in first-appearance order, each customer's visits
    # chronological (stable, so same-day visits keep their relative order),
    # minus that customer's last visit.
    visits = visits.assign(_customer_order=pd.factorize(visits[CUSTOMER_ID_COL])[0])
    visits = visits.sort_values(["_customer_order", VISIT_DATE_COL], kind="stable")
    anchors = visits[visits.duplicated(CUSTOMER_ID_COL, keep="last")].reset_index(drop=True)

    # negative_sample_offsets, vectorized: np.rint rounds half-to-even exactly
    # like the built-in round() it uses.
    offsets = np.maximum(
        1,
        np.rint(
            np.outer(
                anchors[PRESCRIPTION_DAYS_COL].to_numpy(dtype=float),
                NEGATIVE_SAMPLE_FRACTIONS,
            )
        ),
    )
    samples = (
        pd.DataFrame(offsets)
        .stack()
        .rename("_offset")
        .reset_index(level=1, drop=True)
        .rename_axis("_anchor")
        .reset_index()
        .drop_duplicates()
        .sort_values(["_anchor", "_offset"], kind="stable")
    )
    anchor_rows = anchors.loc[samples["_anchor"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: anchor_rows[CUSTOMER_ID_COL],
            VISIT_ID_COL: anchor_rows[VISIT_ID_COL],
            SNAPSHOT_DATE_COL: pd.to_datetime(anchor_rows[VISIT_DATE_COL])
            + pd.to_timedelta(samples["_offset"].to_numpy(), unit="D"),
            NEXT_DAY_VISIT_COL: False,
        },
        columns=MART1_NEGATIVE_SAMPLE_COLUMNS,
    )


def _mart1_positive_samples(raw_visits: pd.DataFrame, chronic_customer_ids: set) -> pd.DataFrame:
    """One Next-Day Visit positive row per actual visit by a Chronic Patient
    (see ADR-0001): snapshot_date = that visit's own 내방일 minus one day,
    내일_방문 = True -- the literal observed instance of the customer walking
    in on the calendar day right after the snapshot (see CONTEXT.md's Next-Day
    Visit definition). Every visit contributes one positive this way,
    including a customer's chronologically last visit -- unlike
    `sample_mart1_negatives`, there's no cycle a positive needs a later visit
    to bound.

    Paired with `sample_mart1_negatives`'s Y=0 rows by
    `build_mart1_training_set` to assemble Mart 1's full historical training
    set. Same output shape (`MART1_NEGATIVE_SAMPLE_COLUMNS`) as
    `sample_mart1_negatives` so the two concatenate directly.
    """
    visits = _one_row_per_visit(raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL])
    visits = visits[visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]
    visit_dates = pd.to_datetime(visits[VISIT_DATE_COL])
    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: visits[CUSTOMER_ID_COL].to_numpy(),
            VISIT_ID_COL: visits.index.to_numpy(),
            SNAPSHOT_DATE_COL: (visit_dates - pd.Timedelta(days=1)).to_numpy(),
            NEXT_DAY_VISIT_COL: True,
        },
        columns=MART1_NEGATIVE_SAMPLE_COLUMNS,
    )


def _build_mart1(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_customer_ids: set,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
) -> pd.DataFrame:
    # Only Chronic Patients (per Revisit Match) belong to Mart 1; Acute
    # patients' visits are excluded here and become part of Mart 3's
    # population instead.
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    target_date = as_of_date + pd.Timedelta(days=1)

    chronic_mask = raw_visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)
    customers = _distinct_customer_ids(raw_visits, mask=chronic_mask)

    # 내일_방문 is the literal "did the customer actually show up on
    # as_of_date + 1" boolean — independent of Revisit Match, which is a
    # separate behavioral definition used only for chronic/acute routing.
    # A visit dated exactly as_of_date + 1 day is necessarily that customer's
    # *next* visit after as_of_date, since no calendar day falls between the
    # two — this equivalence would not hold if target_date were more than one
    # day out.
    visited_next_day = set(raw_visits.loc[visit_dates == target_date, CUSTOMER_ID_COL])

    mart1 = pd.DataFrame(
        {
            CUSTOMER_ID_COL: customers,
            NEXT_DAY_VISIT_COL: customers.isin(visited_next_day),
            # Derived solely from Revisit Match — no diagnosis-code or
            # duration-based path exists (redundant with the inclusion
            # filter above, kept explicit as the feature's own definition).
            CHRONIC_COL: customers.isin(chronic_customer_ids),
        }
    )
    mart1 = mart1.join(
        _mart1_x_features(raw_visits, customers, pd.Series(as_of_date, index=customers.index))
    )
    mart1 = mart1[_is_within_lapse_horizon(mart1, lapse_horizon_days)].reset_index(drop=True)
    mart1 = _attach_mart1_weight(
        mart1, _visits_at_or_before(raw_visits, as_of_date), chronic_customer_ids
    )
    return mart1[MART1_COLUMNS]


def _is_within_lapse_horizon(mart1_rows: pd.DataFrame, lapse_horizon_days: int) -> pd.Series:
    """True for each Mart 1 row whose customer isn't Lapsed as of the row's
    snapshot date: their Anchoring Visit is at most `lapse_horizon_days`
    before it. Reads 마지막방문_경과일, which is exactly the days since the
    Anchoring Visit (see `_anchoring_visit_features`); a row with no
    Anchoring Visit has no recent visit either, so it's out too."""
    return mart1_rows[DAYS_SINCE_LAST_VISIT_COL] <= lapse_horizon_days


def _mart1_x_features(
    raw_visits: pd.DataFrame, customer_ids: pd.Series, snapshot_dates: pd.Series
) -> pd.DataFrame:
    """Mart 1's X-features (see issue #7/#8) for each (고객ID, snapshot date)
    pair -- `customer_ids` and `snapshot_dates` are aligned Series, and the
    result shares their index. Every feature is point-in-time correct as of
    its own row's snapshot date (see docs/adr/0002-point-in-time-correctness.md):
    demographics, as-of family loyalty, visit-timing/medication and
    insurance/차상위 features anchored to the customer's most recent visit at
    or before that date, and the MPR/no-show history aggregates.

    One vectorized pass regardless of how many distinct snapshot dates the
    rows span, so `build_marts` (one date) and `build_mart1_training_set`
    (thousands) share the exact same feature definitions. Numeric features
    come back float64 and boolean ones as nullable `boolean` -- the dtypes
    LightGBM accepts (see MART1_NUMERIC_FEATURE_COLS/MART1_BOOLEAN_FEATURE_COLS).
    """
    snapshot_dates = pd.to_datetime(snapshot_dates)
    visits = _visit_table(raw_visits)
    features = pd.DataFrame(index=customer_ids.index)

    customers = _one_row_per_visit(
        raw_visits, [GENDER_COL, BIRTH_DATE_COL, FAMILY_ID_COL], group_col=CUSTOMER_ID_COL
    )
    features[GENDER_COL] = customer_ids.map(customers[GENDER_COL])
    features[AGE_COL] = _age_in_years(
        pd.to_datetime(customer_ids.map(customers[BIRTH_DATE_COL])), snapshot_dates
    )
    features[FAMILY_VISIT_COUNT_COL] = _family_visit_counts_as_of(
        visits, customer_ids.map(customers[FAMILY_ID_COL]), snapshot_dates
    )

    # Each customer's visits in chronological order (stable, so same-day
    # visits keep their original relative order); rows start..end-1 of this
    # table are each query row's visits at or before its snapshot date.
    history = (
        visits[visits[VISIT_DATE_COL].notna()]
        .sort_values([CUSTOMER_ID_COL, VISIT_DATE_COL], kind="stable")
        .reset_index(drop=True)
    )
    start, end = _as_of_positions(
        pd.Index(history[CUSTOMER_ID_COL]),
        history[VISIT_DATE_COL],
        pd.Index(customer_ids),
        snapshot_dates,
        side="right",
    )
    features = features.join(_anchoring_visit_features(history, start, end, snapshot_dates))
    features = features.join(_mpr_as_of(history, start, end, customer_ids.index))
    features[NO_SHOW_RATE_COL] = _no_show_rate_as_of(history, customer_ids, snapshot_dates)

    for col in MART1_NUMERIC_FEATURE_COLS:
        features[col] = features[col].astype(float)
    for col in MART1_BOOLEAN_FEATURE_COLS:
        features[col] = features[col].astype("boolean")
    return features


def _visit_table(raw_visits: pd.DataFrame) -> pd.DataFrame:
    """One row per 조제판매ID (in first-appearance order) with every
    visit-level attribute Mart 1's as-of features read: customer and family
    identity, 내방일/다음내방일, 처방조제일수, insurance/차상위 status, and
    the visit's primary drug (see `_primary_drug_by_visit`)."""
    visits = _one_row_per_visit(
        raw_visits,
        [
            CUSTOMER_ID_COL,
            FAMILY_ID_COL,
            VISIT_DATE_COL,
            NEXT_EXPECTED_VISIT_COL,
            PRESCRIPTION_DAYS_COL,
            INSURANCE_TYPE_COL,
            NEAR_POVERTY_COL,
        ],
    )
    visits[VISIT_DATE_COL] = pd.to_datetime(visits[VISIT_DATE_COL])
    visits[NEXT_EXPECTED_VISIT_COL] = pd.to_datetime(visits[NEXT_EXPECTED_VISIT_COL])
    visits[PRESCRIPTION_DAYS_COL] = pd.to_numeric(visits[PRESCRIPTION_DAYS_COL])
    return visits.join(_primary_drug_by_visit(raw_visits)).reset_index()


def _primary_drug_by_visit(raw_visits: pd.DataFrame) -> pd.DataFrame:
    """Per 조제판매ID, the 투약일수/속명 of whichever drug has the longest
    투약일수 -- the same drug 장기투약_일수 and 주요_약품속명 both name (see
    CONTEXT.md's Mart 1 entry); on a tie, the visit's first such drug row.
    A visit whose drug rows are all missing 투약일수 has no row here (its
    primary drug is undefined, NA once joined)."""
    medication_days = pd.to_numeric(raw_visits[MEDICATION_DAYS_COL], errors="coerce")
    drugs = raw_visits[[VISIT_ID_COL, INGREDIENT_COL]].assign(
        **{LONG_TERM_MED_DAYS_COL: medication_days}
    )[medication_days.notna()]
    is_longest = drugs[LONG_TERM_MED_DAYS_COL].eq(
        drugs.groupby(VISIT_ID_COL)[LONG_TERM_MED_DAYS_COL].transform("max")
    )
    primary = drugs[is_longest].groupby(VISIT_ID_COL, sort=False).head(1)
    return primary.set_index(VISIT_ID_COL)[[LONG_TERM_MED_DAYS_COL, INGREDIENT_COL]].rename(
        columns={INGREDIENT_COL: PRIMARY_INGREDIENT_COL}
    )


def _age_in_years(birth_dates: pd.Series, as_of_dates: pd.Series) -> pd.Series:
    """Whole-year age as of each row's `as_of_dates` value, accounting for
    whether that year's birthday has already passed."""
    had_birthday_this_year = (birth_dates.dt.month < as_of_dates.dt.month) | (
        (birth_dates.dt.month == as_of_dates.dt.month)
        & (birth_dates.dt.day <= as_of_dates.dt.day)
    )
    return as_of_dates.dt.year - birth_dates.dt.year - (~had_birthday_this_year).astype(int)


def _family_visit_counts_as_of(
    visits: pd.DataFrame, family_ids: pd.Series, snapshot_dates: pd.Series
) -> pd.Series:
    """가족_총내방 for each row: the number of distinct visits by that row's
    가족ID dated strictly before its snapshot date (see `family_totals_as_of`
    for why strictly). 0 for a known family with no prior visit yet; NA for
    a 가족ID that never appears on any visit."""
    counted = (
        visits[visits[FAMILY_ID_COL].notna() & visits[VISIT_DATE_COL].notna()]
        .sort_values([FAMILY_ID_COL, VISIT_DATE_COL], kind="stable")
    )
    start, end = _as_of_positions(
        pd.Index(counted[FAMILY_ID_COL]),
        counted[VISIT_DATE_COL],
        pd.Index(family_ids),
        snapshot_dates,
        side="left",
    )
    counts = pd.Series(end - start, index=family_ids.index)
    return counts.where(family_ids.isin(visits[FAMILY_ID_COL]))


def _anchoring_visit_features(
    history: pd.DataFrame, start: np.ndarray, end: np.ndarray, snapshot_dates: pd.Series
) -> pd.DataFrame:
    """The visit-timing/medication/insurance features derived from each
    row's "anchoring visit" -- its customer's most recent visit at or
    before the row's snapshot date (`history` rows start..end-1, see
    `_mart1_x_features`). Indexed like `snapshot_dates`.

    Uses `<=` (a visit dated exactly on the snapshot date is eligible to
    anchor), matching `_build_mart2`'s latest-consumption-value lookup --
    this is "what do we know as of this snapshot" for a point-feature, not a
    cumulative count. That's a different question from `family_totals_as_of`
    using strict `<`: a cumulative total must exclude the row's own visit to
    avoid a visit counting itself (per docs/adr/0002-point-in-time-correctness.md,
    "up to but excluding that visit"), but there's no equivalent
    self-counting risk in picking which single visit anchors these features.

    Every attribute is read from the anchoring visit alone: a value missing
    on that visit stays missing (NA), never back-filled from an earlier
    visit (see CONTEXT.md "Anchoring Visit"). A row with no visit at or
    before its snapshot date has no anchoring visit yet, so its features
    come back missing (NA) rather than computed from a future visit.
    """
    visit_cols = [
        NEXT_EXPECTED_VISIT_COL,
        LONG_TERM_MED_DAYS_COL,
        PRIMARY_INGREDIENT_COL,
        INSURANCE_TYPE_COL,
        NEAR_POVERTY_COL,
    ]
    if history.empty:
        anchoring = pd.DataFrame(np.nan, index=snapshot_dates.index, columns=ANCHORING_FEATURE_COLS)
        anchoring[NEAR_POVERTY_COL] = False
        return anchoring

    has_anchor = pd.Series(end > start, index=snapshot_dates.index)
    anchor_rows = np.where(has_anchor, end - 1, 0)
    anchoring = history[visit_cols].iloc[anchor_rows].set_axis(snapshot_dates.index)
    anchoring[VISIT_DATE_COL] = pd.to_datetime(
        history[VISIT_DATE_COL].iloc[anchor_rows]
    ).set_axis(snapshot_dates.index)
    anchoring = anchoring.where(has_anchor, np.nan)

    anchoring[DAYS_SINCE_LAST_VISIT_COL] = (snapshot_dates - anchoring[VISIT_DATE_COL]).dt.days
    # 남은_약_일수: days left on the anchoring visit's longest-투약일수 drug
    # (장기투약_일수) minus days elapsed since that visit. Issue #7 doesn't
    # give an explicit formula ("days of medication remaining"), so this is
    # an implementation judgment call, not a spec-given identity. Left
    # unclamped by design -- a negative value means the patient is overdue
    # for refill on that drug, which is itself a meaningful signal for
    # Track 1's visit-probability model, not an error case to hide.
    anchoring[REMAINING_MED_DAYS_COL] = (
        anchoring[LONG_TERM_MED_DAYS_COL] - anchoring[DAYS_SINCE_LAST_VISIT_COL]
    )
    anchoring[TOMORROW_IS_EXPECTED_VISIT_COL] = (
        snapshot_dates + pd.Timedelta(days=1)
        == pd.to_datetime(anchoring[NEXT_EXPECTED_VISIT_COL])
    ).where(has_anchor)
    # 차상위대상자 is a raw 'Y'/blank eligibility flag (same source table and
    # convention as the severity flags) — normalize to boolean the same way.
    anchoring[NEAR_POVERTY_COL] = _eligibility_flag_mask(anchoring[NEAR_POVERTY_COL])
    return anchoring[ANCHORING_FEATURE_COLS]


def _mpr_as_of(
    history: pd.DataFrame, start: np.ndarray, end: np.ndarray, index: pd.Index
) -> pd.DataFrame:
    """복약_순응도 (MPR adherence score) for each row, from its customer's
    visits at or before its snapshot date (`history` rows start..end-1, see
    `_mart1_x_features`), per docs/Research-Log.md's formula:
        Sigma(처방조제일수) / (최종내방일 - 최초내방일 + 최종조제일수) x 100
    NA when there's no eligible visit, or when that denominator isn't a
    usable number -- either the last eligible visit's 처방조제일수 is itself
    missing, or the denominator comes out to zero (a single eligible visit
    with 처방조제일수 == 0) -- rather than propagating a raw NaN or dividing
    by zero.
    """
    if history.empty:
        return pd.DataFrame({MPR_COL: np.nan}, index=index)

    prescription_days = history[PRESCRIPTION_DAYS_COL].to_numpy(dtype=float)
    visit_dates = history[VISIT_DATE_COL].to_numpy("datetime64[ns]")
    cumulative_days = np.concatenate([[0.0], np.cumsum(np.nan_to_num(prescription_days))])

    has_visit = end > start
    first_row = np.where(has_visit, start, 0)
    last_row = np.where(has_visit, end - 1, 0)
    span_days = pd.Series(visit_dates[last_row] - visit_dates[first_row]).dt.days.to_numpy()
    last_prescription_days = prescription_days[last_row]
    denominator = span_days + last_prescription_days
    usable = has_visit & ~np.isnan(last_prescription_days) & (denominator != 0)
    total_days = cumulative_days[end] - cumulative_days[start]
    with np.errstate(divide="ignore", invalid="ignore"):
        mpr = np.where(usable, total_days / denominator * 100, np.nan)
    return pd.DataFrame({MPR_COL: mpr}, index=index)


def _no_show_rate_as_of(
    visits: pd.DataFrame, customer_ids: pd.Series, snapshot_dates: pd.Series
) -> pd.Series:
    """노쇼_비율 (no-show rate) for each row: of its customer's visits whose
    다음내방일 is already resolvable as of the row's snapshot date (both the
    visit and its 다음내방일 on or before that date -- a later 다음내방일
    hasn't happened yet, so we can't yet know whether it'll be kept), the
    proportion not matched by an actual visit by that customer dated exactly
    on that 다음내방일. NA if none are resolvable yet -- there's no history
    to compute a rate from, not evidence of a 0% or 100% rate.
    """
    resolvable = visits[visits[NEXT_EXPECTED_VISIT_COL].notna()]
    kept = pd.MultiIndex.from_frame(
        resolvable[[CUSTOMER_ID_COL, NEXT_EXPECTED_VISIT_COL]]
    ).isin(pd.MultiIndex.from_frame(visits[[CUSTOMER_ID_COL, VISIT_DATE_COL]]))
    resolvable = resolvable.assign(
        _resolved_on=resolvable[[VISIT_DATE_COL, NEXT_EXPECTED_VISIT_COL]].max(axis=1),
        _missed=~kept,
    ).sort_values([CUSTOMER_ID_COL, "_resolved_on"], kind="stable")

    start, end = _as_of_positions(
        pd.Index(resolvable[CUSTOMER_ID_COL]),
        resolvable["_resolved_on"],
        pd.Index(customer_ids),
        snapshot_dates,
        side="right",
    )
    cumulative_missed = np.concatenate([[0], np.cumsum(resolvable["_missed"].to_numpy())])
    resolved_count = end - start
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(
            resolved_count > 0,
            (cumulative_missed[end] - cumulative_missed[start]) / resolved_count,
            np.nan,
        )
    return pd.Series(rate, index=customer_ids.index)


def _attach_mart1_weight(
    mart1: pd.DataFrame, raw_visits: pd.DataFrame, chronic_customer_ids: set
) -> pd.DataFrame:
    """Joins in Mart 1's 학습_가중치 sample weight (see `sample_mart1_weights`).
    `mart1` must already have `고객ID`.

    `raw_visits` is whatever visit population the caller wants severity/
    chronic tiers derived from: `_build_mart1` passes in the as-of-filtered
    visits alongside its as-of Chronic population, keeping `build_marts`'s
    snapshot fully point-in-time correct; `build_mart1_training_set`
    passes in the full, unfiltered table for the weight tiers only (its rows
    are already filtered by Chronic-since Date), matching
    `sample_mart1_weights`'s own live/full-history-by-default behavior when
    called directly (see its docstring).
    """
    weights = sample_mart1_weights(raw_visits, chronic_customer_ids)
    mart1 = mart1.copy()
    mart1[WEIGHT_COL] = mart1[CUSTOMER_ID_COL].map(weights)
    return mart1


def family_totals_as_of(raw_visits: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """가족_총내방 (family cumulative visit count) computed as of `as_of_date`.

    For each 가족ID, counts that family's distinct visits (조제판매ID) whose
    내방일 falls strictly before `as_of_date` — cumulative up to but excluding
    as_of_date itself, per docs/adr/0002-point-in-time-correctness.md — never
    the live `tbl가족총매출` snapshot already sitting on raw_visits.

    가족_총매출 (family revenue) is deliberately not computed here: raw_visits
    carries no per-visit monetary amount, only drug consumption quantities in
    units that aren't comparable across drugs, so there's no correct way to
    recompute it from the columns build_marts receives today. This is a
    permanent scope decision, not a gap awaiting a future extract change —
    see docs/adr/0002-point-in-time-correctness.md (closing #10).

    Returns one row per 가족ID with columns [가족ID, 가족_총내방].
    """
    as_of_date = pd.Timestamp(as_of_date)
    visits = _one_row_per_visit(raw_visits, [FAMILY_ID_COL, VISIT_DATE_COL])

    families = (
        visits[[FAMILY_ID_COL]]
        .drop_duplicates()
        .sort_values(FAMILY_ID_COL)
        .reset_index(drop=True)
    )
    families[FAMILY_VISIT_COUNT_COL] = (
        _family_visit_counts_as_of(
            visits, families[FAMILY_ID_COL], pd.Series(as_of_date, index=families.index)
        )
        .fillna(0)
        .astype(int)
    )
    return families


def _build_mart2(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Per 고객ID×약품ID, the latest single-visit consumption amount at or
    before `as_of_date` — never the global-latest value across the whole
    extract, per docs/adr/0002-point-in-time-correctness.md. A 소모량
    missing on that latest visit stays missing, never back-filled from an
    earlier visit (the same rule as CONTEXT.md "Anchoring Visit").
    """
    eligible = _visits_at_or_before(raw_visits, as_of_date).dropna(subset=[DRUG_ID_COL])
    if eligible.empty:
        return pd.DataFrame(columns=MART2_COLUMNS)

    subset = eligible[[CUSTOMER_ID_COL, DRUG_ID_COL, VISIT_DATE_COL, CONSUMPTION_COL]]
    latest = (
        subset.sort_values(VISIT_DATE_COL, kind="stable")
        .groupby([CUSTOMER_ID_COL, DRUG_ID_COL], sort=False)
        .tail(1)
    )
    mart2 = latest[[CUSTOMER_ID_COL, DRUG_ID_COL, CONSUMPTION_COL]].rename(
        columns={CONSUMPTION_COL: MART2_VALUE_COL}
    )
    return mart2.sort_values(
        [CUSTOMER_ID_COL, DRUG_ID_COL], kind="stable"
    ).reset_index(drop=True)


def _seasons_for(dates: pd.Series) -> pd.Series:
    """Each date's 계절 (season), per `_MONTH_TO_SEASON`."""
    return dates.dt.month.map(_MONTH_TO_SEASON)


def _weekdays_for(dates: pd.Series) -> pd.Series:
    """Each date's 요일 (Korean weekday name), per `MART3_WEEKDAYS`."""
    return dates.dt.dayofweek.map(_WEEKDAY_INDEX_TO_NAME)


def season_and_weekday_for(date) -> tuple[str, str]:
    """The (계절, 요일) pair for a single date, via the same derivation
    `_seasons_for`/`_weekdays_for` use for Mart 3's grid axes -- exposed so a
    caller with just one date (Track 2's target-date lookup, see
    `pipeline.inference`) doesn't need to wrap it in a one-row DataFrame to
    reuse that derivation."""
    date = pd.Timestamp(date)
    return _MONTH_TO_SEASON[date.month], _WEEKDAY_INDEX_TO_NAME[date.dayofweek]


def _mart3_eligible_visits(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_customer_ids: set
) -> pd.DataFrame:
    """Mart 3's shared visit-level population: Acute-Patient visits (the
    Revisit Match complement of Chronic, see ADR-0001) at or before
    `as_of_date`, with a non-null 약품ID. Shared by `_mart3_observations`
    (Mart 3's season x weekday grid input) and `_rare_drug_ids` (the
    rare-drug population filter, see docs/adr/0003-mart3-population-and-
    backoff-thresholds.md) so both agree on exactly which visits count as
    evidence for a given drug.
    """
    acute_visits = raw_visits.loc[~raw_visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]
    return _visits_at_or_before(acute_visits, as_of_date).dropna(subset=[DRUG_ID_COL])


def _mart3_observations(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_customer_ids: set
) -> pd.DataFrame:
    """One row per Acute-Patient drug-consumption observation at or before
    `as_of_date` — Mart 3's population, the Revisit Match complement of
    Chronic (see ADR-0001; Chronic-patient visits belong to Mart 1 instead
    and never contribute here). Each row's 계절/요일 are derived from its own
    내방일, matching `_build_mart2`'s `<=` as-of cutoff (point-in-time
    correctness for backtesting/inference reuse, see
    docs/adr/0002-point-in-time-correctness.md).
    """
    eligible = _mart3_eligible_visits(raw_visits, as_of_date, chronic_customer_ids)
    if eligible.empty:
        return pd.DataFrame(columns=MART3_COLUMNS)

    eligible_dates = pd.to_datetime(eligible[VISIT_DATE_COL])
    return pd.DataFrame(
        {
            DRUG_ID_COL: eligible[DRUG_ID_COL],
            SEASON_COL: _seasons_for(eligible_dates),
            WEEKDAY_COL: _weekdays_for(eligible_dates),
            CONSUMPTION_COL: eligible[CONSUMPTION_COL],
        }
    ).reset_index(drop=True)


def resolve_mart3_backoff(
    observations: pd.DataFrame,
    bucket_min_observations: int = MART3_BUCKET_MIN_OBSERVATIONS,
    season_min_observations: int = MART3_SEASON_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Mart 3's drug x season x weekday grid, from one row per Acute-Patient
    drug-consumption observation (columns 약품ID, 계절, 요일, 소모량).

    Every drug present in `observations` gets one row for each of the 28
    MART3_SEASONS x MART3_WEEKDAYS combinations, via hierarchical backoff
    (see CONTEXT.md "Track 2 Sparse-Bucket Backoff"):

    1. That bucket's own average, if the bucket has >= `bucket_min_observations`
       observations.
    2. Else that drug's season-only average (across every weekday), if the
       season has >= `season_min_observations` observations.
    3. Else that drug's overall average — used unconditionally as the final
       fallback, regardless of how few observations it itself rests on, so
       every bucket resolves to a usable estimate rather than null/zero.

    `bucket_min_observations` and `season_min_observations` are independent
    thresholds (see docs/adr/0003-mart3-population-and-backoff-thresholds.md)
    -- a season pools up to 7 weekdays of bucket data, so reusing one shared
    value would make the season tier nearly vestigial.

    Patient-agnostic by design: this function has no access to 고객ID, so
    Mart 3's rare-drug population filter (see `_build_mart3`) must exclude
    ineligible drugs from `observations` before calling this function, not
    something this function does itself.

    Kept as its own function — reachable through `build_marts`'s Mart 3
    output for ordinary use — so a test can exercise the backoff hierarchy
    directly against a small synthetic observation table without needing a
    full multi-visit raw_visits dataset.
    """
    drug_ids = sorted(observations[DRUG_ID_COL].unique())
    if not drug_ids:
        return pd.DataFrame(columns=MART3_COLUMNS)

    bucket_stats = (
        observations.groupby([DRUG_ID_COL, SEASON_COL, WEEKDAY_COL])[CONSUMPTION_COL]
        .agg(**{_BUCKET_MEAN_COL: "mean", _BUCKET_COUNT_COL: "count"})
        .reset_index()
    )
    season_stats = (
        observations.groupby([DRUG_ID_COL, SEASON_COL])[CONSUMPTION_COL]
        .agg(**{_SEASON_MEAN_COL: "mean", _SEASON_COUNT_COL: "count"})
        .reset_index()
    )
    drug_stats = (
        observations.groupby(DRUG_ID_COL)[CONSUMPTION_COL]
        .mean()
        .rename(_DRUG_MEAN_COL)
        .reset_index()
    )

    grid = pd.MultiIndex.from_product(
        [drug_ids, MART3_SEASONS, MART3_WEEKDAYS],
        names=[DRUG_ID_COL, SEASON_COL, WEEKDAY_COL],
    ).to_frame(index=False)
    grid = grid.merge(bucket_stats, on=[DRUG_ID_COL, SEASON_COL, WEEKDAY_COL], how="left")
    grid = grid.merge(season_stats, on=[DRUG_ID_COL, SEASON_COL], how="left")
    grid = grid.merge(drug_stats, on=DRUG_ID_COL, how="left")

    # NaN counts (bucket/season combinations absent from `observations`)
    # compare False against each threshold, correctly treated as sparse.
    bucket_sufficient = grid[_BUCKET_COUNT_COL] >= bucket_min_observations
    season_sufficient = grid[_SEASON_COUNT_COL] >= season_min_observations

    grid[CONSUMPTION_COL] = grid[_DRUG_MEAN_COL]
    grid.loc[season_sufficient, CONSUMPTION_COL] = grid.loc[season_sufficient, _SEASON_MEAN_COL]
    grid.loc[bucket_sufficient, CONSUMPTION_COL] = grid.loc[bucket_sufficient, _BUCKET_MEAN_COL]

    return grid[MART3_COLUMNS]


def _drug_ids_below_patient_threshold(
    eligible: pd.DataFrame, as_of_date: pd.Timestamp, patient_threshold: int
) -> set:
    """Shared rare-drug counting logic behind `_rare_drug_ids` (Mart 3's
    Acute-scoped filter, see docs/adr/0003-mart3-population-and-backoff-
    thresholds.md) and `chronic_rare_drug_ids` (Track 1's Chronic-scoped
    counterpart, see docs/adr/0004-track1-rare-drug-population-and-
    allocation.md): given a population's eligible visits (already restricted
    to that population, to 내방일 <= `as_of_date`, and to a non-null 약품ID by
    the caller), the drug IDs with fewer than `patient_threshold` distinct
    patients in the trailing 12 months ending at `as_of_date` (visits with
    내방일 > as_of_date - 12 calendar months and <= as_of_date, matching
    `_visits_at_or_before`'s inclusive-upper-bound convention).

    As-of-date correct (see docs/adr/0002-point-in-time-correctness.md): a
    patient whose only qualifying visit falls after `as_of_date`, or before
    the trailing-12-month window, must not count toward this snapshot's
    total. A drug at exactly the threshold is not excluded -- CONTEXT.md's
    rare-drug cutoff is strictly-less-than.
    """
    if eligible.empty:
        return set()

    window_start = as_of_date - pd.DateOffset(months=12)
    visit_dates = pd.to_datetime(eligible[VISIT_DATE_COL])
    windowed = eligible.loc[visit_dates > window_start]

    all_drug_ids = eligible[DRUG_ID_COL].unique()
    patient_counts = (
        windowed.groupby(DRUG_ID_COL)[CUSTOMER_ID_COL]
        .nunique()
        .reindex(all_drug_ids, fill_value=0)
    )
    return set(patient_counts.index[patient_counts < patient_threshold])


def _rare_drug_ids(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_customer_ids: set,
    rare_drug_patient_threshold: int,
) -> set:
    """Drug IDs excluded from Mart 3 entirely (see docs/adr/0003-mart3-
    population-and-backoff-thresholds.md): fewer than
    `rare_drug_patient_threshold` distinct patients among that drug's
    Acute-population visits -- `_mart3_eligible_visits`, the same population
    `_mart3_observations` builds its grid from (see
    `_drug_ids_below_patient_threshold` for the counting methodology).
    """
    eligible = _mart3_eligible_visits(raw_visits, as_of_date, chronic_customer_ids)
    return _drug_ids_below_patient_threshold(eligible, as_of_date, rare_drug_patient_threshold)


def _track1_eligible_visits(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_customer_ids: set
) -> pd.DataFrame:
    """Track 1's rare-drug population counterpart to `_mart3_eligible_visits`
    (see docs/adr/0004-track1-rare-drug-population-and-allocation.md):
    Chronic-Patient visits at or before `as_of_date` with a non-null 약품ID.
    Used only by `chronic_rare_drug_ids` below -- Track 1's own expected-value
    computation draws its population from Mart 1/Mart 2 instead, which are
    already Chronic-only by construction (see `pipeline.inference`).
    """
    chronic_visits = raw_visits.loc[raw_visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]
    return _visits_at_or_before(chronic_visits, as_of_date).dropna(subset=[DRUG_ID_COL])


def chronic_rare_drug_ids(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_customer_ids: set,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
) -> set:
    """Track 1's Chronic-population counterpart to `_rare_drug_ids` (see
    docs/adr/0004-track1-rare-drug-population-and-allocation.md): drugs with
    fewer than `rare_drug_patient_threshold` distinct Chronic patients --
    not Mart 3's Acute population -- in the trailing 12 months ending at
    `as_of_date` (see `_drug_ids_below_patient_threshold` for the shared
    counting methodology). The two counts are independent: the same drug can
    be "rare" under one and not the other, since Track 1 and Track 2 never
    share a patient population.

    Used by `pipeline.inference._track1_drug_demand` to switch a drug from
    ordinary expected-value multiplication to the 100%-allocation sum rule
    (see CONTEXT.md "Track 1 Rare-Drug Allocation").
    """
    eligible = _track1_eligible_visits(raw_visits, as_of_date, chronic_customer_ids)
    return _drug_ids_below_patient_threshold(eligible, as_of_date, rare_drug_patient_threshold)


def _build_mart3(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_customer_ids: set,
    bucket_min_observations: int = MART3_BUCKET_MIN_OBSERVATIONS,
    season_min_observations: int = MART3_SEASON_MIN_OBSERVATIONS,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
) -> pd.DataFrame:
    # No empty-check needed here: resolve_mart3_backoff already returns an
    # empty MART3_COLUMNS frame when `observations` has no rows.
    observations = _mart3_observations(raw_visits, as_of_date, chronic_customer_ids)
    rare_drug_ids = _rare_drug_ids(
        raw_visits, as_of_date, chronic_customer_ids, rare_drug_patient_threshold
    )
    observations = observations.loc[~observations[DRUG_ID_COL].isin(rare_drug_ids)]
    return resolve_mart3_backoff(observations, bucket_min_observations, season_min_observations)
