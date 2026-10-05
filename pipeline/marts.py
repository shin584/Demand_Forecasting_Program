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
split -- sampled train rows from that training set, full daily
`build_mart1_daily_snapshots` for validation and test.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from .pharmacy_calendar import PharmacyCalendar

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
# Whether the pharmacy is closed on the target day (기준일자 + 1) and on the
# snapshot day itself (see CONTEXT.md "Pharmacy Calendar" and issue #39):
# nobody visits on a closed day, and the day after one picks up its visits.
CLOSED_TOMORROW_COL = "내일_휴무"
CLOSED_TODAY_COL = "오늘_휴무"
MPR_NO_SHOW_FEATURE_COLS = [MPR_COL, NO_SHOW_RATE_COL]

# Severity/특례 eligibility flags (see CONTEXT.md "Severe/특례 Weight Tier").
# 차상위대상자 is deliberately excluded here -- it's a copay-assistance
# signal, not a severity signal, and belongs as an X-feature instead (see
# issue #6 and ADR-0001's sibling discussion in CONTEXT.md).
SEVERITY_FLAG_COLS = ["중증암등록대상자", "산전산모대상자", "희귀난치대상자"]

# Internal-only columns, not part of the raw_visits schema.
_DRUG_SET_COL = "_drug_set"
_CLEAN_DRUG_SET_COL = "_clean_drug_set"
_BUCKET_SUM_COL = "_bucket_sum"
_BUCKET_DAYS_COL = "_bucket_days"
_SEASON_SUM_COL = "_season_sum"
_SEASON_DAYS_COL = "_season_days"
_DRUG_SUM_COL = "_drug_sum"
_DRUG_DAYS_COL = "_drug_days"
_CALENDAR_DAY_COL = "_calendar_day"
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
    CLOSED_TOMORROW_COL,
    CLOSED_TODAY_COL,
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
MART1_BOOLEAN_FEATURE_COLS = [
    TOMORROW_IS_EXPECTED_VISIT_COL,
    NEAR_POVERTY_COL,
    CLOSED_TOMORROW_COL,
    CLOSED_TODAY_COL,
]

MART2_COLUMNS = [CUSTOMER_ID_COL, DRUG_ID_COL, MART2_VALUE_COL]
SEASON_COL = "계절"
WEEKDAY_COL = "요일"
# Expected consumption on one calendar day of the bucket, zero days included
# -- not the mean per dispensing (see CONTEXT.md "Mart 3" and issue #36).
MART3_VALUE_COL = "일평균_소모량"
MART3_COLUMNS = [DRUG_ID_COL, SEASON_COL, WEEKDAY_COL, MART3_VALUE_COL]
# Track 2's rare-drug counterpart to Mart 3 (see CONTEXT.md "Track 2
# Rare-Drug Allocation" and issue #37): per rare Acute drug, its latest
# single Acute dispensing as a stock floor -- a stock level, never daily
# demand.
RARE_STOCK_FLOOR_COL = "희귀약_최소재고"
TRACK2_RARE_DRUG_COLUMNS = [DRUG_ID_COL, RARE_STOCK_FLOOR_COL]

# Mart 3's fixed bucket axes (see CONTEXT.md "Mart 3 (Acute Drug Statistics
# Mart)" and "Track 2 Sparse-Bucket Backoff"): standard meteorological
# seasons, and Korean weekday names (matching dataset/D's weekday-named
# folders). Every drug present in Mart 3's population gets a row for every
# one of these 4x7 = 28 combinations, regardless of which buckets that drug
# was actually dispensed in -- a drug with zero winter sales still gets a
# 겨울 row (0 once enough winter days have passed, backed off before that),
# never omitted.
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
# separate bucket-level and season-level sufficiency bars, counted in
# calendar days since the drug's first Acute dispensing, not in dispensings
# (issue #36). Placeholders pending empirical tuning, not fixed business
# requirements (see CONTEXT.md "Decision Thresholds (provisional)").
MART3_BUCKET_MIN_DAYS = 5
MART3_SEASON_MIN_DAYS = 15

# Mart 3's rare-drug population filter (see docs/adr/0003-...): a drug with
# fewer than this many distinct patients (trailing 12 months, as of the
# snapshot date) gets no Mart 3 row at all -- handled by Track 2 Rare-Drug
# Allocation (`track2_rare_drug_allocation`) instead of a shaky seasonal
# estimate. Reuses
# CONTEXT.md's existing rare-drug special-handling cutoff ("Decision
# Thresholds (provisional)") rather than maintaining a second, independently-
# tunable "rare" definition.
RARE_DRUG_PATIENT_THRESHOLD = 5

# The Lapse Horizon (see CONTEXT.md "Lapsed Chronic Patient" and
# docs/adr/0005-lapse-aware-track1-population-and-evaluation.md): a Chronic
# customer whose latest visit is more than this many days before a snapshot
# date is Lapsed as of it -- still Chronic (so their visits from their
# Chronic-since Date on stay out of Mart 3), but
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
# own 처방조제일수 cycle length): sample points at ~15% early, ~50% mid,
# ~90-96% late, then post-cycle multiples so the model sees overdue customers
# who don't come back (see ADR-0005 "Post-cycle negatives"), plus one the day
# before the Lapse Horizon. The points only shape the windows each negative is
# jittered within (see negative_sample_windows and issue #35). The ~1:5
# positive:negative ratio the in-cycle fractions once produced is no longer a
# target.
NEGATIVE_SAMPLE_FRACTIONS = (0.15, 0.50, 0.90, 0.93, 0.96)
POST_CYCLE_NEGATIVE_SAMPLE_MULTIPLES = (1.25, 1.5, 2.0, 3.0, 4.0)
# Default seed for sample_mart1_negatives' jittered draws, so the same
# raw_visits always gives the same Mart 1 Training Set.
NEGATIVE_SAMPLE_SEED = 0

# A negative's sampling-window width in days (see negative_sample_windows),
# which build_mart1_training_set folds into the row's 학습_가중치.
WINDOW_WIDTH_COL = "샘플링_윈도우_폭"

MART1_POSITIVE_SAMPLE_COLUMNS = [
    CUSTOMER_ID_COL,
    VISIT_ID_COL,
    SNAPSHOT_DATE_COL,
    NEXT_DAY_VISIT_COL,
]
MART1_NEGATIVE_SAMPLE_COLUMNS = [*MART1_POSITIVE_SAMPLE_COLUMNS, WINDOW_WIDTH_COL]

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


class Mart1SplitWindows(NamedTuple):
    val_start: pd.Timestamp
    test_start: pd.Timestamp
    end: pd.Timestamp


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
    mart3_bucket_min_days: int = MART3_BUCKET_MIN_DAYS,
    mart3_season_min_days: int = MART3_SEASON_MIN_DAYS,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    pharmacy_calendar: PharmacyCalendar | None = None,
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
    Mart 1 inclusion and 만성질환여부 derive from this one as-of-correct
    result; Mart 3's history instead classifies each visit as of its own
    내방일, from the same Chronic-since Dates (see `_mart3_eligible_visits`). The top-2 high-frequency drug
    exclusion is taken from all of `raw_visits` (see ADR-0005, "Update
    (implementing #26)").

    `mart3_bucket_min_days`/`mart3_season_min_days` (defaults
    `MART3_BUCKET_MIN_DAYS`/`MART3_SEASON_MIN_DAYS`) are Mart 3's
    sparse-bucket backoff thresholds (see `resolve_mart3_backoff`).
    `rare_drug_patient_threshold` (default `RARE_DRUG_PATIENT_THRESHOLD`)
    excludes drugs with too few patients from Mart 3 entirely (see
    `_build_mart3`). `lapse_horizon_days` (default `LAPSE_HORIZON_DAYS`)
    drops Lapsed Chronic customers -- latest visit more than that many days
    before `as_of_date` -- from Mart 1 only; they stay Chronic, so Mart 3
    still excludes their visits from their Chronic-since Date on.
    `pharmacy_calendar` sets Mart 1's 내일_휴무/오늘_휴무 (see
    `_closed_day_features`); without one, no day is closed.
    """
    as_of_date = pd.Timestamp(as_of_date)
    chronic_since = chronic_since_dates(raw_visits)
    chronic_customer_ids = _chronic_customer_ids_as_of(chronic_since, as_of_date)
    mart1 = _build_mart1(
        raw_visits, as_of_date, chronic_customer_ids, lapse_horizon_days, pharmacy_calendar
    )
    mart2 = _build_mart2(raw_visits, as_of_date)
    mart3 = _build_mart3(
        raw_visits,
        as_of_date,
        chronic_since,
        mart3_bucket_min_days,
        mart3_season_min_days,
        rare_drug_patient_threshold,
    )
    return MartResult(mart1=mart1, mart2=mart2, mart3=mart3)


def build_mart1_training_set(
    raw_visits: pd.DataFrame,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    pharmacy_calendar: PharmacyCalendar | None = None,
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
    A row dated on or after the extract's last 내방일 is dropped too: its
    label (a visit on 기준일자 + 1) falls outside the data.
    학습_가중치 tiers are still derived once from the full, unfiltered
    `raw_visits`: the weight only scales a row's loss and is never a model
    input, so it can't leak into predictions. Each negative's weight is
    further scaled by its sampling window's width, normalised to mean 1
    over the negatives kept here, so every in-bounds day counts the same in
    expectation (see `sample_mart1_negatives`); positives keep their tier
    weight. The point, anchoring,
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
            sample_mart1_negatives(raw_visits, chronic_customer_ids, lapse_horizon_days),
        ],
        ignore_index=True,
    )
    is_chronic_as_of_row = (
        samples[SNAPSHOT_DATE_COL] >= samples[CUSTOMER_ID_COL].map(chronic_since)
    )
    # A row's Next-Day label is only observable while 기준일자 + 1 is still
    # inside the extract, so nothing is dated on or after its last 내방일 --
    # which also keeps split_mart1_training_set's windows (and the
    # backtest's default test dates) from running past the data.
    is_labelled = samples[SNAPSHOT_DATE_COL] < pd.to_datetime(raw_visits[VISIT_DATE_COL]).max()
    samples = samples[is_chronic_as_of_row & is_labelled].reset_index(drop=True)

    result = samples.join(
        _mart1_x_features(
            raw_visits, samples[CUSTOMER_ID_COL], samples[SNAPSHOT_DATE_COL], pharmacy_calendar
        )
    )
    result = result[_is_within_lapse_horizon(result, lapse_horizon_days)].reset_index(drop=True)
    # Derived solely from Revisit Match, same as _build_mart1 -- every row
    # here is already restricted to chronic_customer_ids.
    result[CHRONIC_COL] = True
    result = _attach_mart1_weight(result, raw_visits, chronic_customer_ids)
    # Each negative stands for its whole sampling window, so it's weighted by
    # the window's width, normalised to mean 1 over the negatives that made
    # it this far; positives (no window) keep their tier weight.
    window_widths = result[WINDOW_WIDTH_COL].astype(float)
    result[WEIGHT_COL] *= (window_widths / window_widths.mean()).fillna(1.0)
    return result[MART1_TRAINING_COLUMNS]


def mart1_split_windows(
    training_set: pd.DataFrame,
    test_months: int = TEST_WINDOW_MONTHS,
    val_months: int = VAL_WINDOW_MONTHS,
) -> Mart1SplitWindows:
    """The Temporal Split's window boundaries (see docs/Plan.md "시계열 분할"
    and `split_mart1_training_set`): test is the most recent `test_months`
    up to `training_set`'s max 기준일자 (`end`, inclusive), validation the
    `val_months` immediately before that, and train everything earlier.

    Both boundaries are computed from `training_set`'s own max 기준일자 --
    never a hardcoded calendar date -- so test and val always land on the
    same two fixed-size, most-recent windows regardless of how much history
    `training_set` covers; train simply absorbs whatever's left before them.
    This is "cut from the end, unrounded": the two eval windows are protected
    at their planned size, and train isn't padded or trimmed to hit a target
    proportion. Each boundary is inclusive on its more-recent side: a date
    exactly on a cutoff belongs to the newer of the two windows it separates.
    All three are NaT for an empty `training_set`.
    """
    if training_set.empty:
        return Mart1SplitWindows(val_start=pd.NaT, test_start=pd.NaT, end=pd.NaT)
    end = training_set[SNAPSHOT_DATE_COL].max()
    test_start = end - pd.DateOffset(months=test_months)
    val_start = test_start - pd.DateOffset(months=val_months)
    return Mart1SplitWindows(val_start=val_start, test_start=test_start, end=end)


def split_mart1_training_set(
    training_set: pd.DataFrame,
    raw_visits: pd.DataFrame,
    test_months: int = TEST_WINDOW_MONTHS,
    val_months: int = VAL_WINDOW_MONTHS,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    pharmacy_calendar: PharmacyCalendar | None = None,
) -> Mart1Split:
    """Splits Mart 1 into train/val/test over `mart1_split_windows`'s
    windows. Only train is drawn from `training_set`
    (`build_mart1_training_set`'s sampled rows, dated before the validation
    window). Validation and test are full daily snapshots of `raw_visits`
    (`build_mart1_daily_snapshots`, with `lapse_horizon_days`): every
    non-Lapsed Chronic customer on every day of the window, exactly the
    population and base rate Track 1 meets at inference, so early stopping
    and evaluation can't be flattered by the sampling scheme (see
    docs/adr/0005-lapse-aware-track1-population-and-evaluation.md,
    "Evaluation on the inference distribution").

    `raw_visits` must be the extract `training_set` was built from. All
    three parts come back empty for an empty `training_set`.
    """
    windows = mart1_split_windows(training_set, test_months, val_months)
    if pd.isna(windows.end):
        empty = pd.DataFrame(columns=MART1_TRAINING_COLUMNS)
        return Mart1Split(train=empty, val=empty.copy(), test=empty.copy())

    train = training_set[training_set[SNAPSHOT_DATE_COL] < windows.val_start]
    # One pass over both eval windows, sliced afterwards.
    snapshots = build_mart1_daily_snapshots(
        raw_visits, windows.val_start, windows.end, lapse_horizon_days, pharmacy_calendar
    )
    is_test = snapshots[SNAPSHOT_DATE_COL] >= windows.test_start
    val = snapshots[~is_test].reset_index(drop=True)
    test = snapshots[is_test].reset_index(drop=True)
    return Mart1Split(train=train, val=val, test=test)


def build_mart1_daily_snapshots(
    raw_visits: pd.DataFrame,
    start,
    end,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    pharmacy_calendar: PharmacyCalendar | None = None,
) -> pd.DataFrame:
    """Mart 1's full daily snapshots for every day from `start` to `end`
    (inclusive): the rows `build_marts(raw_visits, day).mart1` would hold
    for each day -- every non-Lapsed Chronic customer, with the same
    X-features, real Next-Day Visit label and as-of 학습_가중치 -- each
    tagged with its day as 기준일자 (`MART1_TRAINING_COLUMNS`), ordered by
    기준일자 then 고객ID.

    Built in one vectorized pass over every candidate (customer, day) pair
    rather than one `build_marts` call per day: a customer is a candidate
    from their Chronic-since Date (see `chronic_since_dates`) until
    `lapse_horizon_days` after their last visit, and a gap inside that span
    longer than the horizon is dropped by the same `_is_within_lapse_horizon`
    check `build_marts` applies. Days on or after the extract's last 내방일
    are left out, as in `build_mart1_training_set`, since their label falls
    outside the data. Returns an empty `MART1_TRAINING_COLUMNS`-shaped
    frame when no customer is in the population on any day.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    end = min(end, visit_dates.max() - pd.Timedelta(days=1))
    chronic_since = chronic_since_dates(raw_visits)

    last_visit = visit_dates.groupby(raw_visits[CUSTOMER_ID_COL]).max()
    first_days = chronic_since.clip(lower=start)
    last_days = (
        last_visit.reindex(chronic_since.index) + pd.Timedelta(days=lapse_horizon_days)
    ).clip(upper=end)
    day_counts = np.maximum((last_days - first_days).dt.days.to_numpy() + 1, 0)
    if day_counts.sum() == 0:
        return pd.DataFrame(columns=MART1_TRAINING_COLUMNS)

    # Each candidate customer repeated once per day, with day offsets
    # counting up from 0 within each customer's run.
    run_starts = np.repeat(np.cumsum(day_counts) - day_counts, day_counts)
    offsets = np.arange(day_counts.sum()) - run_starts
    rows = pd.DataFrame(
        {
            SNAPSHOT_DATE_COL: np.repeat(first_days.to_numpy(), day_counts)
            + pd.to_timedelta(offsets, unit="D"),
            CUSTOMER_ID_COL: np.repeat(chronic_since.index.to_numpy(), day_counts),
        }
    ).sort_values([SNAPSHOT_DATE_COL, CUSTOMER_ID_COL], kind="stable", ignore_index=True)

    visited = pd.MultiIndex.from_arrays([raw_visits[CUSTOMER_ID_COL], visit_dates])
    rows[NEXT_DAY_VISIT_COL] = pd.MultiIndex.from_arrays(
        [rows[CUSTOMER_ID_COL], rows[SNAPSHOT_DATE_COL] + pd.Timedelta(days=1)]
    ).isin(visited)
    rows[CHRONIC_COL] = True
    rows = rows.join(
        _mart1_x_features(
            raw_visits, rows[CUSTOMER_ID_COL], rows[SNAPSHOT_DATE_COL], pharmacy_calendar
        )
    )
    rows = rows[_is_within_lapse_horizon(rows, lapse_horizon_days)].reset_index(drop=True)

    # As-of-correct, as `build_marts` gives it (`sample_mart1_weights` over
    # visits up to the day): every row is Chronic, so it's the chronic tier
    # until the customer's first severity-flagged visit, the severity tier
    # from then on.
    severity_since = _severity_since_dates(raw_visits)
    is_severe = rows[SNAPSHOT_DATE_COL] >= rows[CUSTOMER_ID_COL].map(severity_since)
    rows[WEIGHT_COL] = np.where(is_severe, SEVERITY_TIER_WEIGHT, CHRONIC_TIER_WEIGHT)
    return rows[MART1_TRAINING_COLUMNS]


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
    latest-consumption lookup, Mart 3's Acute visit population, Mart 1's
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
    return set(raw_visits.loc[_severity_flag_mask(raw_visits), CUSTOMER_ID_COL])


def _severity_flag_mask(raw_visits: pd.DataFrame) -> pd.Series:
    """True for each raw row flagged by any `SEVERITY_FLAG_COLS` column."""
    mask = pd.Series(False, index=raw_visits.index)
    for col in SEVERITY_FLAG_COLS:
        mask = mask | _eligibility_flag_mask(raw_visits[col])
    return mask


def _severity_since_dates(raw_visits: pd.DataFrame) -> pd.Series:
    """Per 고객ID, the 내방일 of their first visit flagged by any
    `SEVERITY_FLAG_COLS` column -- the day they join the severity weight
    tier as of-correctly (see `_severity_customer_ids`). Customers never
    flagged have no entry."""
    flagged = raw_visits[_severity_flag_mask(raw_visits)]
    return pd.to_datetime(flagged[VISIT_DATE_COL]).groupby(flagged[CUSTOMER_ID_COL]).min()


def _eligibility_flag_mask(column: pd.Series) -> pd.Series:
    """True where a raw 대상자 eligibility flag column value represents
    'yes'.

    Source columns store 'Y' (case-insensitive) for membership and
    None/NaN/blank otherwise (see docs/Research-Log.md "학습 가중치").
    """
    return column.astype("string").str.strip().str.upper().eq("Y").fillna(False)


def negative_sample_windows(
    prescription_days: float, lapse_horizon_days: int = LAPSE_HORIZON_DAYS
) -> list[tuple[int, int]]:
    """Day-offset windows (relative to an anchoring visit) for Mart 1's Y=0
    negative sampling (see CONTEXT.md "Negative Sampling Windows" and
    issue #35), each a half-open [lo, hi) that one negative is drawn from.

    The windows are built around sample points that are proportions of
    `prescription_days` - that visit's own 처방조제일수 cycle length -
    rather than fixed absolute days, so sampling stays meaningful whether
    the observed cycle is 7 days or 60+: in-cycle at 15/50/90/93/96%,
    post-cycle at 1.25/1.5/2/3/4x, plus the day before the Lapse Horizon.
    Points past `lapse_horizon_days`, where the customer is Lapsed and
    never scored, are left out; a missing `prescription_days` leaves only
    the day-before-horizon point. Each window runs between the midpoints
    with its neighbouring points, rounded to whole days; the first starts
    on day 1 and the last ends on the horizon's own day, so together they
    tile days 1..lapse_horizon_days with no gap or overlap. Windows that
    round to nothing are dropped, which is how a short cycle's crowded
    points collapse.
    """
    points = [lapse_horizon_days - 1]
    if not np.isnan(prescription_days):
        points += [
            prescription_days * multiple
            for multiple in NEGATIVE_SAMPLE_FRACTIONS + POST_CYCLE_NEGATIVE_SAMPLE_MULTIPLES
            if prescription_days * multiple <= lapse_horizon_days
        ]
    points.sort()
    day_after_horizon = lapse_horizon_days + 1
    edges = [
        1,
        *(
            min(max(round((a + b) / 2), 1), day_after_horizon)
            for a, b in zip(points, points[1:])
        ),
        day_after_horizon,
    ]
    return [(lo, hi) for lo, hi in zip(edges, edges[1:]) if lo < hi]


def sample_mart1_negatives(
    raw_visits: pd.DataFrame,
    chronic_customer_ids: set | None = None,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    seed: int = NEGATIVE_SAMPLE_SEED,
) -> pd.DataFrame:
    """Mart 1's Y=0 negative-sample rows: one per (anchoring visit,
    `negative_sample_windows` window), on a uniformly random day of that
    window, anchored on every visit by a Chronic Patient (see ADR-0001) -
    including each customer's last known visit, so the model sees customers
    who are overdue and don't come back. Jittering within windows, rather
    than sampling on the points themselves, keeps the model from telling a
    negative apart by where 마지막방문_경과일 lands on a fixed grid (issue
    #35). Draws come from a `seed`ed RNG, so the same input always gives the
    same rows.

    A drawn row is dated that visit's own 내방일 plus the drawn day, and kept
    only if it's at least two days before the anchoring visit's next visit
    (the customer's earliest visit on a later date): a sample on the day
    before it would actually be a Next-Day Visit positive, and one on or
    after it describes a cycle that's already over. It must also be dated
    before the extract's last 내방일, so "no visit on 기준일자 + 1" is still
    observed -- the only bound on samples anchored by a customer's last
    visit, besides the Lapse Horizon the windows already stop at. Those
    bounds are what make the Y=0 label true, not just assumed. They're
    applied after the draw, never by shrinking a window first, so each
    window's expected contribution is the number of its days inside the
    bounds.

    Each row carries its window's width (`WINDOW_WIDTH_COL`), which
    `build_mart1_training_set` folds into 학습_가중치 so every in-bounds day
    counts the same in expectation, as under daily sampling.
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

    extract_end = pd.to_datetime(raw_visits[VISIT_DATE_COL]).max()
    visits = _one_row_per_visit(
        raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL, PRESCRIPTION_DAYS_COL]
    ).reset_index()
    visits = visits[visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]

    # Customers in first-appearance order, each customer's visits
    # chronological (stable, so same-day visits keep their relative order).
    visits = visits.assign(
        _customer_order=pd.factorize(visits[CUSTOMER_ID_COL])[0],
        **{VISIT_DATE_COL: pd.to_datetime(visits[VISIT_DATE_COL])},
    )
    visits = visits.sort_values(["_customer_order", VISIT_DATE_COL], kind="stable")
    # Each visit's next visit: the customer's earliest visit on a strictly
    # later date (a same-day visit isn't one). Visits on a customer's last
    # date have none (NaT), so only the extract's end bounds them.
    visit_days = visits[[CUSTOMER_ID_COL, VISIT_DATE_COL]].drop_duplicates()
    visit_days = visit_days.assign(
        _next_visit_date=visit_days.groupby(CUSTOMER_ID_COL)[VISIT_DATE_COL].shift(-1)
    )
    anchors = visits.merge(visit_days, on=[CUSTOMER_ID_COL, VISIT_DATE_COL], how="left")

    # Windows are computed once per distinct 처방조제일수 (a missing one
    # included), then joined onto every anchor with that value. Rows stay in
    # anchor-then-window order, so a seed always maps to the same draws.
    cycle_codes, cycle_lengths = pd.factorize(
        anchors[PRESCRIPTION_DAYS_COL].astype(float), use_na_sentinel=False
    )
    windows = pd.DataFrame(
        [
            (code, lo, hi)
            for code, cycle_length in enumerate(cycle_lengths)
            for lo, hi in negative_sample_windows(cycle_length, lapse_horizon_days)
        ],
        columns=["_cycle_code", "_lo", "_hi"],
    )
    samples = (
        pd.DataFrame({"_anchor": np.arange(len(anchors)), "_cycle_code": cycle_codes})
        .merge(windows, on="_cycle_code")
        .sort_values(["_anchor", "_lo"], kind="stable")
        .reset_index(drop=True)
    )
    offsets = np.random.default_rng(seed).integers(samples["_lo"], samples["_hi"])

    anchor_rows = anchors.iloc[samples["_anchor"]].reset_index(drop=True)
    snapshot_dates = anchor_rows[VISIT_DATE_COL] + pd.to_timedelta(offsets, unit="D")
    # A visit on the customer's last date has no next visit, so the
    # extract's end alone bounds its samples.
    next_visit_dates = anchor_rows["_next_visit_date"]
    is_before_day_before_next_visit = next_visit_dates.isna() | (
        snapshot_dates < next_visit_dates - pd.Timedelta(days=1)
    )
    is_true_negative = is_before_day_before_next_visit & (snapshot_dates < extract_end)
    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: anchor_rows[CUSTOMER_ID_COL],
            VISIT_ID_COL: anchor_rows[VISIT_ID_COL],
            SNAPSHOT_DATE_COL: snapshot_dates,
            NEXT_DAY_VISIT_COL: False,
            WINDOW_WIDTH_COL: samples["_hi"] - samples["_lo"],
        },
        columns=MART1_NEGATIVE_SAMPLE_COLUMNS,
    )[is_true_negative].reset_index(drop=True)


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
    set. Same columns as `sample_mart1_negatives` minus its window width
    (`MART1_POSITIVE_SAMPLE_COLUMNS`), so the two concatenate directly and a
    positive's width is left missing.
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
        columns=MART1_POSITIVE_SAMPLE_COLUMNS,
    )


def _build_mart1(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_customer_ids: set,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
    pharmacy_calendar: PharmacyCalendar | None = None,
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
        _mart1_x_features(
            raw_visits,
            customers,
            pd.Series(as_of_date, index=customers.index),
            pharmacy_calendar,
        )
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
    raw_visits: pd.DataFrame,
    customer_ids: pd.Series,
    snapshot_dates: pd.Series,
    pharmacy_calendar: PharmacyCalendar | None = None,
) -> pd.DataFrame:
    """Mart 1's X-features (see issue #7/#8) for each (고객ID, snapshot date)
    pair -- `customer_ids` and `snapshot_dates` are aligned Series, and the
    result shares their index. Every feature is point-in-time correct as of
    its own row's snapshot date (see docs/adr/0002-point-in-time-correctness.md):
    demographics, as-of family loyalty, visit-timing/medication and
    insurance/차상위 features anchored to the customer's most recent visit at
    or before that date, the MPR/no-show history aggregates, and
    `_closed_day_features` from `pharmacy_calendar`.

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
    features = features.join(_closed_day_features(snapshot_dates, pharmacy_calendar))

    for col in MART1_NUMERIC_FEATURE_COLS:
        features[col] = features[col].astype(float)
    for col in MART1_BOOLEAN_FEATURE_COLS:
        features[col] = features[col].astype("boolean")
    return features


def _closed_day_features(
    snapshot_dates: pd.Series, pharmacy_calendar: PharmacyCalendar | None
) -> pd.DataFrame:
    """내일_휴무/오늘_휴무 per row: whether `pharmacy_calendar` has the
    pharmacy closed on the row's target day (snapshot date + 1) and on the
    snapshot date itself. A closure is planned in advance, so the calendar
    is read as it stands, never as of the snapshot date (see
    `pipeline.pharmacy_calendar`). No calendar means no day is closed."""
    calendar = pharmacy_calendar or PharmacyCalendar.always_open()
    return pd.DataFrame(
        {
            CLOSED_TOMORROW_COL: calendar.is_closed(snapshot_dates + pd.Timedelta(days=1)),
            CLOSED_TODAY_COL: calendar.is_closed(snapshot_dates),
        }
    )


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


def current_regimen(raw_visits: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """Each customer's Current Regimen as of `as_of_date` (see CONTEXT.md
    "Current Regimen"): the (고객ID, 약품ID) pairs dispensed on their
    Anchoring Visit -- their most recent visit at or before `as_of_date`,
    picked the same way `_mart1_x_features` picks it (same-day visits keep
    their first-appearance order, the last one anchors). Track 1 demand
    multiplies only these pairs' Mart 2 amounts; Mart 2 itself still holds
    every drug a customer has ever been dispensed.

    Returns one row per pair, columns [고객ID, 약품ID], for every customer
    with a visit at or before `as_of_date` -- Chronic or not; the caller
    narrows the population.
    """
    as_of_date = pd.Timestamp(as_of_date)
    visits = _one_row_per_visit(raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL])
    visits[VISIT_DATE_COL] = pd.to_datetime(visits[VISIT_DATE_COL])
    anchoring_visit_ids = (
        visits[visits[VISIT_DATE_COL] <= as_of_date]
        .sort_values([CUSTOMER_ID_COL, VISIT_DATE_COL], kind="stable")
        .groupby(CUSTOMER_ID_COL, sort=False)
        .tail(1)
        .index
    )
    regimen = raw_visits.loc[
        raw_visits[VISIT_ID_COL].isin(anchoring_visit_ids), [CUSTOMER_ID_COL, DRUG_ID_COL]
    ].dropna(subset=[DRUG_ID_COL])
    return regimen.drop_duplicates().reset_index(drop=True)


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
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_since: pd.Series
) -> pd.DataFrame:
    """Mart 3's shared visit-level population: visits at or before
    `as_of_date`, with a non-null 약품ID, made while the customer was still
    Acute -- 내방일 on or before their Chronic-since Date (`chronic_since`,
    from `chronic_since_dates`), or no Chronic-since Date at all.

    Each visit is classified as of its own 내방일, not as of `as_of_date`:
    the visit on a customer's Chronic-since Date is one Track 1 didn't score
    the day before, so it's Track 2's to forecast (and lands in Track 2's
    actual in `pipeline.backtest`). Classifying the whole history as of
    `as_of_date` instead dropped every visit a now-Chronic customer made
    while still Acute -- the customers making most of Track 2's actual
    demand (issue #36). Shared by `_mart3_observations` and
    `_rare_drug_ids` so both agree on exactly which visits count as
    evidence for a given drug.
    """
    visits = _visits_at_or_before(raw_visits, as_of_date).dropna(subset=[DRUG_ID_COL])
    customer_chronic_since = visits[CUSTOMER_ID_COL].map(chronic_since)
    still_acute = customer_chronic_since.isna() | (
        pd.to_datetime(visits[VISIT_DATE_COL]) <= customer_chronic_since
    )
    return visits.loc[still_acute]


def _mart3_observations(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_since: pd.Series
) -> pd.DataFrame:
    """One row per Acute drug-consumption observation at or before
    `as_of_date` (see `_mart3_eligible_visits`), with columns 약품ID, 내방일,
    소모량 -- `resolve_mart3_backoff`'s input. Matches `_build_mart2`'s `<=`
    as-of cutoff (point-in-time correctness for backtesting/inference reuse,
    see docs/adr/0002-point-in-time-correctness.md).
    """
    eligible = _mart3_eligible_visits(raw_visits, as_of_date, chronic_since)
    return pd.DataFrame(
        {
            DRUG_ID_COL: eligible[DRUG_ID_COL],
            VISIT_DATE_COL: pd.to_datetime(eligible[VISIT_DATE_COL]),
            CONSUMPTION_COL: eligible[CONSUMPTION_COL],
        }
    ).reset_index(drop=True)


def _drug_bucket_days(first_dates: pd.Series, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Per drug x 계절 x 요일, how many calendar days of that bucket fall
    between the drug's first dispensing (`first_dates`, indexed by 약품ID)
    and `as_of_date`, both inclusive. Buckets with no such day are absent.
    """
    calendar = pd.DataFrame({_CALENDAR_DAY_COL: pd.date_range(first_dates.min(), as_of_date)})
    calendar[SEASON_COL] = _seasons_for(calendar[_CALENDAR_DAY_COL])
    calendar[WEEKDAY_COL] = _weekdays_for(calendar[_CALENDAR_DAY_COL])
    starts = first_dates.to_numpy()
    frames = []
    for (season, weekday), days in calendar.groupby([SEASON_COL, WEEKDAY_COL]):
        bucket_dates = days[_CALENDAR_DAY_COL].to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    DRUG_ID_COL: first_dates.index,
                    SEASON_COL: season,
                    WEEKDAY_COL: weekday,
                    _BUCKET_DAYS_COL: len(bucket_dates) - np.searchsorted(bucket_dates, starts),
                }
            )
        )
    bucket_days = pd.concat(frames, ignore_index=True)
    return bucket_days.loc[bucket_days[_BUCKET_DAYS_COL] > 0]


def resolve_mart3_backoff(
    observations: pd.DataFrame,
    as_of_date,
    bucket_min_days: int = MART3_BUCKET_MIN_DAYS,
    season_min_days: int = MART3_SEASON_MIN_DAYS,
) -> pd.DataFrame:
    """Mart 3's drug x season x weekday grid of expected daily consumption
    (`MART3_VALUE_COL`), from one row per Acute drug-consumption observation
    (columns 약품ID, 내방일, 소모량) at or before `as_of_date`.

    A drug's history is every calendar day from its first dispensing in
    `observations` to `as_of_date`, and a value is total 소모량 divided by
    calendar days: days the drug wasn't dispensed count as zeros, so a day
    the pharmacy is closed resolves to 0 rather than to a dispensing's worth
    (issue #36), once its bucket has `bucket_min_days` days. A missing
    소모량 counts as 0, matching how `pipeline.backtest` sums actuals. Every drug present gets one row for each of the 28
    MART3_SEASONS x MART3_WEEKDAYS combinations, via hierarchical backoff
    (see CONTEXT.md "Track 2 Sparse-Bucket Backoff"):

    1. That bucket's own daily average, if the bucket has >= `bucket_min_days`
       calendar days in the drug's history.
    2. Else that drug's season-only daily average (across every weekday), if
       the season has >= `season_min_days` calendar days.
    3. Else that drug's overall daily average -- used unconditionally as the
       final fallback, so every bucket resolves to a usable estimate.

    `bucket_min_days` and `season_min_days` are independent thresholds (see
    docs/adr/0003-mart3-population-and-backoff-thresholds.md).

    Patient-agnostic by design: this function has no access to 고객ID, so
    Mart 3's rare-drug population filter (see `_build_mart3`) must exclude
    ineligible drugs from `observations` before calling this function, not
    something this function does itself.

    Kept as its own function — reachable through `build_marts`'s Mart 3
    output for ordinary use — so a test can exercise the backoff hierarchy
    directly against a small synthetic observation table without needing a
    full multi-visit raw_visits dataset.
    """
    if observations.empty:
        return pd.DataFrame(columns=MART3_COLUMNS)

    as_of_date = pd.Timestamp(as_of_date)
    visit_dates = pd.to_datetime(observations[VISIT_DATE_COL])
    observations = observations.assign(
        **{SEASON_COL: _seasons_for(visit_dates), WEEKDAY_COL: _weekdays_for(visit_dates)}
    )
    first_dates = visit_dates.groupby(observations[DRUG_ID_COL]).min()

    bucket_keys = [DRUG_ID_COL, SEASON_COL, WEEKDAY_COL]
    season_keys = [DRUG_ID_COL, SEASON_COL]
    bucket_days = _drug_bucket_days(first_dates, as_of_date)
    season_days = bucket_days.groupby(season_keys)[_BUCKET_DAYS_COL].sum().rename(_SEASON_DAYS_COL)
    drug_days = ((as_of_date - first_dates).dt.days + 1).rename(_DRUG_DAYS_COL)
    consumption = observations[CONSUMPTION_COL]
    bucket_sums = consumption.groupby([observations[k] for k in bucket_keys]).sum()
    season_sums = consumption.groupby([observations[k] for k in season_keys]).sum()
    drug_sums = consumption.groupby(observations[DRUG_ID_COL]).sum()

    grid = pd.MultiIndex.from_product(
        [first_dates.index, MART3_SEASONS, MART3_WEEKDAYS], names=bucket_keys
    ).to_frame(index=False)
    grid = grid.merge(bucket_days, on=bucket_keys, how="left")
    grid = grid.merge(bucket_sums.rename(_BUCKET_SUM_COL).reset_index(), on=bucket_keys, how="left")
    grid = grid.merge(season_days.reset_index(), on=season_keys, how="left")
    grid = grid.merge(season_sums.rename(_SEASON_SUM_COL).reset_index(), on=season_keys, how="left")
    grid = grid.merge(drug_days.reset_index(), on=DRUG_ID_COL, how="left")
    grid = grid.merge(drug_sums.rename(_DRUG_SUM_COL).reset_index(), on=DRUG_ID_COL, how="left")
    # A bucket or season with no calendar day yet, or no dispensing in it,
    # is absent from the stats above: zero days, zero consumption.
    grid = grid.fillna(
        {_BUCKET_DAYS_COL: 0, _BUCKET_SUM_COL: 0.0, _SEASON_DAYS_COL: 0, _SEASON_SUM_COL: 0.0}
    )

    bucket_sufficient = grid[_BUCKET_DAYS_COL] >= bucket_min_days
    season_sufficient = grid[_SEASON_DAYS_COL] >= season_min_days

    grid[MART3_VALUE_COL] = grid[_DRUG_SUM_COL] / grid[_DRUG_DAYS_COL]
    grid.loc[season_sufficient, MART3_VALUE_COL] = (
        grid[_SEASON_SUM_COL] / grid[_SEASON_DAYS_COL]
    )[season_sufficient]
    grid.loc[bucket_sufficient, MART3_VALUE_COL] = (
        grid[_BUCKET_SUM_COL] / grid[_BUCKET_DAYS_COL]
    )[bucket_sufficient]

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

    patient_counts = _trailing_12_month_patient_counts(eligible, as_of_date)
    return set(patient_counts.index[patient_counts < patient_threshold])


def _trailing_12_month_patient_counts(
    eligible: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.Series:
    """Per 약품ID in `eligible`, its distinct patients in the trailing 12
    months ending at `as_of_date` (see `_drug_ids_below_patient_threshold`);
    0 for a drug with no visit in that window."""
    window_start = as_of_date - pd.DateOffset(months=12)
    visit_dates = pd.to_datetime(eligible[VISIT_DATE_COL])
    windowed = eligible.loc[visit_dates > window_start]
    return (
        windowed.groupby(DRUG_ID_COL)[CUSTOMER_ID_COL]
        .nunique()
        .reindex(eligible[DRUG_ID_COL].unique(), fill_value=0)
    )


def _rare_drug_ids(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_since: pd.Series,
    rare_drug_patient_threshold: int,
) -> set:
    """Drug IDs excluded from Mart 3 entirely (see docs/adr/0003-mart3-
    population-and-backoff-thresholds.md): fewer than
    `rare_drug_patient_threshold` distinct patients among that drug's
    Acute visits -- `_mart3_eligible_visits`, the same population
    `_mart3_observations` builds its grid from (see
    `_drug_ids_below_patient_threshold` for the counting methodology).
    """
    eligible = _mart3_eligible_visits(raw_visits, as_of_date, chronic_since)
    return _drug_ids_below_patient_threshold(eligible, as_of_date, rare_drug_patient_threshold)


def track2_rare_drug_allocation(
    raw_visits: pd.DataFrame,
    as_of_date,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
) -> pd.DataFrame:
    """Track 2's stock floor for the rare Acute drugs Mart 3 leaves out
    (`_rare_drug_ids`; see CONTEXT.md "Track 2 Rare-Drug Allocation" and
    issue #37), one row per drug, `TRACK2_RARE_DRUG_COLUMNS`:
    `RARE_STOCK_FLOOR_COL` is its latest single Acute dispensing with a
    known 소모량 (the largest, if several fall on that date) -- enough on
    hand for the next patient.

    A stock level, not daily demand: Track 2 forecasts no daily demand for
    these drugs at all. Their daily rate over-forecast them 2.7-5.3x on the
    v0.3 backtest, since a drug is rare precisely because its recent use
    fell below its history (issue #37).

    Covers only rare drugs with at least one Acute patient in the trailing
    12 months: one not dispensed for a year needs no stock on hand. Acute visits are
    Mart 3's own (`_mart3_eligible_visits`), each classified as of its own
    내방일.
    """
    as_of_date = pd.Timestamp(as_of_date)
    eligible = _mart3_eligible_visits(raw_visits, as_of_date, chronic_since_dates(raw_visits))
    if eligible.empty:
        return pd.DataFrame(columns=TRACK2_RARE_DRUG_COLUMNS)

    patient_counts = _trailing_12_month_patient_counts(eligible, as_of_date)
    rare_recent = patient_counts.index[
        (patient_counts > 0) & (patient_counts < rare_drug_patient_threshold)
    ]
    observations = eligible.loc[eligible[DRUG_ID_COL].isin(rare_recent)]
    if observations.empty:
        return pd.DataFrame(columns=TRACK2_RARE_DRUG_COLUMNS)

    # Skip dispensings with a missing 소모량, so the floor rests on the
    # latest one with a known amount.
    known = observations.dropna(subset=[CONSUMPTION_COL])
    known_dates = pd.to_datetime(known[VISIT_DATE_COL])
    on_latest_date = known_dates == known_dates.groupby(known[DRUG_ID_COL]).transform("max")
    stock_floor = known.loc[on_latest_date].groupby(DRUG_ID_COL)[CONSUMPTION_COL].max()
    return stock_floor.rename(RARE_STOCK_FLOOR_COL).reset_index()[TRACK2_RARE_DRUG_COLUMNS]


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
    be "rare" under one and not the other. A customer's visits from before
    their Chronic-since Date count toward both -- Track 1 serves them now,
    while Mart 3 classifies each visit as of its own date (issue #36).

    Used by `pipeline.inference._track1_drug_demand` to switch a drug from
    ordinary expected-value multiplication to the 100%-allocation sum rule
    (see CONTEXT.md "Track 1 Rare-Drug Allocation").
    """
    eligible = _track1_eligible_visits(raw_visits, as_of_date, chronic_customer_ids)
    return _drug_ids_below_patient_threshold(eligible, as_of_date, rare_drug_patient_threshold)


def _build_mart3(
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_since: pd.Series,
    bucket_min_days: int = MART3_BUCKET_MIN_DAYS,
    season_min_days: int = MART3_SEASON_MIN_DAYS,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
) -> pd.DataFrame:
    # No empty-check needed here: resolve_mart3_backoff already returns an
    # empty MART3_COLUMNS frame when `observations` has no rows.
    observations = _mart3_observations(raw_visits, as_of_date, chronic_since)
    rare_drug_ids = _rare_drug_ids(raw_visits, as_of_date, chronic_since, rare_drug_patient_threshold)
    observations = observations.loc[~observations[DRUG_ID_COL].isin(rare_drug_ids)]
    return resolve_mart3_backoff(observations, as_of_date, bucket_min_days, season_min_days)
