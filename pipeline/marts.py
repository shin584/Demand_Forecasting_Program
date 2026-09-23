"""Mart-construction pipeline: turns raw extracted visit-level rows into
Mart 1 (Visit-Probability), Mart 2 (Customer Drug Profile) and Mart 3
(Acute Drug Statistics).

See CONTEXT.md and docs/adr/ for the design decisions this pipeline encodes.
This module currently implements the pipeline scaffolding, Mart 1's Y label,
Revisit Match-based Chronic/Acute routing, Mart 1's sample-weight tiers,
Mart 1's remaining X-features (demographics, as-of family loyalty,
visit-timing/medication and insurance/차상위 features -- 가족_총매출 is
permanently out of scope, see below -- plus MPR adherence score and
per-patient no-show rate), Mart 2's as-of-date consumption values, and
as-of-date family visit totals; later tickets fill in Mart 3.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

CUSTOMER_ID_COL = "고객ID"
VISIT_ID_COL = "조제판매ID"
VISIT_DATE_COL = "내방일"
NEXT_EXPECTED_VISIT_COL = "다음내방일"
PRESCRIPTION_DAYS_COL = "처방조제일수"
FAMILY_ID_COL = "가족ID"
DRUG_ID_COL = "약품ID"
CONSUMPTION_COL = "소모량"
NEXT_DAY_VISIT_COL = "내일_방문"
CHRONIC_COL = "만성질환여부"
FAMILY_VISIT_COUNT_COL = "가족_총내방"
MART2_VALUE_COL = "최근소모량"
SNAPSHOT_DATE_COL = "기준일자"
WEIGHT_COL = "학습_가중치"

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
]
MART2_COLUMNS = [CUSTOMER_ID_COL, DRUG_ID_COL, MART2_VALUE_COL]
MART3_COLUMNS = ["약품ID", "계절", "요일", "소모량"]

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


class MartResult(NamedTuple):
    mart1: pd.DataFrame
    mart2: pd.DataFrame
    mart3: pd.DataFrame


def build_marts(raw_visits: pd.DataFrame, as_of_date) -> MartResult:
    """Build Mart 1/2/3 from raw visit-level rows, as of a given snapshot date.

    Pure and deterministic: reads only `as_of_date` (never wall-clock "today")
    and never mutates `raw_visits`, so the same inputs always produce the same
    outputs and the same entry point can be reused for training, backtesting,
    and live inference (see docs/adr/0002-point-in-time-correctness.md).
    """
    as_of_date = pd.Timestamp(as_of_date)
    chronic_customer_ids = _chronic_customer_ids(raw_visits)
    mart1 = _build_mart1(raw_visits, as_of_date, chronic_customer_ids)
    mart2 = _build_mart2(raw_visits, as_of_date)
    # Acute-patient visits (every visit whose customer isn't in
    # chronic_customer_ids) are Mart 3's population; the season x weekday
    # aggregation itself is a later ticket's job (see issue #9).
    mart3 = pd.DataFrame(columns=MART3_COLUMNS)
    return MartResult(mart1=mart1, mart2=mart2, mart3=mart3)


def revisit_match(raw_visits: pd.DataFrame) -> pd.Series:
    """The Revisit Match matcher (criterion H+T2, see ADR-0001).

    Returns a boolean Series indexed by 조제판매ID: True if some later visit
    by the same customer shares >=1 drug with this visit - excluding the
    dataset's top-2 highest-frequency drugs, to avoid false matches on
    near-universal OTC drugs - within +/-30 days of this visit's 다음내방일
    (expected next-visit date).
    """
    visits = _visits_with_drug_sets(raw_visits)
    excluded_drug_ids = _top_frequent_drug_ids(raw_visits, TOP_FREQUENT_DRUG_EXCLUDE_COUNT)
    visits[_CLEAN_DRUG_SET_COL] = visits[_DRUG_SET_COL].apply(
        lambda drugs: drugs - excluded_drug_ids
    )

    is_match = pd.Series(False, index=visits.index)
    for ordered in _visits_by_customer_ordered(visits):
        records = ordered.to_dict("records")
        row_positions = ordered.index.to_list()

        for i, current in enumerate(records):
            window_start = current[NEXT_EXPECTED_VISIT_COL] - pd.Timedelta(
                days=REVISIT_MATCH_WINDOW_DAYS
            )
            window_end = current[NEXT_EXPECTED_VISIT_COL] + pd.Timedelta(
                days=REVISIT_MATCH_WINDOW_DAYS
            )

            for later in records[i + 1 :]:
                if later[VISIT_DATE_COL] <= current[VISIT_DATE_COL]:
                    continue
                if not (window_start <= later[VISIT_DATE_COL] <= window_end):
                    continue
                if current[_CLEAN_DRUG_SET_COL] & later[_CLEAN_DRUG_SET_COL]:
                    is_match.loc[row_positions[i]] = True
                    break

    is_match.index = visits[VISIT_ID_COL]
    is_match.index.name = VISIT_ID_COL
    return is_match


def _visits_by_customer_ordered(visits: pd.DataFrame):
    """Yield each customer's visits (one group per 고객ID), sorted
    chronologically by 내방일 (stable, so same-day visits keep their
    original relative order)."""
    for _, group in visits.groupby(CUSTOMER_ID_COL, sort=False):
        yield group.sort_values(VISIT_DATE_COL, kind="stable")


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


def _visits_with_drug_sets(raw_visits: pd.DataFrame) -> pd.DataFrame:
    visits = _one_row_per_visit(
        raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL, NEXT_EXPECTED_VISIT_COL]
    )
    visits[_DRUG_SET_COL] = raw_visits.groupby(VISIT_ID_COL, sort=False)[DRUG_ID_COL].agg(
        lambda drug_ids: frozenset(drug_ids.dropna())
    )
    return visits.reset_index()


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


def _chronic_customer_ids(raw_visits: pd.DataFrame) -> set:
    """Customers with >=1 Revisit Match visit - Chronic Patients (see ADR-0001)."""
    if raw_visits.empty:
        return set()
    matches = revisit_match(raw_visits)
    matched_visit_ids = matches[matches].index
    visit_customers = _visits_with_drug_sets(raw_visits).set_index(VISIT_ID_COL)[
        CUSTOMER_ID_COL
    ]
    return set(visit_customers.loc[matched_visit_ids])


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

    Not yet wired into `build_marts`'s mart1 output -- a later ticket
    assembles this alongside Mart 1's other columns.
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

    Not yet wired into `build_marts`'s mart1 output - a later ticket
    assembles this alongside the Next-Day Visit positives into Mart 1's full
    historical training rows.
    """
    if chronic_customer_ids is None:
        chronic_customer_ids = _chronic_customer_ids(raw_visits)
    if not chronic_customer_ids:
        return pd.DataFrame(columns=MART1_NEGATIVE_SAMPLE_COLUMNS)

    visits = _one_row_per_visit(
        raw_visits, [CUSTOMER_ID_COL, VISIT_DATE_COL, PRESCRIPTION_DAYS_COL]
    )
    visits = visits[visits[CUSTOMER_ID_COL].isin(chronic_customer_ids)]

    rows = []
    for ordered in _visits_by_customer_ordered(visits):
        for visit_id, visit in ordered.iloc[:-1].iterrows():
            for snapshot_date in negative_sample_dates(
                visit[VISIT_DATE_COL], visit[PRESCRIPTION_DAYS_COL]
            ):
                rows.append(
                    {
                        CUSTOMER_ID_COL: visit[CUSTOMER_ID_COL],
                        VISIT_ID_COL: visit_id,
                        SNAPSHOT_DATE_COL: snapshot_date,
                        NEXT_DAY_VISIT_COL: False,
                    }
                )
    return pd.DataFrame(rows, columns=MART1_NEGATIVE_SAMPLE_COLUMNS)


def _build_mart1(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp, chronic_customer_ids: set
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
    return _attach_mart1_x_features(mart1, raw_visits, as_of_date)


def _attach_mart1_x_features(
    mart1: pd.DataFrame, raw_visits: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.DataFrame:
    """Joins in Mart 1's remaining X-features (see issue #7): demographics,
    as-of family loyalty, visit-timing/medication features anchored to each
    customer's most recent visit at or before `as_of_date`, and
    insurance/차상위 status. `mart1` must already have `고객ID`.
    """
    customer_attrs = _customer_attrs_as_of(raw_visits, as_of_date)
    anchoring_visits = _anchoring_visit_features(raw_visits, as_of_date)
    family_visit_counts = family_totals_as_of(raw_visits, as_of_date).set_index(
        FAMILY_ID_COL
    )[FAMILY_VISIT_COUNT_COL]
    mpr_and_no_show = _mpr_and_no_show_as_of(raw_visits, as_of_date)

    customer_ids = mart1[CUSTOMER_ID_COL]
    mart1 = mart1.copy()
    mart1[AGE_COL] = customer_ids.map(customer_attrs[AGE_COL])
    mart1[GENDER_COL] = customer_ids.map(customer_attrs[GENDER_COL])
    mart1[FAMILY_VISIT_COUNT_COL] = (
        customer_ids.map(customer_attrs[FAMILY_ID_COL]).map(family_visit_counts)
    )
    for col in ANCHORING_FEATURE_COLS:
        mart1[col] = customer_ids.map(anchoring_visits[col])
    # 차상위대상자 is a raw 'Y'/blank eligibility flag (same source table and
    # convention as the severity flags) — normalize to boolean the same way.
    mart1[NEAR_POVERTY_COL] = _eligibility_flag_mask(mart1[NEAR_POVERTY_COL])
    for col in MPR_NO_SHOW_FEATURE_COLS:
        mart1[col] = customer_ids.map(mpr_and_no_show[col])
    return mart1[MART1_COLUMNS]


def _customer_attrs_as_of(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Per 고객ID demographic/family attributes that don't vary by visit:
    성별, 나이 (computed as of `as_of_date` from 생년월일), and 가족ID (used
    to look up that customer's as-of family visit total)."""
    customers = _one_row_per_visit(
        raw_visits,
        [GENDER_COL, BIRTH_DATE_COL, FAMILY_ID_COL],
        group_col=CUSTOMER_ID_COL,
    )
    customers[AGE_COL] = _age_in_years(
        pd.to_datetime(customers[BIRTH_DATE_COL]), as_of_date
    )
    return customers


def _age_in_years(birth_dates: pd.Series, as_of_date: pd.Timestamp) -> pd.Series:
    """Whole-year age as of `as_of_date`, accounting for whether that year's
    birthday has already passed."""
    had_birthday_this_year = (birth_dates.dt.month < as_of_date.month) | (
        (birth_dates.dt.month == as_of_date.month) & (birth_dates.dt.day <= as_of_date.day)
    )
    return (
        as_of_date.year - birth_dates.dt.year - (~had_birthday_this_year).astype(int)
    )


def _visit_level_mart1_attrs(raw_visits: pd.DataFrame) -> pd.DataFrame:
    """One row per 조제판매ID with the visit-level attributes Mart 1's
    as-of features anchor to: customer identity, the visit's own 다음내방일,
    insurance/차상위 status, and its primary drug (the drug with the
    longest 투약일수 in that visit — the same drug 장기투약_일수 and
    주요_약품속명 both name, see CONTEXT.md's Mart 1 entry)."""
    visits = _one_row_per_visit(
        raw_visits,
        [
            CUSTOMER_ID_COL,
            VISIT_DATE_COL,
            NEXT_EXPECTED_VISIT_COL,
            INSURANCE_TYPE_COL,
            NEAR_POVERTY_COL,
        ],
    )
    primary_drug = raw_visits.groupby(VISIT_ID_COL, sort=False).apply(
        _primary_drug_for_visit, include_groups=False
    )
    return visits.join(primary_drug)


def _primary_drug_for_visit(visit_drugs: pd.DataFrame) -> pd.Series:
    """Given one visit's drug rows, the 투약일수/속명 of whichever drug has
    the longest 투약일수 -- undefined (NA) if every drug row is missing
    투약일수."""
    medication_days = pd.to_numeric(visit_drugs[MEDICATION_DAYS_COL], errors="coerce")
    if medication_days.notna().any():
        primary_row = medication_days.idxmax()
        return pd.Series(
            {
                LONG_TERM_MED_DAYS_COL: medication_days.loc[primary_row],
                PRIMARY_INGREDIENT_COL: visit_drugs.loc[primary_row, INGREDIENT_COL],
            }
        )
    return pd.Series({LONG_TERM_MED_DAYS_COL: pd.NA, PRIMARY_INGREDIENT_COL: pd.NA})


def _empty_customer_frame(columns: list[str]) -> pd.DataFrame:
    """An empty per-고객ID feature frame with the given columns -- the shared
    empty-result shape for as-of feature builders (see
    `_anchoring_visit_features` and `_mpr_and_no_show_as_of`) when no visit
    is eligible at all as of `as_of_date`."""
    return pd.DataFrame(columns=columns).set_index(pd.Index([], name=CUSTOMER_ID_COL))


def _anchoring_visit_features(
    raw_visits: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.DataFrame:
    """Per 고객ID, the visit-timing/medication/insurance features derived
    from that customer's most recent visit at or before `as_of_date` (their
    "anchoring visit").

    Uses `<=` (a visit dated exactly `as_of_date` is eligible to anchor),
    matching `_build_mart2`'s latest-consumption-value lookup -- this is
    "what do we know as of this snapshot" for a point-feature, not a
    cumulative count. That's a different question from `family_totals_as_of`
    using strict `<`: a cumulative total must exclude the row's own visit to
    avoid a visit counting itself (per docs/adr/0002-point-in-time-correctness.md,
    "up to but excluding that visit"), but there's no equivalent
    self-counting risk in picking which single visit anchors these features.

    A customer with no visit at or before `as_of_date` has no anchoring
    visit yet, so its features come back missing (NA) rather than computed
    from a future visit.
    """
    visits = _visit_level_mart1_attrs(raw_visits).reset_index()
    eligible = visits[visits[VISIT_DATE_COL] <= as_of_date]
    if eligible.empty:
        return _empty_customer_frame(ANCHORING_FEATURE_COLS)

    anchoring = (
        eligible.sort_values(VISIT_DATE_COL, kind="stable")
        .groupby(CUSTOMER_ID_COL, as_index=True, sort=False)
        .last()
    )
    anchoring[DAYS_SINCE_LAST_VISIT_COL] = (
        as_of_date - anchoring[VISIT_DATE_COL]
    ).dt.days
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
        as_of_date + pd.Timedelta(days=1) == anchoring[NEXT_EXPECTED_VISIT_COL]
    )
    return anchoring[ANCHORING_FEATURE_COLS]


def _mpr_and_no_show_as_of(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Per 고객ID, the MPR adherence score (복약_순응도) and no-show rate
    (노쇼_비율) computed purely from that customer's own visit history at or
    before `as_of_date` (see CONTEXT.md's Mart 1 entry and issue #8).

    Unlike `_anchoring_visit_features`, these aren't anchored to a single
    visit -- they aggregate over every one of the customer's visits with
    내방일 <= as_of_date.

    Returns one row per 고객ID (an empty frame if no visits are eligible)
    with columns MPR_NO_SHOW_FEATURE_COLS.
    """
    visits = _one_row_per_visit(
        raw_visits,
        [CUSTOMER_ID_COL, VISIT_DATE_COL, PRESCRIPTION_DAYS_COL, NEXT_EXPECTED_VISIT_COL],
    ).reset_index()
    eligible = visits[visits[VISIT_DATE_COL] <= as_of_date]
    if eligible.empty:
        return _empty_customer_frame(MPR_NO_SHOW_FEATURE_COLS)

    return eligible.groupby(CUSTOMER_ID_COL, sort=False).apply(
        lambda customer_visits: _mpr_and_no_show_for_customer(customer_visits, as_of_date),
        include_groups=False,
    )


def _mpr_and_no_show_for_customer(
    visits: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.Series:
    """MPR adherence score and no-show rate for one customer's eligible
    (내방일 <= as_of_date) visit history.

    MPR (복약_순응도), per docs/Research-Log.md's formula:
        Sigma(처방조제일수) / (최종내방일 - 최초내방일 + 최종조제일수) x 100
    복약_순응도 is NA whenever that denominator isn't a usable number --
    either the last eligible visit's 처방조제일수 is itself missing, or the
    denominator comes out to zero (a single eligible visit with
    처방조제일수 == 0) -- rather than propagating a raw NaN or dividing by
    zero.

    노쇼_비율 (no-show rate): the proportion of this customer's 다음내방일
    values that are themselves already resolvable as of `as_of_date` (i.e.
    다음내방일 <= as_of_date -- a later 다음내방일 hasn't happened yet, so we
    can't yet know whether it'll be kept) and that aren't matched by an
    actual visit dated exactly on that 다음내방일. NA if none of this
    customer's 다음내방일 values are resolvable yet -- there's no history to
    compute a rate from, not evidence of a 0% or 100% rate.
    """
    ordered = visits.sort_values(VISIT_DATE_COL, kind="stable")
    first_visit, last_visit = ordered.iloc[0], ordered.iloc[-1]
    last_prescription_days = last_visit[PRESCRIPTION_DAYS_COL]
    if pd.isna(last_prescription_days):
        mpr = pd.NA
    else:
        denominator = (
            last_visit[VISIT_DATE_COL] - first_visit[VISIT_DATE_COL]
        ).days + last_prescription_days
        mpr = (
            ordered[PRESCRIPTION_DAYS_COL].sum() / denominator * 100
            if denominator
            else pd.NA
        )

    resolvable = ordered[NEXT_EXPECTED_VISIT_COL].notna() & (
        ordered[NEXT_EXPECTED_VISIT_COL] <= as_of_date
    )
    if resolvable.any():
        actual_visit_dates = set(ordered[VISIT_DATE_COL])
        resolvable_next_dates = ordered.loc[resolvable, NEXT_EXPECTED_VISIT_COL]
        no_show_rate = (~resolvable_next_dates.isin(actual_visit_dates)).mean()
    else:
        no_show_rate = pd.NA

    return pd.Series({MPR_COL: mpr, NO_SHOW_RATE_COL: no_show_rate})


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
    prior_visits = visits[visits[VISIT_DATE_COL] < as_of_date]
    counts = prior_visits.groupby(FAMILY_ID_COL).size()
    families[FAMILY_VISIT_COUNT_COL] = (
        families[FAMILY_ID_COL].map(counts).fillna(0).astype(int)
    )
    return families


def _build_mart2(raw_visits: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Per 고객ID×약품ID, the latest single-visit consumption amount at or
    before `as_of_date` — never the global-latest value across the whole
    extract, per docs/adr/0002-point-in-time-correctness.md.
    """
    visit_dates = pd.to_datetime(raw_visits[VISIT_DATE_COL])
    eligible = raw_visits.loc[visit_dates <= as_of_date].dropna(subset=[DRUG_ID_COL])
    if eligible.empty:
        return pd.DataFrame(columns=MART2_COLUMNS)

    subset = eligible[[CUSTOMER_ID_COL, DRUG_ID_COL, VISIT_DATE_COL, CONSUMPTION_COL]]
    latest = (
        subset.sort_values(VISIT_DATE_COL, kind="stable")
        .groupby([CUSTOMER_ID_COL, DRUG_ID_COL], as_index=False, sort=False)
        .last()
    )
    mart2 = latest[[CUSTOMER_ID_COL, DRUG_ID_COL, CONSUMPTION_COL]].rename(
        columns={CONSUMPTION_COL: MART2_VALUE_COL}
    )
    return mart2.sort_values(
        [CUSTOMER_ID_COL, DRUG_ID_COL], kind="stable"
    ).reset_index(drop=True)
