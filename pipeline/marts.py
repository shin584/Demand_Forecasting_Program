"""Mart-construction pipeline: turns raw extracted visit-level rows into
Mart 1 (Visit-Probability), Mart 2 (Customer Drug Profile) and Mart 3
(Acute Drug Statistics).

See CONTEXT.md and docs/adr/ for the design decisions this pipeline encodes.
This module currently implements the pipeline scaffolding, Mart 1's Y label,
Revisit Match-based Chronic/Acute routing, Mart 2's as-of-date consumption
values, and as-of-date family visit totals; later tickets fill in the
remaining columns of all three marts.
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

# Internal-only columns, not part of the raw_visits schema.
_DRUG_SET_COL = "_drug_set"
_CLEAN_DRUG_SET_COL = "_clean_drug_set"

MART1_COLUMNS = [CUSTOMER_ID_COL, NEXT_DAY_VISIT_COL, CHRONIC_COL]
MART2_COLUMNS = [CUSTOMER_ID_COL, DRUG_ID_COL, MART2_VALUE_COL]
MART3_COLUMNS = ["약품ID", "계절", "요일", "소모량"]

# Revisit Match criterion H+T2 (see docs/adr/0001-chronic-patient-behavioral-definition.md).
REVISIT_MATCH_WINDOW_DAYS = 30
TOP_FREQUENT_DRUG_EXCLUDE_COUNT = 2

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


def _one_row_per_visit(raw_visits: pd.DataFrame, cols: list) -> pd.DataFrame:
    """One row per 조제판매ID (a visit may dispense several drugs, one raw row
    each), keeping the first value of each of `cols`."""
    return raw_visits.groupby(VISIT_ID_COL, sort=False).agg(
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
    customers = (
        raw_visits.loc[chronic_mask, CUSTOMER_ID_COL]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    # 내일_방문 is the literal "did the customer actually show up on
    # as_of_date + 1" boolean — independent of Revisit Match, which is a
    # separate behavioral definition used only for chronic/acute routing.
    # A visit dated exactly as_of_date + 1 day is necessarily that customer's
    # *next* visit after as_of_date, since no calendar day falls between the
    # two — this equivalence would not hold if target_date were more than one
    # day out.
    visited_next_day = set(raw_visits.loc[visit_dates == target_date, CUSTOMER_ID_COL])

    return pd.DataFrame(
        {
            CUSTOMER_ID_COL: customers,
            NEXT_DAY_VISIT_COL: customers.isin(visited_next_day),
            # Derived solely from Revisit Match — no diagnosis-code or
            # duration-based path exists (redundant with the inclusion
            # filter above, kept explicit as the feature's own definition).
            CHRONIC_COL: customers.isin(chronic_customer_ids),
        }
    )


def family_totals_as_of(raw_visits: pd.DataFrame, as_of_date) -> pd.DataFrame:
    """가족_총내방 (family cumulative visit count) computed as of `as_of_date`.

    For each 가족ID, counts that family's distinct visits (조제판매ID) whose
    내방일 falls strictly before `as_of_date` — cumulative up to but excluding
    as_of_date itself, per docs/adr/0002-point-in-time-correctness.md — never
    the live `tbl가족총매출` snapshot already sitting on raw_visits.

    가족_총매출 (family revenue) is deliberately not computed here: raw_visits
    carries no per-visit monetary amount, only drug consumption quantities in
    units that aren't comparable across drugs, so there's no correct way to
    recompute it from the columns build_marts receives today. See issue #10.

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
