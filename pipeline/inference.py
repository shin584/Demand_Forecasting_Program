"""Track 1 inference: scores every Chronic customer in a `build_marts`
snapshot and produces the pharmacist Visit List plus Track 1's per-drug
expected demand -- ordinary expected-value multiplication for most drugs,
the Chronic-population rare-drug allocation override for the rest. Also
combines that with Track 2's Mart 3 lookup into the final per-drug
order-quantity table.

See CONTEXT.md ("Visit List", "Safety Stock", "Track 1 Rare-Drug
Allocation") and docs/adr/0004-track1-rare-drug-population-and-allocation.md
for the design this encodes. Issue #20 built `run_track1_inference`; issue
#21 (this module's `run_daily_forecast`/`_combine_order_quantities`) added
Track 2's lookup and the combined table on top of its output; issue #22
added the Chronic-population rare-drug allocation override into
`_track1_drug_demand`, completing `run_daily_forecast`'s final behavior.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .marts import (
    CONSUMPTION_COL,
    CUSTOMER_ID_COL,
    DRUG_ID_COL,
    DRUG_NAME_COL,
    LAPSE_HORIZON_DAYS,
    MART2_VALUE_COL,
    MART3_BUCKET_MIN_OBSERVATIONS,
    MART3_SEASON_MIN_OBSERVATIONS,
    RARE_DRUG_PATIENT_THRESHOLD,
    SEASON_COL,
    SNAPSHOT_DATE_COL,
    WEEKDAY_COL,
    build_marts,
    chronic_rare_drug_ids,
    current_regimen,
    season_and_weekday_for,
)
from .model import CHRONIC_VISIT_PROB_CUTOFF, prepare_track1_features

# Track 1 inference's own output columns (see CONTEXT.md "Visit List").
VISIT_PROB_COL = "예측방문확률"
TRACK1_DEMAND_COL = "track1_기댓값"
VISIT_LIST_COLUMNS = [SNAPSHOT_DATE_COL, CUSTOMER_ID_COL, VISIT_PROB_COL]
# Every scored customer, not only those meeting the cutoff -- the Visit List
# is this frame filtered by `chronic_visit_prob_cutoff` (see issue #25).
SCORED_POPULATION_COLUMNS = VISIT_LIST_COLUMNS
TRACK1_DEMAND_COLUMNS = [SNAPSHOT_DATE_COL, DRUG_ID_COL, TRACK1_DEMAND_COL]

# The combined order-quantity table's own columns (see CONTEXT.md "Safety
# Stock" and issue #21/#17's settled schema).
TRACK2_STAT_COL = "track2_통계값"
FINAL_ORDER_COL = "최종발주량"
ORDER_QUANTITY_COLUMNS = [
    SNAPSHOT_DATE_COL,
    DRUG_ID_COL,
    DRUG_NAME_COL,
    TRACK1_DEMAND_COL,
    TRACK2_STAT_COL,
    FINAL_ORDER_COL,
]

# Safety-stock buffer applied to the combined Track 1 + Track 2 subtotal (see
# CONTEXT.md "Safety Stock"): a placeholder pending empirical tuning, not a
# fixed business requirement (see CONTEXT.md "Decision Thresholds
# (provisional)").
SAFETY_STOCK_BUFFER = 1.2


class Track1Result(NamedTuple):
    visit_list: pd.DataFrame
    drug_demand: pd.DataFrame
    scored_population: pd.DataFrame


class ForecastResult(NamedTuple):
    order_quantities: pd.DataFrame
    visit_list: pd.DataFrame
    scored_population: pd.DataFrame


def run_track1_inference(
    raw_visits: pd.DataFrame,
    as_of_date,
    model,
    chronic_visit_prob_cutoff: float = CHRONIC_VISIT_PROB_CUTOFF,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
) -> Track1Result:
    """Scores every Chronic customer in `build_marts`'s Mart 1 snapshot for
    `as_of_date` and returns the pharmacist Visit List alongside Track 1's
    per-drug expected demand.

    `model` is anything exposing a scikit-learn-shaped `predict_proba(X)` --
    a stub in tests, T1's real trained `LGBMClassifier` in production (see
    `pipeline.model.train_track1_model`); this function has no LightGBM-
    specific coupling. X-features are prepared the same way training does
    (`pipeline.model.prepare_track1_features`), so train/inference can't drift.

    Every customer in the snapshot is scored -- no pre-filtering beyond Mart
    1's own population: Chronic customers not Lapsed past
    `lapse_horizon_days` (see `build_marts`). The Visit List
    (`VISIT_LIST_COLUMNS`: 기준일자, 고객ID, 예측방문확률) contains exactly the
    customers whose predicted probability is `>= chronic_visit_prob_cutoff`;
    `scored_population` (`SCORED_POPULATION_COLUMNS`) is every scored
    customer with their probability, cutoff or not.

    `drug_demand` (`TRACK1_DEMAND_COLUMNS`: 기준일자, 약품ID, track1_기댓값) is
    the ordinary expected-value formula -- probability x Mart 2 latest
    consumption, summed per drug across every Chronic customer whose Current
    Regimen (the drugs on their Anchoring Visit, see
    `pipeline.marts.current_regimen`) includes it -- for most drugs. A drug
    with fewer than `rare_drug_patient_threshold` distinct Chronic patients
    (trailing 12 months, see `pipeline.marts.chronic_rare_drug_ids`) instead
    uses the 100%-allocation rule: the sum of each qualifying customer's
    full latest Mart 2 consumption for that drug, again only where it's in
    their Current Regimen, "qualifying" meaning predicted probability
    `>= chronic_visit_prob_cutoff` (see CONTEXT.md "Track 1 Rare-Drug
    Allocation" and docs/adr/0004). Every ordinary-drug customer contributes
    via the expected-value formula regardless of `chronic_visit_prob_cutoff`
    -- that cutoff only gates the Visit List and the rare-drug rule, not the
    ordinary sum, since expected value already scales a low-probability
    customer's contribution down rather than needing a hard cutoff.
    """
    as_of_date = pd.Timestamp(as_of_date)
    mart1, mart2, _mart3 = build_marts(
        raw_visits, as_of_date, lapse_horizon_days=lapse_horizon_days
    )
    return _track1_from_marts(
        mart1,
        mart2,
        raw_visits,
        as_of_date,
        model,
        chronic_visit_prob_cutoff,
        rare_drug_patient_threshold,
    )


def _track1_from_marts(
    mart1: pd.DataFrame,
    mart2: pd.DataFrame,
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    model,
    chronic_visit_prob_cutoff: float,
    rare_drug_patient_threshold: int,
) -> Track1Result:
    """`run_track1_inference`'s body, given an already-built Mart 1/Mart 2
    snapshot -- so `run_daily_forecast` can reuse its own `build_marts` call
    instead of building the same snapshot twice."""
    probabilities = _score_mart1(mart1, model)

    scored_population = _build_scored_population(
        mart1[CUSTOMER_ID_COL], probabilities, as_of_date
    )
    visit_list = _build_visit_list(scored_population, chronic_visit_prob_cutoff)
    drug_demand = _track1_drug_demand(
        mart1[CUSTOMER_ID_COL],
        probabilities,
        mart2,
        raw_visits,
        as_of_date,
        chronic_visit_prob_cutoff,
        rare_drug_patient_threshold,
    )
    return Track1Result(
        visit_list=visit_list, drug_demand=drug_demand, scored_population=scored_population
    )


def _score_mart1(mart1: pd.DataFrame, model) -> pd.Series:
    """Predicted visit probability per Mart 1 row (positive-class column of
    `model.predict_proba`), indexed the same as `mart1` itself."""
    features = prepare_track1_features(mart1)
    return pd.Series(model.predict_proba(features)[:, 1], index=mart1.index)


def _build_scored_population(
    customer_ids: pd.Series,
    probabilities: pd.Series,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    scored_population = pd.DataFrame(
        {
            SNAPSHOT_DATE_COL: as_of_date,
            CUSTOMER_ID_COL: customer_ids,
            VISIT_PROB_COL: probabilities,
        }
    )
    return scored_population.reset_index(drop=True)[SCORED_POPULATION_COLUMNS]


def _build_visit_list(
    scored_population: pd.DataFrame, chronic_visit_prob_cutoff: float
) -> pd.DataFrame:
    visit_list = scored_population[
        scored_population[VISIT_PROB_COL] >= chronic_visit_prob_cutoff
    ]
    return visit_list.reset_index(drop=True)[VISIT_LIST_COLUMNS]


def _track1_drug_demand(
    customer_ids: pd.Series,
    probabilities: pd.Series,
    mart2: pd.DataFrame,
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    chronic_visit_prob_cutoff: float,
    rare_drug_patient_threshold: int,
) -> pd.DataFrame:
    """Track 1's per-drug expected demand, summed per drug across every
    Chronic customer whose Current Regimen includes it (see CONTEXT.md
    "Current Regimen" and "Track 1 Rare-Drug Allocation"). Only Mart 2 rows
    for drugs on the customer's Anchoring Visit count: a drug they were
    dispensed earlier but not on that visit contributes nothing, under
    either formula below.

    Most drugs use the ordinary expected-value formula: probability x Mart 2
    latest consumption. A drug with fewer than `rare_drug_patient_threshold`
    distinct Chronic patients in the trailing 12 months (see
    `pipeline.marts.chronic_rare_drug_ids`) uses the 100%-allocation rule
    instead: the sum of each qualifying customer's full latest Mart 2
    consumption, where qualifying means predicted probability
    `>= chronic_visit_prob_cutoff` -- a customer below that cutoff
    contributes 0, not a scaled-down amount, and ordinary expected-value
    multiplication is never applied to a drug once it's below the threshold.

    `mart2` covers every customer (Chronic and Acute alike, per
    `build_marts`) and every drug they've ever been dispensed, but only
    Chronic customers -- the ones in `customer_ids`,
    Mart 1's own population -- have a predicted probability at all; an Acute
    customer's Mart 2 row maps to no probability (NaN) and is dropped below
    either way, so only Chronic consumption ever contributes here.
    """
    chronic_customer_ids = set(customer_ids)
    rare_drug_ids = chronic_rare_drug_ids(
        raw_visits, as_of_date, chronic_customer_ids, rare_drug_patient_threshold
    )

    regimen_consumption = mart2.merge(
        current_regimen(raw_visits, as_of_date), on=[CUSTOMER_ID_COL, DRUG_ID_COL]
    )
    probability_by_customer = pd.Series(probabilities.to_numpy(), index=customer_ids.to_numpy())
    matched_probability = regimen_consumption[CUSTOMER_ID_COL].map(probability_by_customer)
    is_rare_drug = regimen_consumption[DRUG_ID_COL].isin(rare_drug_ids)

    ordinary_demand = matched_probability * regimen_consumption[MART2_VALUE_COL]
    # NaN (rather than 0.0) below cutoff -- dropped by the dropna below, the
    # same "doesn't contribute at all" treatment an Acute customer's NaN
    # probability already gets, rather than a separate zero-value row.
    rare_allocation = regimen_consumption[MART2_VALUE_COL].where(
        matched_probability >= chronic_visit_prob_cutoff
    )
    per_row_demand = rare_allocation.where(is_rare_drug, ordinary_demand)

    demand = (
        pd.DataFrame({DRUG_ID_COL: regimen_consumption[DRUG_ID_COL], TRACK1_DEMAND_COL: per_row_demand})
        .dropna(subset=[TRACK1_DEMAND_COL])
        .groupby(DRUG_ID_COL, as_index=False)[TRACK1_DEMAND_COL]
        .sum()
    )
    demand.insert(0, SNAPSHOT_DATE_COL, as_of_date)
    return demand[TRACK1_DEMAND_COLUMNS]


def run_daily_forecast(
    raw_visits: pd.DataFrame,
    as_of_date,
    model,
    chronic_visit_prob_cutoff: float = CHRONIC_VISIT_PROB_CUTOFF,
    safety_stock_buffer: float = SAFETY_STOCK_BUFFER,
    mart3_bucket_min_observations: int = MART3_BUCKET_MIN_OBSERVATIONS,
    mart3_season_min_observations: int = MART3_SEASON_MIN_OBSERVATIONS,
    rare_drug_patient_threshold: int = RARE_DRUG_PATIENT_THRESHOLD,
    lapse_horizon_days: int = LAPSE_HORIZON_DAYS,
) -> ForecastResult:
    """The full daily forecast: Track 1's Visit List and per-drug demand
    (`run_track1_inference`) combined with Track 2's Mart 3 lookup into the
    final per-drug order-quantity table.

    `mart3_bucket_min_observations`/`mart3_season_min_observations` are
    forwarded to `build_marts` for Mart 3 only (see that function) -- they
    don't affect Track 1's own Mart 1/Mart 2-based computation. `rare_drug_patient_threshold` is forwarded to both:
    Mart 3's Acute-population rare-drug filter and Track 1's own
    Chronic-population rare-drug allocation override (see CONTEXT.md
    "Decision Thresholds (provisional)" and docs/adr/0004) -- one shared
    threshold *value*, applied independently to each track's own population.
    `lapse_horizon_days` is forwarded to `build_marts` and only narrows
    Track 1's Mart 1 population (see `run_track1_inference`).

    `order_quantities` (`ORDER_QUANTITY_COLUMNS`: 기준일자, 약품ID, 약품명,
    track1_기댓값, track2_통계값, 최종발주량) is the union of every drug
    appearing in Track 1's or Track 2's result, zero-filled on whichever side
    is absent, with 최종발주량 = (track1_기댓값 + track2_통계값) x
    `safety_stock_buffer` (see CONTEXT.md "Safety Stock"). `visit_list` and
    `scored_population` are passed through from Track 1 unchanged.
    """
    as_of_date = pd.Timestamp(as_of_date)
    # One snapshot serves both tracks: the Mart 3 thresholds passed here
    # don't touch Mart 1/Mart 2, so Track 1 sees exactly what its own
    # `build_marts` call (same `lapse_horizon_days`) would have built.
    mart1, mart2, mart3 = build_marts(
        raw_visits,
        as_of_date,
        mart3_bucket_min_observations=mart3_bucket_min_observations,
        mart3_season_min_observations=mart3_season_min_observations,
        rare_drug_patient_threshold=rare_drug_patient_threshold,
        lapse_horizon_days=lapse_horizon_days,
    )
    track1 = _track1_from_marts(
        mart1,
        mart2,
        raw_visits,
        as_of_date,
        model,
        chronic_visit_prob_cutoff,
        rare_drug_patient_threshold,
    )
    order_quantities = _combine_order_quantities(
        track1.drug_demand, mart3, raw_visits, as_of_date, safety_stock_buffer
    )
    return ForecastResult(
        order_quantities=order_quantities,
        visit_list=track1.visit_list,
        scored_population=track1.scored_population,
    )


def _combine_order_quantities(
    track1_drug_demand: pd.DataFrame,
    mart3: pd.DataFrame,
    raw_visits: pd.DataFrame,
    as_of_date: pd.Timestamp,
    safety_stock_buffer: float,
) -> pd.DataFrame:
    """Combines Track 1's per-drug demand with Track 2's Mart 3 lookup for
    the target date (`as_of_date + 1 day`) into `ORDER_QUANTITY_COLUMNS`.

    A zero-filled outer union on 약품ID: a drug missing from one track
    contributes 0 to that track's column rather than dropping the row, so no
    drug that either track has something to say about is silently omitted.
    """
    target_date = as_of_date + pd.Timedelta(days=1)
    track2_drug_demand = _track2_drug_demand(mart3, target_date)

    combined = track1_drug_demand[[DRUG_ID_COL, TRACK1_DEMAND_COL]].merge(
        track2_drug_demand, on=DRUG_ID_COL, how="outer"
    )
    combined[TRACK1_DEMAND_COL] = combined[TRACK1_DEMAND_COL].fillna(0.0)
    combined[TRACK2_STAT_COL] = combined[TRACK2_STAT_COL].fillna(0.0)
    combined[FINAL_ORDER_COL] = (
        combined[TRACK1_DEMAND_COL] + combined[TRACK2_STAT_COL]
    ) * safety_stock_buffer

    combined[DRUG_NAME_COL] = combined[DRUG_ID_COL].map(_resolve_drug_names(raw_visits))
    combined[SNAPSHOT_DATE_COL] = as_of_date
    return combined.sort_values(DRUG_ID_COL, kind="stable").reset_index(drop=True)[
        ORDER_QUANTITY_COLUMNS
    ]


def _track2_drug_demand(mart3: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    """Track 2's per-drug statistical estimate (`TRACK2_STAT_COL`): Mart 3's
    existing season x weekday backoff grid, looked up at `target_date`'s own
    (계절, 요일) bucket -- no new statistical logic, only a lookup (see
    CONTEXT.md "Track 2 Sparse-Bucket Backoff")."""
    season, weekday = season_and_weekday_for(target_date)
    bucket = mart3.loc[(mart3[SEASON_COL] == season) & (mart3[WEEKDAY_COL] == weekday)]
    demand = bucket[[DRUG_ID_COL, CONSUMPTION_COL]].rename(columns={CONSUMPTION_COL: TRACK2_STAT_COL})
    # `mart3` may be resolve_mart3_backoff's untyped empty frame (no drugs at
    # all) -- pin the dtype so downstream fillna never downcasts from object.
    return demand.astype({TRACK2_STAT_COL: float})


def _resolve_drug_names(raw_visits: pd.DataFrame) -> pd.Series:
    """Per 약품ID, that drug's 약품명 from any matching `raw_visits` row (the
    first one found, per issue #21's acceptance criteria) -- there's no
    separate drug-master table in this repo, so the flat extract's own 약품명
    column is the only source.

    Deliberately not as-of-date filtered, unlike every other lookup in this
    module: a drug's name isn't a point-in-time-sensitive fact the way
    consumption or visit-probability figures are, so there's no leakage risk
    in resolving it from the full `raw_visits` table."""
    named = raw_visits.dropna(subset=[DRUG_ID_COL]).drop_duplicates(subset=[DRUG_ID_COL])
    return named.set_index(DRUG_ID_COL)[DRUG_NAME_COL]
