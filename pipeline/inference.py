"""Track 1 inference: scores every Chronic customer in a `build_marts`
snapshot and produces the pharmacist Visit List plus Track 1's per-drug
expected demand for ordinary (non-rare) drugs. Also combines that with
Track 2's Mart 3 lookup into the final per-drug order-quantity table.

See CONTEXT.md ("Visit List", "Safety Stock", "Track 1 Rare-Drug
Allocation") and docs/adr/0004-track1-rare-drug-population-and-allocation.md
for the design this encodes. Issue #20 built `run_track1_inference`; issue
#21 (this module's `run_daily_forecast`/`_combine_order_quantities`) adds
Track 2's lookup and the combined table on top of its output. The
Chronic-population rare-drug allocation override (issue #22) still applies
the ordinary expected-value formula to every drug, including ones below the
Chronic-population rare-drug cutoff -- that override slots into
`run_track1_inference`/`_track1_drug_demand` separately.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .marts import (
    CONSUMPTION_COL,
    CUSTOMER_ID_COL,
    DRUG_ID_COL,
    DRUG_NAME_COL,
    MART2_VALUE_COL,
    MART3_BUCKET_MIN_OBSERVATIONS,
    MART3_SEASON_MIN_OBSERVATIONS,
    RARE_DRUG_PATIENT_THRESHOLD,
    SEASON_COL,
    SNAPSHOT_DATE_COL,
    WEEKDAY_COL,
    build_marts,
    season_and_weekday_for,
)
from .model import CHRONIC_VISIT_PROB_CUTOFF, prepare_track1_features

# Track 1 inference's own output columns (see CONTEXT.md "Visit List").
VISIT_PROB_COL = "예측방문확률"
TRACK1_DEMAND_COL = "track1_기댓값"
VISIT_LIST_COLUMNS = [SNAPSHOT_DATE_COL, CUSTOMER_ID_COL, VISIT_PROB_COL]
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


class ForecastResult(NamedTuple):
    order_quantities: pd.DataFrame
    visit_list: pd.DataFrame


def run_track1_inference(
    raw_visits: pd.DataFrame,
    as_of_date,
    model,
    chronic_visit_prob_cutoff: float = CHRONIC_VISIT_PROB_CUTOFF,
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
    1's own Chronic population (see `build_marts`). The Visit List
    (`VISIT_LIST_COLUMNS`: 기준일자, 고객ID, 예측방문확률) contains exactly the
    customers whose predicted probability is `>= chronic_visit_prob_cutoff`.

    `drug_demand` (`TRACK1_DEMAND_COLUMNS`: 기준일자, 약품ID, track1_기댓값) is
    the ordinary expected-value formula -- probability x Mart 2 latest
    consumption, summed per drug across every Chronic customer who has a
    Mart 2 row for it -- applied here to every drug, since the
    Chronic-population rare-drug override (docs/adr/0004) isn't implemented
    yet (see issue #22, which replaces this per-drug for drugs below that
    cutoff). Every customer contributes via the expected-value formula
    regardless of `chronic_visit_prob_cutoff` -- that cutoff only gates the
    Visit List, not this sum, since expected value already scales a
    low-probability customer's contribution down rather than needing a hard
    cutoff.
    """
    as_of_date = pd.Timestamp(as_of_date)
    mart1, mart2, _mart3 = build_marts(raw_visits, as_of_date)

    probabilities = _score_mart1(mart1, model)

    visit_list = _build_visit_list(
        mart1[CUSTOMER_ID_COL], probabilities, as_of_date, chronic_visit_prob_cutoff
    )
    drug_demand = _track1_drug_demand(mart1[CUSTOMER_ID_COL], probabilities, mart2, as_of_date)
    return Track1Result(visit_list=visit_list, drug_demand=drug_demand)


def _score_mart1(mart1: pd.DataFrame, model) -> pd.Series:
    """Predicted visit probability per Mart 1 row (positive-class column of
    `model.predict_proba`), indexed the same as `mart1` itself."""
    features = prepare_track1_features(mart1)
    return pd.Series(model.predict_proba(features)[:, 1], index=mart1.index)


def _build_visit_list(
    customer_ids: pd.Series,
    probabilities: pd.Series,
    as_of_date: pd.Timestamp,
    chronic_visit_prob_cutoff: float,
) -> pd.DataFrame:
    visit_list = pd.DataFrame(
        {
            SNAPSHOT_DATE_COL: as_of_date,
            CUSTOMER_ID_COL: customer_ids,
            VISIT_PROB_COL: probabilities,
        }
    )
    visit_list = visit_list[visit_list[VISIT_PROB_COL] >= chronic_visit_prob_cutoff]
    return visit_list.reset_index(drop=True)[VISIT_LIST_COLUMNS]


def _track1_drug_demand(
    customer_ids: pd.Series,
    probabilities: pd.Series,
    mart2: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Track 1's ordinary per-drug expected demand (see CONTEXT.md "Track 1
    Rare-Drug Allocation"): probability x Mart 2 latest consumption, summed
    per drug.

    `mart2` covers every customer (Chronic and Acute alike, per
    `build_marts`), but only Chronic customers -- the ones in `customer_ids`,
    Mart 1's own population -- have a predicted probability at all; an Acute
    customer's Mart 2 row maps to no probability and is dropped below, so
    only Chronic consumption ever contributes here.
    """
    probability_by_customer = pd.Series(probabilities.to_numpy(), index=customer_ids.to_numpy())
    matched_probability = mart2[CUSTOMER_ID_COL].map(probability_by_customer)
    expected_demand = matched_probability * mart2[MART2_VALUE_COL]

    demand = (
        pd.DataFrame({DRUG_ID_COL: mart2[DRUG_ID_COL], TRACK1_DEMAND_COL: expected_demand})
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
) -> ForecastResult:
    """The full daily forecast: Track 1's Visit List and per-drug demand
    (`run_track1_inference`) combined with Track 2's Mart 3 lookup into the
    final per-drug order-quantity table.

    `mart3_bucket_min_observations`/`mart3_season_min_observations`/
    `rare_drug_patient_threshold` are forwarded to `build_marts` for Mart 3
    (see that function) -- they don't affect Track 1's own Mart 1/Mart 2-based
    computation, which `run_track1_inference` still derives via its own
    (default-threshold) `build_marts` call.

    `order_quantities` (`ORDER_QUANTITY_COLUMNS`: 기준일자, 약품ID, 약품명,
    track1_기댓값, track2_통계값, 최종발주량) is the union of every drug
    appearing in Track 1's or Track 2's result, zero-filled on whichever side
    is absent, with 최종발주량 = (track1_기댓값 + track2_통계값) x
    `safety_stock_buffer` (see CONTEXT.md "Safety Stock").
    """
    as_of_date = pd.Timestamp(as_of_date)
    track1 = run_track1_inference(raw_visits, as_of_date, model, chronic_visit_prob_cutoff)
    _mart1, _mart2, mart3 = build_marts(
        raw_visits,
        as_of_date,
        mart3_bucket_min_observations,
        mart3_season_min_observations,
        rare_drug_patient_threshold,
    )
    order_quantities = _combine_order_quantities(
        track1.drug_demand, mart3, raw_visits, as_of_date, safety_stock_buffer
    )
    return ForecastResult(order_quantities=order_quantities, visit_list=track1.visit_list)


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
