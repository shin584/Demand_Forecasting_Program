"""Track 1 inference: scores every Chronic customer in a `build_marts`
snapshot and produces the pharmacist Visit List plus Track 1's per-drug
expected demand for ordinary (non-rare) drugs.

See CONTEXT.md ("Visit List", "Track 1 Rare-Drug Allocation") and
docs/adr/0004-track1-rare-drug-population-and-allocation.md for the design
this encodes, and issue #20 for this module's own scope -- Track 2's Mart 3
lookup and the final combined order-quantity table (issue #21), and the
Chronic-population rare-drug allocation override (issue #22), are built
separately on top of `run_track1_inference`'s output.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from .marts import CUSTOMER_ID_COL, DRUG_ID_COL, MART2_VALUE_COL, SNAPSHOT_DATE_COL, build_marts
from .model import CHRONIC_VISIT_PROB_CUTOFF, prepare_track1_features

# Track 1 inference's own output columns (see CONTEXT.md "Visit List" and
# issue #17's order-quantity table schema -- 약품명/track2_통계값/최종발주량
# are added on top of TRACK1_DEMAND_COLUMNS by issue #21, not here).
VISIT_PROB_COL = "예측방문확률"
TRACK1_DEMAND_COL = "track1_기댓값"
VISIT_LIST_COLUMNS = [SNAPSHOT_DATE_COL, CUSTOMER_ID_COL, VISIT_PROB_COL]
TRACK1_DEMAND_COLUMNS = [SNAPSHOT_DATE_COL, DRUG_ID_COL, TRACK1_DEMAND_COL]


class Track1Result(NamedTuple):
    visit_list: pd.DataFrame
    drug_demand: pd.DataFrame


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
