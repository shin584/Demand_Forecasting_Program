import numpy as np
import pandas as pd

from conftest import (
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.inference import (
    TRACK1_DEMAND_COL,
    TRACK1_DEMAND_COLUMNS,
    VISIT_LIST_COLUMNS,
    VISIT_PROB_COL,
    run_track1_inference,
)
from pipeline.marts import CUSTOMER_ID_COL, DRUG_ID_COL, SNAPSHOT_DATE_COL


class StubModel:
    """A minimal `predict_proba`-only stand-in for T1's real LightGBM model
    (see issue #20's "no coupling to LightGBM internals in tests"
    acceptance criterion): returns a fixed probability per row, in the same
    row order `run_track1_inference` feeds it Mart 1's customers -- which is
    ascending 고객ID order (see `_distinct_customer_ids`), so a test only
    needs to know its fixture's chronic customer IDs' sort order, never any
    actual feature values.
    """

    def __init__(self, probabilities: list[float]):
        self._probabilities = np.array(probabilities)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert len(X) == len(self._probabilities)
        return np.column_stack([1 - self._probabilities, self._probabilities])


def _two_chronic_customers_with_drug_consumption(
    as_of_date: str, drug_id: int, consumption_1: float, consumption_2: float
) -> pd.DataFrame:
    """Two independently-established Chronic customers (1 and 2, see
    ADR-0001), each with one additional dated `as_of_date` visit consuming
    `drug_id` at their own consumption amount -- Mart 2's input for that
    drug."""
    return make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=101, visit_id_start=10)
        + make_independent_chronic_match_visits(customer_id=2, drug_id=102, visit_id_start=12)
        + [
            make_visit_row(
                조제판매ID=100,
                고객ID=1,
                내방일=as_of_date,
                약품ID=drug_id,
                소모량=consumption_1,
            ),
            make_visit_row(
                조제판매ID=101,
                고객ID=2,
                내방일=as_of_date,
                약품ID=drug_id,
                소모량=consumption_2,
            ),
        ]
    )


def test_scores_every_chronic_customer_with_no_prefiltering():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.1, 0.2])  # both below any real cutoff

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.0
    )

    # Both Chronic customers got scored and appear once cutoff admits everyone.
    assert sorted(result.visit_list[CUSTOMER_ID_COL]) == [1, 2]


def test_visit_list_contains_exactly_customers_at_or_above_cutoff():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.5])  # customer 1 below, customer 2 at/above cutoff

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert list(result.visit_list[CUSTOMER_ID_COL]) == [2]


def test_visit_list_reports_the_predicted_probability():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.5])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert result.visit_list[VISIT_PROB_COL].iloc[0] == 0.5


def test_visit_list_columns_and_snapshot_date():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.5])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert list(result.visit_list.columns) == VISIT_LIST_COLUMNS
    assert (result.visit_list[SNAPSHOT_DATE_COL] == pd.Timestamp("2024-01-01")).all()


def test_visit_list_admits_a_customer_exactly_at_the_cutoff():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.3, 0.1])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert list(result.visit_list[CUSTOMER_ID_COL]) == [1]


def test_track1_drug_demand_is_probability_weighted_consumption_summed_per_drug():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.8])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    demand = result.drug_demand.set_index(DRUG_ID_COL)[TRACK1_DEMAND_COL]
    # 0.2*40 + 0.8*20 = 24.0 -- summed across both customers regardless of
    # either one's individual cutoff standing (see Research-Log.md's
    # "방문 확률이 있는 모든 고객의 기댓값을 약품별로 합산").
    assert demand.loc[501] == 24.0


def test_track1_drug_demand_excludes_acute_customers_entirely():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=101, visit_id_start=10)
        + [
            make_visit_row(
                조제판매ID=100, 고객ID=1, 내방일="2024-01-01", 약품ID=501, 소모량=40.0
            ),
            # Customer 2 has only this single visit -- Acute, never scored
            # by Track 1 -- consuming a drug no Chronic customer touches.
            make_visit_row(
                조제판매ID=200, 고객ID=2, 내방일="2024-01-01", 약품ID=777, 소모량=999.0
            ),
        ]
    )
    model = StubModel([0.5])  # one row: only customer 1 is Chronic

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert 777 not in set(result.drug_demand[DRUG_ID_COL])
    assert 501 in set(result.drug_demand[DRUG_ID_COL])


def test_drug_demand_output_columns_and_snapshot_date():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.8])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert list(result.drug_demand.columns) == TRACK1_DEMAND_COLUMNS
    assert (result.drug_demand[SNAPSHOT_DATE_COL] == pd.Timestamp("2024-01-01")).all()


def test_default_cutoff_matches_chronic_visit_prob_cutoff():
    from pipeline.model import CHRONIC_VISIT_PROB_CUTOFF

    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    just_below = CHRONIC_VISIT_PROB_CUTOFF - 0.01
    just_at = CHRONIC_VISIT_PROB_CUTOFF
    model = StubModel([just_below, just_at])

    result = run_track1_inference(raw_visits, as_of_date="2024-01-01", model=model)

    assert list(result.visit_list[CUSTOMER_ID_COL]) == [2]
