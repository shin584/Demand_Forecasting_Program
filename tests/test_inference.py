import numpy as np
import pandas as pd
import pytest

from conftest import (
    make_high_frequency_filler_visits,
    make_independent_chronic_match_visits,
    make_raw_visits,
    make_visit_row,
)
from pipeline.inference import (
    FINAL_ORDER_COL,
    ORDER_QUANTITY_COLUMNS,
    SAFETY_STOCK_BUFFER,
    SCORED_POPULATION_COLUMNS,
    TRACK1_DEMAND_COL,
    TRACK1_DEMAND_COLUMNS,
    TRACK2_STAT_COL,
    VISIT_LIST_COLUMNS,
    VISIT_PROB_COL,
    run_daily_forecast,
    run_track1_inference,
)
from pipeline.marts import (
    CUSTOMER_ID_COL,
    DRUG_ID_COL,
    DRUG_NAME_COL,
    SNAPSHOT_DATE_COL,
)


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


def _n_chronic_customers_with_drug_consumption(
    as_of_date: str, drug_id: int, consumptions: list[float]
) -> pd.DataFrame:
    """`len(consumptions)` independently-established Chronic customers (see
    ADR-0001), each with one additional dated `as_of_date` visit consuming
    `drug_id` at their own consumption amount -- Mart 2's input for that
    drug. Generalizes `_two_chronic_customers_with_drug_consumption` to an
    arbitrary customer count, needed to straddle the Chronic-population
    rare-drug threshold (see docs/adr/0004-track1-rare-drug-population-and-
    allocation.md and issue #22)."""
    rows = make_high_frequency_filler_visits()
    for i, consumption in enumerate(consumptions, start=1):
        rows += make_independent_chronic_match_visits(
            customer_id=i, drug_id=1000 + i, visit_id_start=10 * i
        )
        rows.append(
            make_visit_row(
                조제판매ID=1000 + i,
                고객ID=i,
                내방일=as_of_date,
                약품ID=drug_id,
                소모량=consumption,
            )
        )
    return make_raw_visits(rows)


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


def test_scored_population_lists_every_scored_customer_regardless_of_cutoff():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.5])

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    scored = result.scored_population
    assert list(scored.columns) == SCORED_POPULATION_COLUMNS
    assert scored.set_index(CUSTOMER_ID_COL)[VISIT_PROB_COL].to_dict() == {1: 0.2, 2: 0.5}
    assert (scored[SNAPSHOT_DATE_COL] == pd.Timestamp("2024-01-01")).all()


def test_daily_forecast_passes_scored_population_through_unchanged():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )

    track1 = run_track1_inference(raw_visits, as_of_date="2024-01-01", model=StubModel([0.2, 0.8]))
    result = run_daily_forecast(raw_visits, as_of_date="2024-01-01", model=StubModel([0.2, 0.8]))

    pd.testing.assert_frame_equal(result.scored_population, track1.scored_population)


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

    # rare_drug_patient_threshold=1: only 2 Chronic patients ever touch drug
    # 501 in this fixture, so the default threshold (5) would otherwise route
    # this drug into the rare-drug allocation rule (issue #22) instead of the
    # ordinary formula this test exercises.
    result = run_track1_inference(
        raw_visits,
        as_of_date="2024-01-01",
        model=model,
        chronic_visit_prob_cutoff=0.3,
        rare_drug_patient_threshold=1,
    )

    demand = result.drug_demand.set_index(DRUG_ID_COL)[TRACK1_DEMAND_COL]
    # 0.2*40 + 0.8*20 = 24.0 -- summed across both customers regardless of
    # either one's individual cutoff standing (see Research-Log.md's
    # "방문 확률이 있는 모든 고객의 기댓값을 약품별로 합산").
    assert demand.loc[501] == 24.0


# --- Chronic-population rare-drug allocation override (issue #22, see
# docs/adr/0004-track1-rare-drug-population-and-allocation.md). ---


def test_rare_drug_allocation_sums_qualifying_customers_full_consumption():
    raw_visits = _n_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumptions=[40.0, 20.0, 10.0]
    )
    # 3 Chronic patients ever touch drug 501 -- below the default rare-drug
    # threshold of 5 (see docs/adr/0004), so the 100%-allocation rule applies
    # instead of ordinary probability-weighted multiplication.
    model = StubModel([0.1, 0.5, 0.9])  # only customers 2 and 3 clear 0.3

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    demand = result.drug_demand.set_index(DRUG_ID_COL)[TRACK1_DEMAND_COL]
    # Full latest consumption summed for qualifying customers 2 (20.0) and 3
    # (10.0) only -- customer 1 (below cutoff) contributes 0, not a scaled
    # amount, and the qualifying values are summed (30.0), not averaged
    # (15.0) or maxed (20.0).
    assert demand.loc[501] == 30.0


def test_rare_drug_allocation_never_applies_probability_weighting():
    raw_visits = _n_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumptions=[40.0, 20.0, 10.0]
    )
    model = StubModel([0.31, 0.9, 0.9])  # all three clear the cutoff

    result = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    demand = result.drug_demand.set_index(DRUG_ID_COL)[TRACK1_DEMAND_COL]
    # If ordinary expected-value multiplication were (wrongly) still applied,
    # this would come out to 0.31*40 + 0.9*20 + 0.9*10 = 39.4, not 70.0 --
    # the full, unweighted sum across every qualifying customer.
    assert demand.loc[501] == 70.0


def test_drug_at_exactly_the_rare_drug_threshold_uses_ordinary_expected_value():
    raw_visits = _n_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumptions=[40.0, 20.0, 10.0]
    )
    model = StubModel([0.2, 0.5, 0.9])

    result = run_track1_inference(
        raw_visits,
        as_of_date="2024-01-01",
        model=model,
        chronic_visit_prob_cutoff=0.3,
        rare_drug_patient_threshold=3,  # exactly 3 patients -- not below 3
    )

    demand = result.drug_demand.set_index(DRUG_ID_COL)[TRACK1_DEMAND_COL]
    # Ordinary expected-value formula: 0.2*40 + 0.5*20 + 0.9*10 = 27.0 --
    # CONTEXT.md's rare-drug cutoff is strictly-less-than, so a drug at
    # exactly the threshold is not rare.
    assert demand.loc[501] == pytest.approx(27.0)


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


# --- run_daily_forecast: combines Track 1's drug demand with Track 2's Mart
# 3 lookup into the final order-quantity table (issue #21). ---


def _acute_observation_visits(drug_id: int, visit_date: str, consumption: float) -> pd.DataFrame:
    """Filler visits (so no real drug gets swept into the Revisit Match
    top-2 exclusion) plus a single Acute customer's (no revisit match)
    observation of `drug_id` -- Mart 3's input for that drug."""
    return make_raw_visits(
        make_high_frequency_filler_visits()
        + [
            make_visit_row(
                조제판매ID=1, 고객ID=1, 내방일=visit_date, 약품ID=drug_id, 소모량=consumption
            )
        ]
    )


def test_order_quantities_columns_and_snapshot_date():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.8])

    result = run_daily_forecast(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    assert list(result.order_quantities.columns) == ORDER_QUANTITY_COLUMNS
    assert (result.order_quantities[SNAPSHOT_DATE_COL] == pd.Timestamp("2024-01-01")).all()


def test_drug_with_only_a_track2_contribution_has_zero_track1_demand():
    # A single Acute customer (no Chronic customers at all) observed on a
    # Monday in January -- 겨울/월요일, the target date's own bucket below.
    raw_visits = _acute_observation_visits(drug_id=501, visit_date="2024-01-08", consumption=42.0)
    model = StubModel([])  # no Chronic customers to score

    result = run_daily_forecast(
        raw_visits,
        as_of_date="2024-01-14",  # target date 2024-01-15 is also a Monday
        model=model,
        rare_drug_patient_threshold=1,
        mart3_bucket_min_observations=1,
    )

    order = result.order_quantities.set_index(DRUG_ID_COL)
    assert order.loc[501, TRACK1_DEMAND_COL] == 0.0
    assert order.loc[501, TRACK2_STAT_COL] == 42.0


def test_drug_with_only_a_track1_contribution_has_zero_track2_estimate():
    # 약품ID=501 here is only ever consumed by Chronic customers, so it never
    # enters Mart 3's Acute-only population at all.
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.8])

    # rare_drug_patient_threshold=1: only 2 Chronic patients ever touch drug
    # 501 in this fixture, so the default threshold (5) would otherwise route
    # this drug into the rare-drug allocation rule (issue #22) instead of the
    # ordinary formula this test exercises.
    result = run_daily_forecast(
        raw_visits,
        as_of_date="2024-01-01",
        model=model,
        chronic_visit_prob_cutoff=0.3,
        rare_drug_patient_threshold=1,
    )

    order = result.order_quantities.set_index(DRUG_ID_COL)
    assert order.loc[501, TRACK1_DEMAND_COL] == 24.0  # 0.2*40 + 0.8*20
    assert order.loc[501, TRACK2_STAT_COL] == 0.0


def test_final_order_quantity_is_buffered_sum_of_both_tracks():
    raw_visits = make_raw_visits(
        make_high_frequency_filler_visits()
        + make_independent_chronic_match_visits(customer_id=1, drug_id=101, visit_id_start=10)
        + make_independent_chronic_match_visits(customer_id=2, drug_id=102, visit_id_start=12)
        + [
            make_visit_row(
                조제판매ID=100, 고객ID=1, 내방일="2024-01-14", 약품ID=501, 소모량=40.0
            ),
            make_visit_row(
                조제판매ID=101, 고객ID=2, 내방일="2024-01-14", 약품ID=501, 소모량=20.0
            ),
            # Acute customer (no revisit match) buys the same drug, landing
            # in the target date's own 겨울/월요일 bucket.
            make_visit_row(
                조제판매ID=102, 고객ID=3, 내방일="2024-01-08", 약품ID=501, 소모량=42.0
            ),
        ]
    )
    model = StubModel([0.2, 0.8])

    result = run_daily_forecast(
        raw_visits,
        as_of_date="2024-01-14",
        model=model,
        chronic_visit_prob_cutoff=0.3,
        rare_drug_patient_threshold=1,
        mart3_bucket_min_observations=1,
        safety_stock_buffer=2.0,
    )

    order = result.order_quantities.set_index(DRUG_ID_COL)
    track1_value = order.loc[501, TRACK1_DEMAND_COL]
    track2_value = order.loc[501, TRACK2_STAT_COL]
    assert track1_value == 24.0  # 0.2*40 + 0.8*20
    assert track2_value == 42.0
    assert order.loc[501, FINAL_ORDER_COL] == pytest.approx((track1_value + track2_value) * 2.0)


def test_default_safety_stock_buffer_is_1_2():
    assert SAFETY_STOCK_BUFFER == 1.2


def test_drug_name_resolved_from_raw_visits():
    raw_visits = _acute_observation_visits(drug_id=501, visit_date="2024-01-08", consumption=42.0)
    raw_visits.loc[raw_visits[DRUG_ID_COL] == 501, DRUG_NAME_COL] = "감기약"
    model = StubModel([])

    result = run_daily_forecast(
        raw_visits,
        as_of_date="2024-01-14",
        model=model,
        rare_drug_patient_threshold=1,
        mart3_bucket_min_observations=1,
    )

    order = result.order_quantities.set_index(DRUG_ID_COL)
    assert order.loc[501, DRUG_NAME_COL] == "감기약"


def test_visit_list_is_passed_through_unchanged():
    raw_visits = _two_chronic_customers_with_drug_consumption(
        "2024-01-01", drug_id=501, consumption_1=40.0, consumption_2=20.0
    )
    model = StubModel([0.2, 0.8])

    track1 = run_track1_inference(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )
    model = StubModel([0.2, 0.8])
    result = run_daily_forecast(
        raw_visits, as_of_date="2024-01-01", model=model, chronic_visit_prob_cutoff=0.3
    )

    pd.testing.assert_frame_equal(result.visit_list, track1.visit_list)
