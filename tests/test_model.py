import numpy as np
import pandas as pd

from pipeline.marts import (
    AGE_COL,
    CHRONIC_COL,
    CUSTOMER_ID_COL,
    DAYS_SINCE_LAST_VISIT_COL,
    FAMILY_VISIT_COUNT_COL,
    GENDER_COL,
    INSURANCE_TYPE_COL,
    LONG_TERM_MED_DAYS_COL,
    MART1_TRAINING_COLUMNS,
    MPR_COL,
    NEAR_POVERTY_COL,
    NEXT_DAY_VISIT_COL,
    NO_SHOW_RATE_COL,
    PRIMARY_INGREDIENT_COL,
    REMAINING_MED_DAYS_COL,
    SNAPSHOT_DATE_COL,
    TOMORROW_IS_EXPECTED_VISIT_COL,
    WEIGHT_COL,
)
from pipeline.model import (
    load_track1_model,
    prepare_track1_features,
    save_track1_model,
    train_track1_model,
)


def _make_mart1_training_frame(n: int, seed: int) -> pd.DataFrame:
    """A synthetic Mart 1 training-set-shaped frame (matching
    MART1_TRAINING_COLUMNS) with a deterministic, learnable relationship
    between 마지막방문_경과일 and 내일_방문: a customer at/near their
    ~30-day prescription cycle (25-35 days since last visit) is due, and
    actually visits; everyone else doesn't. Genuinely learnable by a
    tree-based classifier without being a trivial single-feature passthrough,
    since the other X-features are populated with realistic noise around it.
    """
    rng = np.random.default_rng(seed)
    days_since_last_visit = rng.uniform(0, 60, size=n)
    next_day_visit = (days_since_last_visit >= 25) & (days_since_last_visit <= 35)

    return pd.DataFrame(
        {
            SNAPSHOT_DATE_COL: pd.date_range("2024-01-01", periods=n, freq="D"),
            CUSTOMER_ID_COL: np.arange(n),
            NEXT_DAY_VISIT_COL: next_day_visit,
            CHRONIC_COL: True,
            AGE_COL: rng.integers(20, 80, size=n),
            GENDER_COL: rng.choice(["남", "여"], size=n),
            FAMILY_VISIT_COUNT_COL: rng.integers(0, 10, size=n),
            DAYS_SINCE_LAST_VISIT_COL: days_since_last_visit,
            REMAINING_MED_DAYS_COL: rng.uniform(-10, 30, size=n),
            TOMORROW_IS_EXPECTED_VISIT_COL: rng.choice([True, False], size=n),
            LONG_TERM_MED_DAYS_COL: rng.uniform(1, 90, size=n),
            PRIMARY_INGREDIENT_COL: rng.choice(["아스피린", "메트포르민", "암로디핀"], size=n),
            INSURANCE_TYPE_COL: rng.choice(["건강보험", "의료급여"], size=n),
            NEAR_POVERTY_COL: rng.choice([True, False], size=n),
            MPR_COL: rng.uniform(0, 100, size=n),
            NO_SHOW_RATE_COL: rng.uniform(0, 1, size=n),
            WEIGHT_COL: 1.0,
        }
    )


def _train_val():
    train = _make_mart1_training_frame(300, seed=1)
    val = _make_mart1_training_frame(100, seed=2)
    return train, val


def test_returns_model_and_metrics_with_expected_keys_in_range():
    train, val = _train_val()

    result = train_track1_model(train, val)

    assert result.model is not None
    assert set(result.metrics) == {"auc", "base_rate", "precision", "recall"}
    for value in result.metrics.values():
        assert 0.0 <= value <= 1.0


def test_learns_the_synthetic_relationship_reasonably_well():
    # A strong sanity check that training actually happened against real
    # signal (categorical dtype conversion succeeded, target wiring is
    # correct) -- not just that some number in [0, 1] came back.
    train, val = _train_val()

    result = train_track1_model(train, val)

    assert result.metrics["auc"] > 0.8


def test_early_stopping_applied_against_validation_set():
    train, val = _train_val()

    result = train_track1_model(train, val)

    # LightGBM only ever receives `val` as its eval set (see
    # train_track1_model), so a best_iteration_ below the generous max round
    # count demonstrates early stopping triggered from monitoring it.
    assert result.model.best_iteration_ is not None
    assert result.model.best_iteration_ > 0
    assert result.model.best_iteration_ < result.model.n_estimators


def test_cutoff_argument_changes_precision_and_recall():
    train, val = _train_val()

    lenient = train_track1_model(train, val, cutoff=0.01)
    strict = train_track1_model(train, val, cutoff=0.99)

    # A near-zero cutoff calls almost everything positive (near-perfect
    # recall, weaker precision); a near-one cutoff does the reverse.
    assert lenient.metrics["recall"] >= strict.metrics["recall"]
    assert lenient.metrics["precision"] <= strict.metrics["precision"]


def _fixed_feature_vector() -> dict:
    """One X-feature vector at 마지막방문_경과일=90, outside
    `_make_mart1_training_frame`'s [0, 60) draw range -- so `_conflict_rows`
    below are the only evidence any trained model ever sees for it."""
    return {
        AGE_COL: 50,
        GENDER_COL: "여",
        FAMILY_VISIT_COUNT_COL: 2,
        DAYS_SINCE_LAST_VISIT_COL: 90.0,
        REMAINING_MED_DAYS_COL: 10.0,
        TOMORROW_IS_EXPECTED_VISIT_COL: False,
        LONG_TERM_MED_DAYS_COL: 30.0,
        PRIMARY_INGREDIENT_COL: "아스피린",
        INSURANCE_TYPE_COL: "건강보험",
        NEAR_POVERTY_COL: False,
        MPR_COL: 80.0,
        NO_SHOW_RATE_COL: 0.1,
    }


def _conflict_rows(heavy_label: bool, light_label: bool) -> pd.DataFrame:
    """10 rows sharing `_fixed_feature_vector`'s exact X-features: 5 labeled
    `heavy_label` at 학습_가중치=100, 5 labeled `light_label` at weight=1 --
    directly conflicting evidence for the same feature vector, distinguished
    only by weight. If `train_track1_model` actually threads 학습_가중치 into
    LightGBM's sample_weight, the model should follow the heavily-weighted
    label for this vector; if the weight were dropped or ignored, the two
    labels would roughly cancel out (~50/50) instead."""
    vector = _fixed_feature_vector()
    rows = [
        {
            **vector,
            SNAPSHOT_DATE_COL: pd.Timestamp("2030-01-01") + pd.Timedelta(days=i),
            CUSTOMER_ID_COL: 900000 + i,
            CHRONIC_COL: True,
            NEXT_DAY_VISIT_COL: heavy_label if i < 5 else light_label,
            WEIGHT_COL: 100.0 if i < 5 else 1.0,
        }
        for i in range(10)
    ]
    return pd.DataFrame(rows, columns=MART1_TRAINING_COLUMNS)


def test_sample_weight_is_threaded_into_training():
    train, val = _train_val()
    heavy_true = pd.concat(
        [train, _conflict_rows(heavy_label=True, light_label=False)], ignore_index=True
    )
    heavy_false = pd.concat(
        [train, _conflict_rows(heavy_label=False, light_label=True)], ignore_index=True
    )

    model_heavy_true = train_track1_model(heavy_true, val).model
    model_heavy_false = train_track1_model(heavy_false, val).model

    vector_row = prepare_track1_features(pd.DataFrame([_fixed_feature_vector()]))
    proba_heavy_true = model_heavy_true.predict_proba(vector_row)[0, 1]
    proba_heavy_false = model_heavy_false.predict_proba(vector_row)[0, 1]

    assert proba_heavy_true > proba_heavy_false


def test_model_persists_and_reloads_to_an_equivalent_usable_model(tmp_path):
    train, val = _train_val()
    result = train_track1_model(train, val)
    model_path = tmp_path / "track1_lgbm.pkl"

    save_track1_model(result.model, model_path)
    reloaded = load_track1_model(model_path)

    X_val = prepare_track1_features(val)
    original_proba = result.model.predict_proba(X_val)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_val)[:, 1]
    np.testing.assert_allclose(original_proba, reloaded_proba)
