import numpy as np
import pandas as pd
import pytest

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
    CalibratedTrack1Model,
    PlattCalibrator,
    load_track1_model,
    prepare_track1_features,
    save_track1_model,
    train_track1_model,
    tune_chronic_cutoff,
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


def test_returns_calibrated_model_metrics_and_cutoff_tuning():
    train, val = _train_val()

    result = train_track1_model(train, val)

    assert isinstance(result.model, CalibratedTrack1Model)
    assert set(result.metrics) == {"auc", "base_rate", "mean_p"}
    for value in result.metrics.values():
        assert 0.0 <= value <= 1.0
    assert result.cutoff_tuning.cutoff is not None
    assert 0.0 < result.cutoff_tuning.f1 <= 1.0


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
    classifier = result.model.classifier
    assert classifier.best_iteration_ is not None
    assert classifier.best_iteration_ > 0
    assert classifier.best_iteration_ < classifier.n_estimators


def _noisy_frame(n: int, seed: int, due_visit_rate: float) -> pd.DataFrame:
    """`_make_mart1_training_frame`, but only `due_visit_rate` of the due
    customers (25-35 days since last visit) actually visit -- so the true
    next-day probability for a due customer is `due_visit_rate`, not 1."""
    frame = _make_mart1_training_frame(n, seed)
    rng = np.random.default_rng(seed + 100)
    frame[NEXT_DAY_VISIT_COL] = frame[NEXT_DAY_VISIT_COL] & (rng.uniform(size=n) < due_visit_rate)
    return frame


def test_calibrated_validation_probabilities_match_the_validation_base_rate():
    # Train oversamples visits (every due customer visits); validation is the
    # real distribution, where only a fifth of them do. The raw scores carry
    # train's base rate; calibration must bring them down to validation's.
    train = _make_mart1_training_frame(600, seed=1)
    val = _noisy_frame(2000, seed=2, due_visit_rate=0.2)

    result = train_track1_model(train, val)

    X_val = prepare_track1_features(val)
    raw_mean = result.model.classifier.predict_proba(X_val)[:, 1].mean()
    calibrated_mean = result.model.predict_proba(X_val)[:, 1].mean()
    base_rate = val[NEXT_DAY_VISIT_COL].mean()
    assert raw_mean > 2 * base_rate
    assert calibrated_mean == pytest.approx(base_rate, abs=0.01)
    assert result.metrics["mean_p"] == pytest.approx(calibrated_mean)
    assert result.metrics["base_rate"] == pytest.approx(base_rate)


def test_platt_calibrator_recovers_a_known_sigmoid():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.01, 0.99, size=50_000)
    true_p = 1 / (1 + np.exp(-(0.5 * np.log(raw / (1 - raw)) - 2.0)))
    y = rng.uniform(size=raw.size) < true_p

    calibrator = PlattCalibrator.fit(raw, y)

    assert calibrator.slope == pytest.approx(0.5, abs=0.05)
    assert calibrator.intercept == pytest.approx(-2.0, abs=0.1)
    np.testing.assert_allclose(calibrator.transform(raw), true_p, atol=0.02)


def test_platt_calibrator_stays_finite_on_perfectly_separable_scores():
    raw = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([False, False, False, True, True, True])

    calibrator = PlattCalibrator.fit(raw, y)

    calibrated = calibrator.transform(raw)
    assert np.isfinite([calibrator.slope, calibrator.intercept]).all()
    assert (calibrated[:3] < 0.5).all() and (calibrated[3:] > 0.5).all()


class StubModel:
    """`predict_proba`-only stand-in returning given positive-class
    probabilities, so a test can check exactly what calibration does to them."""

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probabilities, self._probabilities])


def test_calibrated_model_applies_the_calibrator_to_raw_scores():
    raw = np.array([0.2, 0.5, 0.9])
    calibrator = PlattCalibrator(slope=2.0, intercept=-1.0)
    model = CalibratedTrack1Model(classifier=StubModel(raw), calibrator=calibrator)

    proba = model.predict_proba(pd.DataFrame(index=range(3)))

    expected = 1 / (1 + np.exp(-(2.0 * np.log(raw / (1 - raw)) - 1.0)))
    np.testing.assert_allclose(proba[:, 1], expected)
    np.testing.assert_allclose(proba[:, 0], 1 - expected)


# 10 Next-Day Visits among 20 rows: cutoff 0.9 admits 5 of them, 0.8
# admits 7, 0.6 the same 7 plus 3 non-visits, 0.4 all 10 plus 3 non-visits,
# 0.1 everyone.
_TUNING_PROBA = np.array([0.9] * 5 + [0.8] * 2 + [0.6] * 3 + [0.4] * 3 + [0.1] * 7)
_TUNING_Y = np.array([True] * 7 + [False] * 3 + [True] * 3 + [False] * 7)


def test_tune_cutoff_picks_the_f1_maximising_cutoff():
    # F1 per cutoff: 0.9 -> 0.667, 0.8 -> 0.824, 0.6 -> 0.7, 0.4 -> 0.870,
    # 0.1 -> 0.667.
    tuning = tune_chronic_cutoff(_TUNING_Y, _TUNING_PROBA)

    assert tuning.cutoff == pytest.approx(0.4)
    assert tuning.recall == pytest.approx(1.0)
    assert tuning.precision == pytest.approx(10 / 13)
    assert tuning.f1 == pytest.approx(2 * (10 / 13) / (1 + 10 / 13))


def test_tune_cutoff_breaks_f1_ties_towards_the_higher_cutoff():
    # Cutoff 0.9 (precision 1, recall 1/2) and 0.3 (precision 1/2, recall 1)
    # both reach F1 2/3; the higher one gives the shorter Visit List.
    y = np.array([True, False, False, True])
    proba = np.array([0.9, 0.6, 0.3, 0.3])

    tuning = tune_chronic_cutoff(y, proba)

    assert tuning.cutoff == pytest.approx(0.9)
    assert tuning.f1 == pytest.approx(2 / 3)


def test_tune_cutoff_reports_clearly_when_validation_has_no_visits():
    y = np.zeros(5, dtype=bool)
    proba = np.linspace(0.1, 0.5, 5)

    with pytest.warns(UserWarning, match="No Chronic cutoff"):
        tuning = tune_chronic_cutoff(y, proba)

    assert tuning.cutoff is None
    assert np.isnan(tuning.precision) and np.isnan(tuning.recall) and np.isnan(tuning.f1)


def test_training_reports_test_window_sum_of_probabilities_against_actual_visits():
    train = _make_mart1_training_frame(600, seed=1)
    val = _noisy_frame(2000, seed=2, due_visit_rate=0.2)
    test = _noisy_frame(2000, seed=3, due_visit_rate=0.2)

    result = train_track1_model(train, val, test)

    expected_sum_p = result.model.predict_proba(prepare_track1_features(test))[:, 1].sum()
    actual_visits = test[NEXT_DAY_VISIT_COL].sum()
    assert result.sum_p_check.sum_p == pytest.approx(expected_sum_p)
    assert result.sum_p_check.actual_visits == actual_visits
    assert result.sum_p_check.ratio == pytest.approx(expected_sum_p / actual_visits)
    # Same distribution as validation, so calibration should carry over.
    assert 0.5 < result.sum_p_check.ratio < 2.0


def test_training_without_a_test_window_reports_no_sum_p_check():
    train, val = _train_val()

    result = train_track1_model(train, val)

    assert result.sum_p_check is None


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

    # The raw classifiers, not the calibrated models: each one's calibrator
    # is fit separately, which this test isn't about.
    model_heavy_true = train_track1_model(heavy_true, val).model.classifier
    model_heavy_false = train_track1_model(heavy_false, val).model.classifier

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

    # The calibrator round-trips with the classifier: reloaded probabilities
    # are the calibrated ones, not the raw scores.
    assert reloaded.calibrator == result.model.calibrator
    X_val = prepare_track1_features(val)
    original_proba = result.model.predict_proba(X_val)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_val)[:, 1]
    np.testing.assert_allclose(original_proba, reloaded_proba)
