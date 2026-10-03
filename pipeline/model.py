"""Track 1 model training: turns Mart 1's train/validation split (see
`pipeline.marts.split_mart1_training_set`) into a persisted LightGBM
visit-probability classifier.

See CONTEXT.md and docs/adr/ for the design this encodes, and issue #19 for
this module's own scope -- Track 1's inference path (scoring, rare-drug
allocation, combining with Track 2) is built separately on top of the
`TrainedTrack1Model` this module produces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import precision_recall_curve, roc_auc_score

from .marts import (
    GENDER_COL,
    INSURANCE_TYPE_COL,
    MART1_COLUMNS,
    MART1_NON_FEATURE_COLS,
    NEXT_DAY_VISIT_COL,
    PRIMARY_INGREDIENT_COL,
    WEIGHT_COL,
)

# Track 1's Chronic visit-probability cutoff on *calibrated* probabilities
# (see CONTEXT.md "Visit-Probability Calibration" and "Decision Thresholds
# (provisional)"): gates the Visit List and the rare-drug allocation rule.
# `train_track1_model` re-tunes it on each validation window
# (`tune_chronic_cutoff`); this is the default wherever no tuned cutoff is
# passed. Still the untuned placeholder: on the v0.3 extract no cutoff meets
# both tuning targets below (issue #32).
CHRONIC_VISIT_PROB_CUTOFF = 0.3

# The provisional targets `tune_chronic_cutoff` tunes the cutoff against (see
# CONTEXT.md "Decision Thresholds (provisional)"): the cutoff must catch at
# least this share of next-day Chronic visits, at no worse than this
# precision.
CHRONIC_CUTOFF_MIN_RECALL = 0.70
CHRONIC_CUTOFF_MIN_PRECISION = 0.20

# Raw scores are clipped this far from 0 and 1 before taking their logit, so
# a saturated LightGBM score can't become an infinite Platt input.
_LOGIT_CLIP = 1e-12

# Mart 1 X-features LightGBM should treat as categorical (native pandas
# `category` dtype, see CONTEXT.md's Mart 1 entry and issue #19) rather than
# numeric, so LightGBM's built-in categorical split handling applies and no
# separate encoding/decoding layer needs to stay in sync between train and
# inference. Boolean flags (차상위대상자, 내일이_예약일) are deliberately left
# as plain booleans -- LightGBM already splits a two-valued column correctly
# without the categorical machinery.
CATEGORICAL_FEATURE_COLS = [GENDER_COL, PRIMARY_INGREDIENT_COL, INSURANCE_TYPE_COL]

# MART1_COLUMNS entries that are model X-features -- everything except
# MART1_NON_FEATURE_COLS' identifier/label/weight/Chronic-only-population
# constant (만성질환여부 is always True across Track 1's training set and
# single-snapshot mart1 alike -- see `build_mart1_training_set` -- so it
# carries no signal).
FEATURE_COLS = [col for col in MART1_COLUMNS if col not in MART1_NON_FEATURE_COLS]

DEFAULT_TRACK1_MODEL_PATH = Path("pipeline/artifacts/track1_lgbm.pkl")

# Generous upper bound on boosting rounds -- early stopping (below) is what
# actually decides how many get used; this only bounds the worst case.
LIGHTGBM_MAX_BOOST_ROUNDS = 500
LIGHTGBM_EARLY_STOPPING_ROUNDS = 20


@dataclass(frozen=True)
class PlattCalibrator:
    """A Platt (sigmoid) map from a raw visit score to a calibrated
    probability: `sigmoid(slope * logit(raw) + intercept)` (see CONTEXT.md
    "Visit-Probability Calibration")."""

    slope: float
    intercept: float

    @classmethod
    def fit(cls, raw_proba, y) -> PlattCalibrator:
        """Fits by maximum likelihood on `raw_proba` (positive-class scores)
        against the 0/1 labels `y`, with Platt's smoothed targets
        (`(N+ + 1) / (N+ + 2)` and `1 / (N- + 2)`) so perfectly separable
        scores still give a finite fit."""
        scores = _logit(raw_proba)
        y = np.asarray(y, dtype=bool)
        n_pos = y.sum()
        n_neg = y.size - n_pos
        targets = np.where(y, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))

        def loss_and_grad(params):
            z = params[0] * scores + params[1]
            loss = np.sum(np.logaddexp(0.0, z) - targets * z)
            residual = expit(z) - targets
            return loss, np.array([residual @ scores, residual.sum()])

        fitted = minimize(loss_and_grad, x0=[1.0, 0.0], jac=True, method="L-BFGS-B")
        return cls(slope=float(fitted.x[0]), intercept=float(fitted.x[1]))

    def transform(self, raw_proba) -> np.ndarray:
        return expit(self.slope * _logit(raw_proba) + self.intercept)


@dataclass(frozen=True)
class CalibratedTrack1Model:
    """Track 1's classifier with its Platt calibrator: what training returns,
    what gets saved and loaded, and what inference and the backtest score
    with. `predict_proba` gives calibrated probabilities, so every consumer
    of a visit probability sees them without knowing calibration exists.

    `classifier` is anything with a scikit-learn-shaped `predict_proba` --
    the trained `LGBMClassifier` in production, a stub in tests."""

    classifier: object
    calibrator: PlattCalibrator

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        calibrated = self.calibrator.transform(self.classifier.predict_proba(X)[:, 1])
        return np.column_stack([1 - calibrated, calibrated])


class CutoffTuning(NamedTuple):
    """`tune_chronic_cutoff`'s result: the chosen cutoff and the validation
    precision/recall at it -- all None/NaN when no cutoff met both targets --
    plus the targets it was tuned against."""

    cutoff: float | None
    precision: float
    recall: float
    min_recall: float
    min_precision: float


class SumPCheck(NamedTuple):
    """Σp of calibrated probabilities over the whole test window against the
    actual number of next-day Chronic visits in it (see CONTEXT.md
    "Visit-Probability Calibration"). `ratio` is `sum_p / actual_visits`."""

    sum_p: float
    actual_visits: int
    ratio: float


class TrainedTrack1Model(NamedTuple):
    model: CalibratedTrack1Model
    metrics: dict[str, float]
    cutoff_tuning: CutoffTuning
    sum_p_check: SumPCheck | None


class _Mart1Features(NamedTuple):
    features: pd.DataFrame
    target: pd.Series
    weight: pd.Series


def train_track1_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame | None = None,
    min_recall: float = CHRONIC_CUTOFF_MIN_RECALL,
    min_precision: float = CHRONIC_CUTOFF_MIN_PRECISION,
) -> TrainedTrack1Model:
    """Trains Track 1's LightGBM visit-probability classifier on `train`,
    early-stopped against `val`'s logloss (never `train`'s -- `val` is the
    only eval set ever passed to LightGBM, so there is nothing else for early
    stopping to monitor), then Platt-calibrates it on `val` and tunes the
    Chronic cutoff on the calibrated `val` probabilities.

    `train`/`val`/`test` are Mart 1 rows (matching
    `MART1_TRAINING_COLUMNS`/`MART1_COLUMNS` -- see
    `pipeline.marts.split_mart1_training_set`): X-features (`FEATURE_COLS`)
    plus `내일_방문` (Y) and `학습_가중치` (sample weight). Any extra columns
    (e.g. `기준일자`) are ignored. Categorical X-features (`CATEGORICAL_FEATURE_COLS`)
    are passed to LightGBM as native pandas `category` dtype. `val` should be
    the validation window's full daily snapshots: calibration maps the raw
    scores, whose base rate is an artifact of Negative Sampling, onto `val`'s
    real visit rate. The calibrator is fit unweighted for the same reason --
    it targets the real daily rate, not the tier-weighted one.

    Returns a `TrainedTrack1Model`:

    - `model`: the `CalibratedTrack1Model` (classifier + calibrator).
    - `metrics` on `val`: `auc`, `base_rate` (share of rows labelled
      내일_방문) and `mean_p` (mean calibrated probability, which should
      match `base_rate`).
    - `cutoff_tuning`: `tune_chronic_cutoff` on the calibrated `val`
      probabilities, against `min_recall`/`min_precision`.
    - `sum_p_check`: `test`'s Σp against its actual next-day visits, or None
      when no `test` is given. `test` is never used for fitting.
    """
    X_train, y_train, weight_train = _features_target_weight(train)
    X_val, y_val, _ = _features_target_weight(val)

    classifier = lgb.LGBMClassifier(n_estimators=LIGHTGBM_MAX_BOOST_ROUNDS, random_state=0)
    classifier.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    raw_val_proba = classifier.predict_proba(X_val)[:, 1]
    model = CalibratedTrack1Model(
        classifier=classifier, calibrator=PlattCalibrator.fit(raw_val_proba, y_val)
    )
    val_proba = model.calibrator.transform(raw_val_proba)
    metrics = {
        "auc": roc_auc_score(y_val, val_proba),
        "base_rate": y_val.mean(),
        "mean_p": val_proba.mean(),
    }
    return TrainedTrack1Model(
        model=model,
        metrics=metrics,
        cutoff_tuning=tune_chronic_cutoff(y_val, val_proba, min_recall, min_precision),
        sum_p_check=None if test is None else _sum_p_check(model, test),
    )


def tune_chronic_cutoff(
    y,
    proba,
    min_recall: float = CHRONIC_CUTOFF_MIN_RECALL,
    min_precision: float = CHRONIC_CUTOFF_MIN_PRECISION,
) -> CutoffTuning:
    """The highest cutoff (a customer is predicted to visit when `proba >=
    cutoff`, as the Visit List does) that still catches `min_recall` of the
    actual Next-Day Visits in `y`, provided precision there is at least
    `min_precision` -- the smallest Visit List that meets both targets.

    When no cutoff meets both, warns and returns `cutoff=None` (precision and
    recall NaN) rather than silently choosing one.
    """
    y = np.asarray(y, dtype=bool)
    if y.any():
        precision, recall, thresholds = precision_recall_curve(y, proba)
        # The curve's last point (precision 1, recall 0) has no threshold.
        meets_targets = (recall[:-1] >= min_recall) & (precision[:-1] >= min_precision)
    else:
        meets_targets = np.zeros(0, dtype=bool)

    if not meets_targets.any():
        warnings.warn(
            f"No Chronic cutoff reaches {min_recall:.0%} recall with at least "
            f"{min_precision:.0%} precision on the validation probabilities.",
            stacklevel=2,
        )
        return CutoffTuning(
            cutoff=None,
            precision=float("nan"),
            recall=float("nan"),
            min_recall=min_recall,
            min_precision=min_precision,
        )

    # Thresholds ascend, so the last feasible one is the highest.
    best = np.flatnonzero(meets_targets)[-1]
    return CutoffTuning(
        cutoff=float(thresholds[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        min_recall=min_recall,
        min_precision=min_precision,
    )


def _sum_p_check(model: CalibratedTrack1Model, test: pd.DataFrame) -> SumPCheck:
    X_test, y_test, _ = _features_target_weight(test)
    sum_p = float(model.predict_proba(X_test)[:, 1].sum())
    actual_visits = int(y_test.sum())
    ratio = sum_p / actual_visits if actual_visits else float("nan")
    return SumPCheck(sum_p=sum_p, actual_visits=actual_visits, ratio=ratio)


def _logit(proba) -> np.ndarray:
    return logit(np.clip(np.asarray(proba, dtype=float), _LOGIT_CLIP, 1 - _LOGIT_CLIP))


def prepare_track1_features(mart1_rows: pd.DataFrame) -> pd.DataFrame:
    """LightGBM-ready X-features (`FEATURE_COLS`, with `CATEGORICAL_FEATURE_COLS`
    as native pandas `category` dtype) from any `MART1_COLUMNS`-shaped frame.

    Shared by `_features_target_weight` (training) and Track 1 inference
    (`pipeline.inference`), so a Mart 1 row becomes a model input the same way
    in both places and the two can never drift out of sync.
    """
    features = mart1_rows[FEATURE_COLS].copy()
    for col in CATEGORICAL_FEATURE_COLS:
        features[col] = features[col].astype("category")
    return features


def _features_target_weight(mart1_rows: pd.DataFrame) -> _Mart1Features:
    """`mart1_rows` (Mart 1 training-set/`MART1_COLUMNS`-shaped rows) split
    into LightGBM-ready X-features (categorical columns as native `category`
    dtype), the `내일_방문` target as 0/1, and the `학습_가중치` sample
    weight."""
    features = prepare_track1_features(mart1_rows)
    target = mart1_rows[NEXT_DAY_VISIT_COL].astype(int)
    weight = mart1_rows[WEIGHT_COL]
    return _Mart1Features(features=features, target=target, weight=weight)


def save_track1_model(
    model: CalibratedTrack1Model, path: Path | str = DEFAULT_TRACK1_MODEL_PATH
) -> None:
    """Persists a trained Track 1 model (`TrainedTrack1Model.model`: the
    classifier together with its calibrator) to `path` via `joblib`, creating
    parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_track1_model(path: Path | str = DEFAULT_TRACK1_MODEL_PATH) -> CalibratedTrack1Model:
    """Reloads a calibrated Track 1 model previously persisted by
    `save_track1_model`."""
    return joblib.load(Path(path))
