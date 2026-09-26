"""Track 1 model training: turns Mart 1's train/validation split (see
`pipeline.marts.split_mart1_training_set`) into a persisted LightGBM
visit-probability classifier.

See CONTEXT.md and docs/adr/ for the design this encodes, and issue #19 for
this module's own scope -- Track 1's inference path (scoring, rare-drug
allocation, combining with Track 2) is built separately on top of the
`TrainedTrack1Model` this module produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from .marts import (
    GENDER_COL,
    INSURANCE_TYPE_COL,
    MART1_COLUMNS,
    MART1_NON_FEATURE_COLS,
    NEXT_DAY_VISIT_COL,
    PRIMARY_INGREDIENT_COL,
    WEIGHT_COL,
)

# Track 1's visit-probability decision threshold (see CONTEXT.md "Decision
# Thresholds (provisional)"): a placeholder pending empirical tuning, used
# here only to compute validation-set precision/recall at a fixed operating
# point so the model's sanity can be sanity-checked before it drives real
# order quantities (see issue #17/#19).
CHRONIC_VISIT_PROB_CUTOFF = 0.3

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


class TrainedTrack1Model(NamedTuple):
    model: lgb.LGBMClassifier
    metrics: dict[str, float]


class _Mart1Features(NamedTuple):
    features: pd.DataFrame
    target: pd.Series
    weight: pd.Series


def train_track1_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cutoff: float = CHRONIC_VISIT_PROB_CUTOFF,
) -> TrainedTrack1Model:
    """Trains Track 1's LightGBM visit-probability classifier on `train`,
    early-stopped against `val`'s logloss (never `train`'s -- `val` is the
    only eval set ever passed to LightGBM, so there is nothing else for early
    stopping to monitor).

    `train`/`val` are Mart 1 training-set rows (matching
    `MART1_TRAINING_COLUMNS`/`MART1_COLUMNS` -- see
    `pipeline.marts.split_mart1_training_set`): X-features (`FEATURE_COLS`)
    plus `내일_방문` (Y) and `학습_가중치` (sample weight). Any extra columns
    (e.g. `기준일자`) are ignored. Categorical X-features (`CATEGORICAL_FEATURE_COLS`)
    are passed to LightGBM as native pandas `category` dtype.

    Returns a `TrainedTrack1Model` with the fitted classifier and a metrics
    dict (`auc`, `precision`, `recall`) computed on `val` -- precision/recall
    at `cutoff` (default `CHRONIC_VISIT_PROB_CUTOFF`).
    """
    X_train, y_train, weight_train = _features_target_weight(train)
    X_val, y_val, _ = _features_target_weight(val)

    model = lgb.LGBMClassifier(n_estimators=LIGHTGBM_MAX_BOOST_ROUNDS, random_state=0)
    model.fit(
        X_train,
        y_train,
        sample_weight=weight_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="logloss",
        callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = val_proba >= cutoff
    metrics = {
        "auc": roc_auc_score(y_val, val_proba),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall": recall_score(y_val, val_pred, zero_division=0),
    }
    return TrainedTrack1Model(model=model, metrics=metrics)


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
    model: lgb.LGBMClassifier, path: Path | str = DEFAULT_TRACK1_MODEL_PATH
) -> None:
    """Persists a trained Track 1 classifier (`TrainedTrack1Model.model`) to
    `path` via `joblib`, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_track1_model(path: Path | str = DEFAULT_TRACK1_MODEL_PATH) -> lgb.LGBMClassifier:
    """Reloads a Track 1 classifier previously persisted by `save_track1_model`."""
    return joblib.load(Path(path))
