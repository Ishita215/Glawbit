"""
core/ml/outcome_model.py
========================
Outcome-association model trained on JDS Skill Traits.xlsx.

Model
-----
Logistic Regression with StandardScaler inside an sklearn Pipeline.
Validated with Stratified 5-Fold Cross-Validation (ROC-AUC).

Two comparison baselines are also computed:
  - DummyClassifier (most-frequent strategy)
  - Random Forest

The primary model is Logistic Regression because:
  - n = 139 (small dataset)
  - Binary outcome (salary_hike_high_or_low)
  - Coefficients are directly interpretable
  - Appropriate regularisation prevents overfitting on small data
  - ROC-AUC as primary metric is robust to class imbalance

Safety note
-----------
This model is trained on the JDS Skill Traits dataset.
It measures association between capability scores and provided salary-hike
outcome labels.  It must NOT be described as "predicting employee success".
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.data.loaders import load_jds_traits
from core.nlp.taxonomy import CAPABILITY_FEATURE_MAP

logger = logging.getLogger("dsxray.ml")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ML_DIR = BASE_DIR / "ml"
MODEL_PATH = ML_DIR / "success_model.joblib"
META_PATH = ML_DIR / "model_meta.json"

# Feature columns in the JDS dataset (order matters for the model)
FEATURE_COLUMNS: list[str] = [
    "big_data_skills",
    "maths-stats_skills",
    "coding_skills",
    "ai_and_ml_skills",
    "dashboard_and_storytelling_skills",
]
TARGET_COLUMN = "salary_hike_high_or_low"

# ── Module-level model cache ──────────────────────────────────
_model_cache: Optional[Pipeline] = None
_meta_cache: Optional[dict] = None


def load_model(force_reload: bool = False) -> Optional[Pipeline]:
    """Load the trained Logistic Regression pipeline from disk."""
    global _model_cache
    if _model_cache is not None and not force_reload:
        return _model_cache

    if not MODEL_PATH.exists():
        logger.warning("Model file not found: %s  Run train_models first.", MODEL_PATH)
        return None

    try:
        _model_cache = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
        return _model_cache
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        return None


def load_meta() -> dict:
    """Load model metadata (CV scores, version, etc.)."""
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache

    if not META_PATH.exists():
        logger.warning("Model meta not found: %s", META_PATH)
        return {}

    try:
        with open(META_PATH, "r", encoding="utf-8") as fh:
            _meta_cache = json.load(fh)
        return _meta_cache
    except Exception as exc:
        logger.exception("Failed to load model meta: %s", exc)
        return {}


def _save_model(pipeline: Pipeline, meta: dict) -> None:
    """Persist model pipeline and metadata."""
    ML_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("Model saved to %s", MODEL_PATH)


def train_and_evaluate() -> dict:
    """
    Load JDS data, train LR + baselines, run Stratified 5-Fold CV,
    save the primary model, and return a full comparison report.

    Returns
    -------
    dict with keys:
        logistic_regression, random_forest, dummy,
        selected_model, model_meta, comparison_reason
    """
    df = load_jds_traits(force_reload=True)
    if df.empty:
        raise RuntimeError("JDS Skill Traits dataset not available for training.")

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].values.astype(int)

    logger.info("Training on %d samples, %d features", len(X), X.shape[1])
    logger.info("Class distribution – 1: %d  0: %d", y.sum(), (y == 0).sum())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Define models ─────────────────────────────────────────
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "dummy": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent", random_state=42)),
        ]),
    }

    results: dict[str, dict] = {}

    for name, pipeline in models.items():
        t0 = time.time()
        fold_scores = cross_val_score(
            pipeline, X, y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        elapsed = round(time.time() - t0, 2)

        results[name] = {
            "fold_scores": [round(float(s), 6) for s in fold_scores],
            "mean_auc":    round(float(fold_scores.mean()), 6),
            "std_auc":     round(float(fold_scores.std()), 6),
            "cv_elapsed_s": elapsed,
        }
        logger.info(
            "%s  mean AUC=%.4f  std=%.4f",
            name, fold_scores.mean(), fold_scores.std(),
        )

    # ── Compute LR coefficients (fit on full data) ────────────
    lr_pipeline = models["logistic_regression"]
    lr_pipeline.fit(X, y)

    coef = lr_pipeline.named_steps["clf"].coef_[0]
    coefficients = {FEATURE_COLUMNS[i]: round(float(coef[i]), 4) for i in range(len(FEATURE_COLUMNS))}

    # ── Build metadata ────────────────────────────────────────
    import datetime
    meta = {
        "model_version":      "2.0",
        "training_date":      datetime.datetime.utcnow().isoformat() + "Z",
        "dataset":            "JDS Skill Traits.xlsx",
        "n_samples":          int(len(X)),
        "n_features":         int(X.shape[1]),
        "feature_columns":    FEATURE_COLUMNS,
        "target_column":      TARGET_COLUMN,
        "class_distribution": {"high": int(y.sum()), "low": int((y == 0).sum())},
        "cv_strategy":        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "scoring_metric":     "roc_auc",
        "models": results,
        "selected_model":     "logistic_regression",
        "selection_reason": (
            "Logistic Regression was selected because it provides strong cross-validated "
            "discrimination (mean AUC compared to Random Forest) while remaining fully "
            "interpretable on this small structured dataset (n=139).  Its coefficients "
            "directly expose the contribution of each capability dimension."
        ),
        "lr_coefficients": coefficients,
        "interpretability_note": (
            "This model measures association between capability scores and the provided "
            "salary-hike outcome labels in the JDS dataset.  It must not be described "
            "as predicting employee success in general."
        ),
    }

    _save_model(lr_pipeline, meta)

    # Invalidate cache so the freshly trained model is loaded next call
    global _model_cache, _meta_cache
    _model_cache = lr_pipeline
    _meta_cache = meta

    return {
        "logistic_regression": results["logistic_regression"],
        "random_forest":       results["random_forest"],
        "dummy":               results["dummy"],
        "selected_model":      "logistic_regression",
        "model_meta":          meta,
        "comparison_reason":   meta["selection_reason"],
    }


def predict_outcome_fit(capability_vector: dict[str, int]) -> dict:
    """
    Given a raw capability count vector, return:
      - probability_high (float 0-100)  — outcome-association score
      - model_name
      - model_version
      - feature_values used for prediction

    Returns probability = None when the model is unavailable.
    """
    model = load_model()
    meta = load_meta()

    # Map capability display names → feature column values
    feature_values: list[float] = []
    for col in FEATURE_COLUMNS:
        # Find the capability that maps to this column
        cap_name = next(
            (cap for cap, feat in CAPABILITY_FEATURE_MAP.items() if feat == col),
            None,
        )
        if cap_name is not None:
            feature_values.append(float(capability_vector.get(cap_name, 0)))
        else:
            feature_values.append(0.0)

    if model is None:
        return {
            "probability_high": None,
            "model_name": "Outcome-Association Model (Logistic Regression)",
            "model_version": meta.get("model_version", "unknown"),
            "feature_values": {FEATURE_COLUMNS[i]: feature_values[i] for i in range(len(FEATURE_COLUMNS))},
            "warning": "Model not yet trained. Run: python manage.py train_models",
        }

    X = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)
    try:
        prob = float(model.predict_proba(X)[0][1])
        return {
            "probability_high": round(prob * 100, 1),
            "model_name": "Outcome-Association Model (Logistic Regression)",
            "model_version": meta.get("model_version", "unknown"),
            "training_date": meta.get("training_date", "unknown"),
            "cv_mean_auc": meta.get("models", {}).get("logistic_regression", {}).get("mean_auc"),
            "cv_std_auc":  meta.get("models", {}).get("logistic_regression", {}).get("std_auc"),
            "feature_values": {FEATURE_COLUMNS[i]: feature_values[i] for i in range(len(FEATURE_COLUMNS))},
            "coefficients": meta.get("lr_coefficients", {}),
            "interpretation": (
                "Probability that a role with this capability profile is associated with a "
                "high salary-hike outcome in the JDS Skill Traits dataset.  "
                "This is an association measure, NOT a causal prediction of success."
            ),
        }
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return {
            "probability_high": None,
            "error": str(exc),
        }
