"""
CustomerIQ — Churn Prediction.

Trains and evaluates churn prediction models:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest (stronger, with feature importance)

Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
Models saved to models/ directory using joblib.

Churn label definition:
  A customer is labelled "churned" (1) if they made no purchases during the
  CHURN_FUTURE_DAYS-day window after the feature cutoff date.
  See feature_engineering.py for the full methodology.

Risk thresholds:
  Low Risk   : churn_probability < RISK_LOW_THRESHOLD  (default 30%)
  Medium Risk: RISK_LOW_THRESHOLD ≤ prob < RISK_HIGH_THRESHOLD
  High Risk  : churn_probability ≥ RISK_HIGH_THRESHOLD (default 60%)
"""
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    RISK_LOW_THRESHOLD,
    RISK_HIGH_THRESHOLD,
    MODELS_DIR,
    MODEL_FILES,
)

logger = logging.getLogger("customeriq.churn")


# ---------------------------------------------------------------------------
# Feature columns used for modelling
# ---------------------------------------------------------------------------
CHURN_FEATURE_COLS: list[str] = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "n_unique_products",
    "tenure_days",
    "is_male",
    "age",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_churn_models(
    churn_features: pd.DataFrame,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """
    Train Logistic Regression and Random Forest churn models.

    Parameters
    ----------
    churn_features : pd.DataFrame
        Output of feature_engineering.build_churn_features().
        Must have columns listed in CHURN_FEATURE_COLS plus 'churned'.
    models_dir : Path
        Directory to save trained models.

    Returns
    -------
    dict
        {
          "best_model_name": str,
          "results": {model_name: metrics_dict},
          "feature_importance": pd.DataFrame,
          "best_model": fitted_estimator,
          "scaler": fitted_scaler,
          "feature_cols": list[str],
        }
    """
    # Check sufficient data
    n_samples = len(churn_features)
    if n_samples < 100:
        raise ValueError(
            f"Insufficient data for churn modelling: {n_samples} customers. "
            "Need at least 100 customers with a defined churn label."
        )

    n_churned = churn_features["churned"].sum()
    n_active = n_samples - n_churned
    logger.info(
        "Churn modelling: %d customers | %d churned (%.1f%%) | %d active",
        n_samples, n_churned, n_churned / n_samples * 100, n_active,
    )

    if n_churned < 10 or n_active < 10:
        raise ValueError(
            "Insufficient class representation for churn modelling. "
            f"Churned: {n_churned}, Active: {n_active}. "
            "Need at least 10 samples per class."
        )

    # Prepare features
    available_cols = [c for c in CHURN_FEATURE_COLS if c in churn_features.columns]
    X = churn_features[available_cols].fillna(0).values
    y = churn_features["churned"].values

    # Time-based split: use chronological order if recency is a proxy for time
    # Here we use a random split with fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ML_TEST_SIZE,
        random_state=ML_RANDOM_STATE,
        stratify=y,
    )
    logger.info(
        "Train: %d samples | Test: %d samples", len(X_train), len(X_test)
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results: dict[str, dict] = {}

    # --- Logistic Regression (baseline) ---
    lr = LogisticRegression(
        max_iter=1000,
        random_state=ML_RANDOM_STATE,
        class_weight="balanced",
    )
    lr.fit(X_train_scaled, y_train)
    lr_metrics = _evaluate_model(lr, X_test_scaled, y_test, "Logistic Regression")
    results["Logistic Regression"] = lr_metrics

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=ML_RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)
    rf_metrics = _evaluate_model(rf, X_test_scaled, y_test, "Random Forest")
    results["Random Forest"] = rf_metrics

    # Determine best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = rf if best_name == "Random Forest" else lr
    logger.info("Best model: %s (ROC-AUC: %.4f)", best_name, results[best_name]["roc_auc"])

    # Feature importance (from Random Forest)
    fi_df = pd.DataFrame({
        "feature": available_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save model artifacts
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / MODEL_FILES["churn_best"])
    joblib.dump(scaler, models_dir / MODEL_FILES["churn_scaler"])
    joblib.dump(available_cols, models_dir / MODEL_FILES["churn_features"])
    logger.info("Model artifacts saved to %s", models_dir)

    return {
        "best_model_name": best_name,
        "results": results,
        "feature_importance": fi_df,
        "best_model": best_model,
        "scaler": scaler,
        "feature_cols": available_cols,
        "all_models": {"Logistic Regression": lr, "Random Forest": rf},
    }


def generate_churn_predictions(
    churn_features: pd.DataFrame,
    model: Any,
    scaler: Any,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Generate churn predictions for all customers.

    Parameters
    ----------
    churn_features : pd.DataFrame
        Feature table with customer_id column.
    model : fitted sklearn estimator
        Trained churn model.
    scaler : fitted sklearn scaler
        Fitted StandardScaler.
    feature_cols : list[str]
        Feature columns used during training.

    Returns
    -------
    pd.DataFrame
        customer_id, churn_probability, churn_prediction, risk_level
    """
    available = [c for c in feature_cols if c in churn_features.columns]
    X = churn_features[available].fillna(0).values
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[:, 1]
    pred = model.predict(X_scaled)

    def assign_risk(p: float) -> str:
        if p < RISK_LOW_THRESHOLD:
            return "Low Risk"
        elif p < RISK_HIGH_THRESHOLD:
            return "Medium Risk"
        else:
            return "High Risk"

    predictions = pd.DataFrame({
        "customer_id": churn_features["user_id"].values,
        "churn_probability": proba.round(4),
        "churn_prediction": pred,
        "risk_level": [assign_risk(p) for p in proba],
    })

    risk_counts = predictions["risk_level"].value_counts()
    logger.info("Risk distribution:\n%s", risk_counts.to_string())
    return predictions


def load_churn_model(models_dir: Path = MODELS_DIR) -> tuple[Any, Any, list[str]]:
    """
    Load the saved churn model, scaler, and feature list.

    Returns
    -------
    (model, scaler, feature_cols)
    """
    model = joblib.load(models_dir / MODEL_FILES["churn_best"])
    scaler = joblib.load(models_dir / MODEL_FILES["churn_scaler"])
    feature_cols = joblib.load(models_dir / MODEL_FILES["churn_features"])
    return model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Internal evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a trained model and return a metrics dictionary."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    logger.info(
        "%s | Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        model_name, acc, prec, rec, f1, auc,
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }
