"""
Tests for churn feature generation and model training.

Uses synthetic data — no real dataset required.
Key validation: churn features use only historical data,
and labels are based on future behaviour.
"""
import sys
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import build_churn_features
from src.churn import train_churn_models, generate_churn_predictions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_churn_data(n_customers: int = 200, seed: int = 42):
    """
    Build synthetic orders and order_items for churn testing.
    Creates a 2-year history with a 90-day holdout.
    """
    rng = np.random.default_rng(seed)

    # Two-year date range
    all_dates = pd.date_range("2022-01-01", "2024-01-01", freq="D")

    # Cutoff: 90 days before the end
    cutoff = pd.Timestamp("2023-10-04", tz="UTC")

    users = pd.DataFrame({
        "id": range(1, n_customers + 1),
        "first_name": [f"User{i}" for i in range(n_customers)],
        "last_name": ["Test"] * n_customers,
        "email": [f"u{i}@test.com" for i in range(n_customers)],
        "age": rng.integers(18, 65, n_customers),
        "gender": rng.choice(["M", "F"], n_customers),
        "country": ["US"] * n_customers,
        "state": ["CA"] * n_customers,
        "city": ["LA"] * n_customers,
        "created_at": pd.date_range("2022-01-01", periods=n_customers, freq="h").astype(str),
    })

    orders_list = []
    items_list = []
    order_id = 1
    item_id = 1

    for uid in range(1, n_customers + 1):
        n_orders = rng.integers(1, 12)
        for _ in range(n_orders):
            date_idx = rng.integers(0, len(all_dates))
            date = all_dates[date_idx]
            orders_list.append({
                "order_id": order_id,
                "user_id": uid,
                "status": "complete",
                "created_at": pd.Timestamp(date, tz="UTC"),
                "num_of_item": 1,
            })
            items_list.append({
                "id": item_id,
                "order_id": order_id,
                "user_id": uid,
                "product_id": rng.integers(1, 20),
                "status": "complete",
                "created_at": pd.Timestamp(date, tz="UTC"),
                "sale_price": float(rng.uniform(20, 300)),
            })
            order_id += 1
            item_id += 1

    orders = pd.DataFrame(orders_list)
    order_items = pd.DataFrame(items_list)
    return users, orders, order_items


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildChurnFeatures:
    def setup_method(self):
        self.users, self.orders, self.items = make_synthetic_churn_data(150)

    def test_output_has_required_columns(self):
        feat = build_churn_features(self.orders, self.items, self.users)
        required = {"user_id", "frequency", "monetary", "recency", "churned"}
        assert required.issubset(set(feat.columns))

    def test_churn_label_binary(self):
        feat = build_churn_features(self.orders, self.items, self.users)
        assert set(feat["churned"].unique()).issubset({0, 1})

    def test_no_future_data_in_features(self):
        """Verify recency is calculated from cutoff, not from future."""
        feat = build_churn_features(self.orders, self.items, self.users)
        # Recency should be non-negative (all historical)
        assert (feat["recency"] >= 0).all()

    def test_frequency_positive(self):
        feat = build_churn_features(self.orders, self.items, self.users)
        # All customers had at least 1 order in history
        assert (feat["frequency"] >= 1).all()

    def test_monetary_non_negative(self):
        feat = build_churn_features(self.orders, self.items, self.users)
        assert (feat["monetary"] >= 0).all()

    def test_one_row_per_customer(self):
        feat = build_churn_features(self.orders, self.items, self.users)
        assert feat["user_id"].nunique() == len(feat)

    def test_both_churn_classes_present(self):
        """Must have at least some churned and some active."""
        feat = build_churn_features(self.orders, self.items, self.users)
        assert feat["churned"].sum() > 0
        assert (feat["churned"] == 0).sum() > 0


class TestTrainChurnModels:
    def setup_method(self):
        self.users, self.orders, self.items = make_synthetic_churn_data(300)
        self.feat = build_churn_features(self.orders, self.items, self.users)

    def test_model_trains_and_returns_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_churn_models(self.feat, models_dir=Path(tmpdir))
            assert "results" in results
            assert "best_model_name" in results
            assert results["best_model_name"] in ("Logistic Regression", "Random Forest")

    def test_roc_auc_in_valid_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_churn_models(self.feat, models_dir=Path(tmpdir))
            for model_name, metrics in results["results"].items():
                auc = metrics["roc_auc"]
                assert 0.0 <= auc <= 1.0, f"{model_name} AUC out of range: {auc}"

    def test_feature_importance_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_churn_models(self.feat, models_dir=Path(tmpdir))
            fi = results["feature_importance"]
            assert isinstance(fi, pd.DataFrame)
            assert "feature" in fi.columns
            assert "importance" in fi.columns
            assert (fi["importance"] >= 0).all()

    def test_predictions_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = train_churn_models(self.feat, models_dir=Path(tmpdir))
            preds = generate_churn_predictions(
                self.feat,
                results["best_model"],
                results["scaler"],
                results["feature_cols"],
            )
            assert "customer_id" in preds.columns
            assert "churn_probability" in preds.columns
            assert "risk_level" in preds.columns
            assert preds["churn_probability"].between(0, 1).all()
            assert set(preds["risk_level"].unique()).issubset(
                {"Low Risk", "Medium Risk", "High Risk"}
            )

    def test_insufficient_data_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            small_feat = self.feat.head(5)
            with pytest.raises(ValueError, match="Insufficient"):
                train_churn_models(small_feat, models_dir=Path(tmpdir))
