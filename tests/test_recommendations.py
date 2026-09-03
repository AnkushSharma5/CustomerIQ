"""
Tests for recommendation engine.

Uses small synthetic data.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommendation import build_recommendations, get_customer_recommendations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_products(n: int = 20) -> pd.DataFrame:
    categories = ["Clothing", "Electronics", "Books", "Sports", "Home"]
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"Product {i}" for i in range(1, n + 1)],
        "category": [categories[i % len(categories)] for i in range(n)],
        "brand": [f"Brand{i % 5}" for i in range(n)],
        "cost": [20.0] * n,
        "retail_price": [50.0] * n,
    })


def make_order_items(n_customers: int = 20, n_products: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows = []
    item_id = 1
    for uid in range(1, n_customers + 1):
        n_purchases = rng.integers(2, 8)
        product_ids = rng.choice(range(1, n_products + 1), n_purchases, replace=False)
        for pid in product_ids:
            rows.append({
                "id": item_id,
                "order_id": item_id,
                "user_id": uid,
                "product_id": int(pid),
                "status": "complete",
                "created_at": "2023-01-01",
                "sale_price": 50.0,
            })
            item_id += 1
    return pd.DataFrame(rows)


def make_segments(n_customers: int = 20) -> pd.DataFrame:
    segments = ["Champions", "Loyal Customers", "At Risk", "New Customers"]
    return pd.DataFrame({
        "customer_id": range(1, n_customers + 1),
        "segment": [segments[i % len(segments)] for i in range(n_customers)],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildRecommendations:
    def setup_method(self):
        self.products = make_products(20)
        self.order_items = make_order_items(20, 20)
        self.segments = make_segments(20)

    def test_returns_dataframe(self):
        recs = build_recommendations(self.order_items, self.products, self.segments)
        assert isinstance(recs, pd.DataFrame)

    def test_required_columns_present(self):
        recs = build_recommendations(self.order_items, self.products, self.segments)
        required = {"customer_id", "product_id", "affinity_score", "explanation", "rank"}
        assert required.issubset(set(recs.columns))

    def test_no_already_purchased_products_recommended(self):
        """Recommended products must not be in the customer's purchase history."""
        recs = build_recommendations(self.order_items, self.products, self.segments)
        for customer_id in recs["customer_id"].unique():
            purchased = set(
                self.order_items[self.order_items["user_id"] == customer_id]["product_id"]
            )
            recommended = set(
                recs[recs["customer_id"] == customer_id]["product_id"]
            )
            overlap = purchased & recommended
            assert len(overlap) == 0, (
                f"Customer {customer_id} was recommended already-purchased products: {overlap}"
            )

    def test_affinity_score_in_0_1_range(self):
        recs = build_recommendations(self.order_items, self.products, self.segments)
        assert recs["affinity_score"].between(0, 1).all()

    def test_explanation_is_non_empty_string(self):
        recs = build_recommendations(self.order_items, self.products, self.segments)
        assert (recs["explanation"].str.len() > 0).all()

    def test_max_recommendations_per_customer(self):
        from src.config import MAX_RECOMMENDATIONS
        recs = build_recommendations(self.order_items, self.products, self.segments)
        counts = recs.groupby("customer_id").size()
        assert (counts <= MAX_RECOMMENDATIONS).all()

    def test_get_customer_recommendations(self):
        recs = build_recommendations(self.order_items, self.products, self.segments)
        if len(recs) > 0:
            customer_id = recs["customer_id"].iloc[0]
            cust_recs = get_customer_recommendations(customer_id, recs, n=3)
            assert len(cust_recs) <= 3
            assert (cust_recs["customer_id"] == customer_id).all()
