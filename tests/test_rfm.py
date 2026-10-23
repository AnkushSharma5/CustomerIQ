"""
Tests for RFM calculation.

Uses small synthetic DataFrames.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_engineering import build_rfm_features
from src.segmentation import compute_rfm_scores


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_orders_for_rfm() -> pd.DataFrame:
    """Orders for 5 customers with varying behaviour."""
    data = [
        # (order_id, user_id, status, created_at)
        (1, 1, "complete", "2023-11-01"),  # customer 1: recent
        (2, 1, "complete", "2023-11-15"),
        (3, 2, "complete", "2023-01-01"),  # customer 2: old, infrequent
        (4, 3, "complete", "2023-09-01"),  # customer 3: mid
        (5, 3, "complete", "2023-10-01"),
        (6, 3, "complete", "2023-11-01"),
        (7, 4, "complete", "2023-06-01"),  # customer 4: moderate
        (8, 5, "complete", "2023-11-20"),  # customer 5: very recent
    ]
    return pd.DataFrame(data, columns=["order_id", "user_id", "status", "created_at"])


def _make_items_for_rfm(orders: pd.DataFrame) -> pd.DataFrame:
    """Order items with sale prices."""
    items = []
    prices = {1: [120.0, 80.0], 2: [200.0], 3: [50.0, 75.0, 100.0], 4: [60.0], 5: [300.0]}
    item_id = 1
    for _, row in orders.iterrows():
        price_list = prices.get(int(row["order_id"]), [50.0])
        for price in price_list:
            items.append({
                "id": item_id,
                "order_id": row["order_id"],
                "user_id": row["user_id"],
                "product_id": item_id % 5 + 1,
                "status": "complete",
                "created_at": row["created_at"],
                "sale_price": price,
            })
            item_id += 1
    return pd.DataFrame(items)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildRfmFeatures:
    def setup_method(self):
        self.orders = _make_orders_for_rfm()
        self.items = _make_items_for_rfm(self.orders)
        self.obs_date = pd.Timestamp("2023-12-01", tz="UTC")

    def test_one_row_per_customer(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        assert rfm["customer_id"].nunique() == rfm["customer_id"].count()

    def test_all_customers_present(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        expected_ids = set(self.orders["user_id"].unique())
        actual_ids = set(rfm["customer_id"].tolist())
        assert expected_ids == actual_ids

    def test_recency_non_negative(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        assert (rfm["recency"] >= 0).all()

    def test_frequency_positive(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        assert (rfm["frequency"] >= 1).all()

    def test_monetary_non_negative(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        assert (rfm["monetary"] >= 0).all()

    def test_customer_with_most_orders_has_highest_frequency(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        # Customer 3 has 3 orders (most)
        customer_3_freq = rfm[rfm["customer_id"] == 3]["frequency"].iloc[0]
        assert customer_3_freq == 3

    def test_most_recent_customer_has_lowest_recency(self):
        rfm = build_rfm_features(self.orders, self.items, self.obs_date)
        # Customer 5 ordered 2023-11-20, closest to obs_date 2023-12-01
        min_recency_customer = rfm.loc[rfm["recency"].idxmin(), "customer_id"]
        assert min_recency_customer == 5

    def test_observation_date_cannot_precede_purchase_data(self):
        with pytest.raises(ValueError, match="earlier than the latest completed order"):
            build_rfm_features(
                self.orders,
                self.items,
                pd.Timestamp("2023-11-01", tz="UTC"),
            )

    def test_non_purchasing_statuses_are_not_used_as_rfm_fallback(self):
        cancelled = self.orders.assign(status="cancelled")

        with pytest.raises(ValueError, match="No completed orders found"):
            build_rfm_features(cancelled, self.items, self.obs_date)


class TestRfmScoring:
    def test_scores_range_1_to_5(self):
        orders = _make_orders_for_rfm()
        items = _make_items_for_rfm(orders)
        obs = pd.Timestamp("2023-12-01", tz="UTC")
        rfm = build_rfm_features(orders, items, obs)
        rfm_scored = compute_rfm_scores(rfm)

        for col in ["r_score", "f_score", "m_score"]:
            assert rfm_scored[col].between(1, 5).all(), f"{col} out of range"

    def test_rfm_score_is_sum_of_components(self):
        orders = _make_orders_for_rfm()
        items = _make_items_for_rfm(orders)
        obs = pd.Timestamp("2023-12-01", tz="UTC")
        rfm = build_rfm_features(orders, items, obs)
        rfm_scored = compute_rfm_scores(rfm)

        computed_sum = rfm_scored["r_score"] + rfm_scored["f_score"] + rfm_scored["m_score"]
        assert (rfm_scored["rfm_score"] == computed_sum).all()

    def test_required_columns_present(self):
        orders = _make_orders_for_rfm()
        items = _make_items_for_rfm(orders)
        obs = pd.Timestamp("2023-12-01", tz="UTC")
        rfm = build_rfm_features(orders, items, obs)
        rfm_scored = compute_rfm_scores(rfm)

        required = {"customer_id", "recency", "frequency", "monetary",
                    "r_score", "f_score", "m_score", "rfm_score"}
        assert required.issubset(set(rfm_scored.columns))

    def test_equal_values_share_scores_independent_of_input_order(self):
        rfm = pd.DataFrame({
            "customer_id": range(1, 11),
            "recency": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
            "frequency": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "monetary": [100, 100, 100, 200, 200, 300, 300, 400, 400, 400],
        })

        forward = compute_rfm_scores(rfm).set_index("customer_id")[
            ["r_score", "f_score", "m_score"]
        ].sort_index()
        reversed_order = compute_rfm_scores(rfm.iloc[::-1]).set_index("customer_id")[
            ["r_score", "f_score", "m_score"]
        ].sort_index()

        pd.testing.assert_frame_equal(forward, reversed_order)
        for feature, score in (("recency", "r_score"), ("frequency", "f_score"), ("monetary", "m_score")):
            tied = rfm.groupby(feature)["customer_id"].apply(list)
            scored = forward[score]
            assert all(scored.loc[ids].nunique() == 1 for ids in tied)

    def test_constant_values_receive_one_shared_middle_score(self):
        rfm = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "recency": [5, 5, 5],
            "frequency": [2, 2, 2],
            "monetary": [100.0, 100.0, 100.0],
        })

        scored = compute_rfm_scores(rfm)

        assert (scored[["r_score", "f_score", "m_score"]] == 3).all().all()

    def test_missing_or_non_numeric_rfm_values_fail_clearly(self):
        rfm = pd.DataFrame({
            "customer_id": [1, 2],
            "recency": [1, None],
            "frequency": [1, 2],
            "monetary": [10, 20],
        })
        with pytest.raises(ValueError, match="finite numeric"):
            compute_rfm_scores(rfm)
