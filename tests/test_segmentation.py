"""
Tests for customer segmentation.

Uses small synthetic data.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.segmentation import (
    compute_rfm_scores,
    assign_segments,
    estimate_clv,
    SEGMENT_RULES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_rfm_df(n: int = 50) -> pd.DataFrame:
    """Synthetic RFM DataFrame with varied behaviour."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "last_purchase_date": pd.date_range("2023-01-01", periods=n, freq="W"),
        "recency": rng.integers(1, 365, n).astype(float),
        "frequency": rng.integers(1, 20, n).astype(float),
        "monetary": rng.uniform(10, 5000, n).round(2),
    })


def make_customer_agg(n: int = 50) -> pd.DataFrame:
    """Synthetic customer aggregates for CLV testing."""
    import numpy as np
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "total_orders": rng.integers(1, 15, n),
        "total_revenue": rng.uniform(50, 5000, n).round(2),
        "avg_order_value": rng.uniform(20, 500, n).round(2),
        "tenure_days": rng.integers(30, 1000, n).astype(float),
        "first_order_date": pd.date_range("2021-01-01", periods=n, freq="W"),
        "last_order_date": pd.date_range("2022-01-01", periods=n, freq="W"),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignSegments:
    def setup_method(self):
        self.rfm = make_rfm_df(60)
        self.rfm_scored = compute_rfm_scores(self.rfm)

    def test_all_customers_get_segment(self):
        result = assign_segments(self.rfm_scored)
        assert result["segment"].notna().all()
        assert (result["segment"] != "").all()

    def test_segment_values_are_known(self):
        result = assign_segments(self.rfm_scored)
        valid_segments = {r["name"] for r in SEGMENT_RULES} | {"Other"}
        for seg in result["segment"].unique():
            assert seg in valid_segments, f"Unexpected segment: {seg}"

    def test_champions_have_high_scores(self):
        result = assign_segments(self.rfm_scored)
        champions = result[result["segment"] == "Champions"]
        if len(champions) > 0:
            assert (champions["r_score"] >= 4).all()
            assert (champions["f_score"] >= 4).all()
            assert (champions["m_score"] >= 4).all()

    def test_segment_column_preserved_with_other_columns(self):
        result = assign_segments(self.rfm_scored)
        expected_cols = {"customer_id", "recency", "frequency", "monetary",
                         "r_score", "f_score", "m_score", "rfm_score", "segment"}
        assert expected_cols.issubset(set(result.columns))

    def test_no_duplicate_customers(self):
        result = assign_segments(self.rfm_scored)
        assert result["customer_id"].nunique() == len(result)


class TestEstimateCLV:
    def test_clv_non_negative(self):
        agg = make_customer_agg(30)
        result = estimate_clv(agg)
        assert (result["estimated_clv"] >= 0).all()

    def test_clv_column_created(self):
        agg = make_customer_agg(30)
        result = estimate_clv(agg)
        assert "estimated_clv" in result.columns

    def test_high_frequency_customer_has_higher_clv(self):
        """Customer with higher annual frequency should have higher CLV."""
        agg = pd.DataFrame({
            "customer_id": [1, 2],
            "total_orders": [10, 2],
            "total_revenue": [1000, 200],
            "avg_order_value": [100, 100],
            "tenure_days": [365.0, 365.0],
            "first_order_date": [pd.Timestamp("2022-01-01")] * 2,
            "last_order_date": [pd.Timestamp("2023-01-01")] * 2,
        })
        result = estimate_clv(agg)
        clv_high = result[result["customer_id"] == 1]["estimated_clv"].iloc[0]
        clv_low = result[result["customer_id"] == 2]["estimated_clv"].iloc[0]
        assert clv_high > clv_low
