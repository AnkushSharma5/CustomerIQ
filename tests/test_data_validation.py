"""
Tests for data validation.

Uses small synthetic DataFrames so no real dataset is required.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_validator import (
    validate_tables,
    _validate_single_table,
    _check_referential_integrity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_users(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(1, n + 1),
        "first_name": [f"First{i}" for i in range(n)],
        "last_name": [f"Last{i}" for i in range(n)],
        "email": [f"user{i}@test.com" for i in range(n)],
        "age": [25 + i for i in range(n)],
        "gender": ["M" if i % 2 == 0 else "F" for i in range(n)],
        "country": ["US"] * n,
        "state": ["CA"] * n,
        "city": ["LA"] * n,
        "created_at": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
    })


def make_orders(n: int = 20, user_ids=None) -> pd.DataFrame:
    if user_ids is None:
        user_ids = list(range(1, 11))
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "user_id": [user_ids[i % len(user_ids)] for i in range(n)],
        "status": ["complete"] * n,
        "created_at": pd.date_range("2021-01-01", periods=n, freq="W").astype(str),
        "num_of_item": [2] * n,
    })


def make_order_items(n: int = 40, order_ids=None, user_ids=None, product_ids=None) -> pd.DataFrame:
    if order_ids is None:
        order_ids = list(range(1, 21))
    if user_ids is None:
        user_ids = list(range(1, 11))
    if product_ids is None:
        product_ids = list(range(1, 11))
    return pd.DataFrame({
        "id": range(1, n + 1),
        "order_id": [order_ids[i % len(order_ids)] for i in range(n)],
        "user_id": [user_ids[i % len(user_ids)] for i in range(n)],
        "product_id": [product_ids[i % len(product_ids)] for i in range(n)],
        "status": ["complete"] * n,
        "created_at": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
        "sale_price": [50.0 + i for i in range(n)],
    })


def make_products(n: int = 10) -> pd.DataFrame:
    cats = (["Clothing", "Electronics", "Books"] * (n // 3 + 1))[:n]
    depts = (["Men", "Women"] * (n // 2 + 1))[:n]
    return pd.DataFrame({
        "id": list(range(1, n + 1)),
        "name": [f"Product {i}" for i in range(n)],
        "category": cats,
        "brand": [f"Brand{i}" for i in range(n)],
        "department": depts,
        "cost": [20.0 + i for i in range(n)],
        "retail_price": [50.0 + i for i in range(n)],
    })


def make_events(n: int = 30) -> pd.DataFrame:
    event_types = (["purchase", "view", "cart"] * (n // 3 + 1))[:n]
    return pd.DataFrame({
        "id": list(range(1, n + 1)),
        "user_id": [i % 10 + 1 for i in range(n)],
        "event_type": event_types,
        "created_at": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
    })


def make_inventory(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(1, n + 1),
        "product_id": range(1, n + 1),
        "created_at": pd.date_range("2020-01-01", periods=n, freq="W").astype(str),
        "sold_at": pd.date_range("2021-01-01", periods=n, freq="W").astype(str),
        "cost": [20.0 + i for i in range(n)],
    })


def make_dc(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(1, n + 1),
        "name": [f"DC {i}" for i in range(n)],
        "latitude": [37.0 + i for i in range(n)],
        "longitude": [-122.0 - i for i in range(n)],
    })


def make_all_tables() -> dict:
    return {
        "users": make_users(),
        "orders": make_orders(),
        "order_items": make_order_items(),
        "products": make_products(),
        "events": make_events(),
        "inventory_items": make_inventory(),
        "distribution_centers": make_dc(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateTables:
    def test_valid_tables_pass(self):
        tables = make_all_tables()
        report = validate_tables(tables)
        assert report["passed"] is True

    def test_empty_table_fails(self):
        tables = make_all_tables()
        tables["users"] = pd.DataFrame(columns=tables["users"].columns)
        report = validate_tables(tables)
        assert report["passed"] is False

    def test_missing_column_fails(self):
        tables = make_all_tables()
        tables["users"] = tables["users"].drop(columns=["email"])
        report = validate_tables(tables)
        assert report["passed"] is False
        issues = report["tables"]["users"]["issues"]
        assert any("email" in issue for issue in issues)

    def test_referential_integrity_order_items(self):
        tables = make_all_tables()
        # Introduce orphan order_id
        bad_row = pd.DataFrame([{
            "id": 9999, "order_id": 9999, "user_id": 1, "product_id": 1,
            "status": "complete", "created_at": "2021-01-01", "sale_price": 50.0,
        }])
        tables["order_items"] = pd.concat([tables["order_items"], bad_row], ignore_index=True)
        report = validate_tables(tables)
        ri_errors = report.get("referential_integrity", [])
        assert len(ri_errors) > 0

    def test_negative_price_warning(self):
        tables = make_all_tables()
        tables["order_items"].loc[0, "sale_price"] = -10.0
        report = validate_tables(tables)
        warnings = report["tables"]["order_items"]["warnings"]
        assert any("negative" in w.lower() for w in warnings)

    def test_validation_report_structure(self):
        tables = make_all_tables()
        report = validate_tables(tables)
        assert "tables" in report
        assert "passed" in report
        for tname in tables:
            assert tname in report["tables"]
            t = report["tables"][tname]
            assert "passed" in t
            assert "row_count" in t
            assert "issues" in t
            assert "warnings" in t
