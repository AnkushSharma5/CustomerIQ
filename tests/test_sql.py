import pandas as pd

from src.analytics import compute_monthly_kpis
from src.duckdb_manager import load_tables_to_duckdb, run_sql_file


def _monthly_fixture():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "user_id": [1, 1, 2],
        "status": ["complete"] * 3,
        "created_at": pd.to_datetime(
            ["2024-01-05", "2024-02-05", "2024-02-10"], utc=True
        ),
    })
    items = pd.DataFrame({
        "order_id": [1, 1, 2, 3],
        "user_id": [1, 1, 1, 2],
        "product_id": [1, 2, 1, 2],
        "status": ["complete"] * 4,
        "created_at": pd.to_datetime(
            ["2024-01-05", "2024-01-05", "2024-02-05", "2024-02-10"], utc=True
        ),
        "sale_price": [10.0, 30.0, 20.0, 50.0],
    })
    products = pd.DataFrame({
        "id": [1, 2],
        "name": ["One", "Two"],
        "category": ["A", "B"],
        "brand": ["Brand A", "Brand B"],
    })
    return orders, items, products


def test_monthly_kpi_sql_matches_pandas_and_dashboard_schema(tmp_path):
    orders, items, products = _monthly_fixture()
    db_path = tmp_path / "customeriq.duckdb"
    load_tables_to_duckdb({"orders": orders, "order_items": items, "products": products}, db_path)

    sql_result = run_sql_file("monthly_kpis.sql", db_path).set_index("year_month")
    pandas_result = compute_monthly_kpis(orders, items, pd.DataFrame()).set_index("year_month")

    assert set(sql_result.columns) == set(pandas_result.columns)
    pd.testing.assert_series_equal(
        sql_result["total_revenue"].astype(float).sort_index(),
        pandas_result["total_revenue"].astype(float).sort_index(),
        check_names=False,
    )
    assert sql_result.loc["2024-01", "total_orders"] == 1
    assert sql_result.loc["2024-02", "avg_order_value"] == 35.0
    assert sql_result.loc["2024-02", "new_customers"] == 1
    assert sql_result.loc["2024-02", "repeat_customers"] == 1
    assert sql_result.loc["2024-02", "repeat_customer_rate"] == 0.5


def test_rfm_sql_keeps_customer_order_grain_and_counts_line_items_once(tmp_path):
    orders, items, products = _monthly_fixture()
    db_path = tmp_path / "customeriq.duckdb"
    load_tables_to_duckdb({"orders": orders, "order_items": items, "products": products}, db_path)

    rfm = run_sql_file("rfm.sql", db_path).set_index("customer_id")

    assert rfm.loc[1, "frequency"] == 2
    assert rfm.loc[1, "monetary"] == 60.0
    assert rfm.loc[1, "recency"] == 5
    assert rfm.loc[2, "frequency"] == 1
    assert rfm.loc[2, "monetary"] == 50.0
    assert rfm.loc[2, "recency"] == 0
