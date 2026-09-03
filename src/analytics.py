"""
CustomerIQ — Analytics Module.

Computes:
1. Monthly KPIs (Revenue, Orders, Customers, AOV, New/Repeat customers)
2. Cohort Analysis (customer acquisition cohort × retention month heatmap)
3. Product & Category Analytics (revenue, volume, customer count by category)
"""
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger("customeriq.analytics")


# ---------------------------------------------------------------------------
# Monthly KPIs
# ---------------------------------------------------------------------------

def compute_monthly_kpis(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute monthly KPI metrics.

    Metrics
    -------
    - year_month: YYYY-MM label
    - total_revenue: sum of sale_price for completed orders
    - total_orders: number of distinct completed orders
    - unique_customers: distinct customers placing orders
    - avg_order_value: revenue / orders
    - new_customers: customers placing their FIRST order this month
    - repeat_customers: customers who had a prior order before this month
    - repeat_customer_rate: repeat / total unique

    Parameters
    ----------
    orders, order_items, users : pd.DataFrame
        Cleaned tables.

    Returns
    -------
    pd.DataFrame
        One row per year-month.
    """
    completed_statuses = {"complete", "shipped", "processing"}
    comp_orders = orders[orders["status"].isin(completed_statuses)].copy()
    comp_items = order_items[order_items["status"].isin(completed_statuses)].copy()

    # Parse dates if needed
    comp_orders["created_at"] = pd.to_datetime(comp_orders["created_at"], utc=True)
    comp_items["created_at"] = pd.to_datetime(comp_items["created_at"], utc=True)

    comp_orders["year_month"] = comp_orders["created_at"].dt.to_period("M")

    # Revenue per order from items
    order_revenue = (
        comp_items.groupby("order_id")["sale_price"]
        .sum()
        .reset_index(name="order_revenue")
    )
    comp_orders = comp_orders.merge(order_revenue, on="order_id", how="left")
    comp_orders["order_revenue"] = comp_orders["order_revenue"].fillna(0)

    # Monthly aggregates
    monthly = comp_orders.groupby("year_month").agg(
        total_revenue=("order_revenue", "sum"),
        total_orders=("order_id", "nunique"),
        unique_customers=("user_id", "nunique"),
    ).reset_index()

    monthly["avg_order_value"] = (monthly["total_revenue"] / monthly["total_orders"]).round(2)

    # New vs repeat customers
    # First order per customer
    first_order = comp_orders.groupby("user_id")["year_month"].min().reset_index(name="first_month")
    comp_orders = comp_orders.merge(first_order, on="user_id", how="left")
    comp_orders["is_new"] = comp_orders["year_month"] == comp_orders["first_month"]

    new_cust = comp_orders.groupby("year_month").apply(
        lambda g: g[g["is_new"]]["user_id"].nunique()
    ).reset_index(name="new_customers")
    repeat_cust = comp_orders.groupby("year_month").apply(
        lambda g: g[~g["is_new"]]["user_id"].nunique()
    ).reset_index(name="repeat_customers")

    monthly = monthly.merge(new_cust, on="year_month", how="left")
    monthly = monthly.merge(repeat_cust, on="year_month", how="left")
    monthly["repeat_customer_rate"] = (
        monthly["repeat_customers"] / monthly["unique_customers"].clip(lower=1)
    ).round(4)

    monthly["year_month"] = monthly["year_month"].astype(str)
    monthly = monthly.sort_values("year_month").reset_index(drop=True)

    logger.info("Monthly KPIs computed: %d months.", len(monthly))
    return monthly


# ---------------------------------------------------------------------------
# Cohort Analysis
# ---------------------------------------------------------------------------

def compute_cohort_retention(
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute cohort retention matrix.

    Methodology:
    - Cohort = month of a customer's first order
    - For each subsequent month, track what fraction of cohort customers
      placed at least one order.

    Returns
    -------
    pd.DataFrame
        Wide format: rows = cohort_month, columns = months_since (0, 1, 2, ...)
        Values = retention rate (0.0 – 1.0).
    """
    completed_statuses = {"complete", "shipped", "processing"}
    comp = orders[orders["status"].isin(completed_statuses)].copy()
    comp["created_at"] = pd.to_datetime(comp["created_at"], utc=True)
    comp["order_month"] = comp["created_at"].dt.to_period("M")

    # Cohort = first order month per customer
    first_order = (
        comp.groupby("user_id")["order_month"]
        .min()
        .reset_index(name="cohort_month")
    )
    comp = comp.merge(first_order, on="user_id", how="left")

    # Months since first order
    comp["months_since"] = (comp["order_month"] - comp["cohort_month"]).apply(
        lambda x: x.n if hasattr(x, "n") else 0
    )

    # Cohort size (number of unique customers in each cohort)
    cohort_sizes = (
        first_order.groupby("cohort_month")["user_id"]
        .nunique()
        .reset_index(name="cohort_size")
    )

    # Active customers per cohort per months_since
    cohort_activity = (
        comp.groupby(["cohort_month", "months_since"])["user_id"]
        .nunique()
        .reset_index(name="active_customers")
    )
    cohort_activity = cohort_activity.merge(cohort_sizes, on="cohort_month", how="left")
    cohort_activity["retention_rate"] = (
        cohort_activity["active_customers"] / cohort_activity["cohort_size"]
    ).round(4)

    # Pivot to wide format
    retention_matrix = cohort_activity.pivot_table(
        index="cohort_month",
        columns="months_since",
        values="retention_rate",
    )
    retention_matrix.index = retention_matrix.index.astype(str)
    retention_matrix.columns = [int(c) for c in retention_matrix.columns]

    logger.info(
        "Cohort retention matrix: %d cohorts × %d months.",
        len(retention_matrix), len(retention_matrix.columns),
    )
    return retention_matrix


# ---------------------------------------------------------------------------
# Product & Category Analytics
# ---------------------------------------------------------------------------

def compute_category_analytics(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute revenue, volume, and customer metrics by product category.

    Parameters
    ----------
    order_items, products : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: category, total_revenue, order_count, unique_customers,
                 avg_selling_price, product_count.
    """
    completed_statuses = {"complete", "shipped", "processing"}
    oi = order_items[order_items["status"].isin(completed_statuses)].copy()
    oi["sale_price"] = pd.to_numeric(oi["sale_price"], errors="coerce")

    # Join category from products
    oi = oi.merge(
        products[["id", "name", "category", "brand"]].rename(columns={"id": "product_id"}),
        on="product_id",
        how="left",
    )
    oi = oi.dropna(subset=["category"])

    cat_agg = oi.groupby("category").agg(
        total_revenue=("sale_price", "sum"),
        order_count=("order_id", "nunique"),
        unique_customers=("user_id", "nunique"),
        avg_selling_price=("sale_price", "mean"),
        product_count=("product_id", "nunique"),
    ).reset_index()

    cat_agg["total_revenue"] = cat_agg["total_revenue"].round(2)
    cat_agg["avg_selling_price"] = cat_agg["avg_selling_price"].round(2)
    cat_agg = cat_agg.sort_values("total_revenue", ascending=False).reset_index(drop=True)

    logger.info("Category analytics computed: %d categories.", len(cat_agg))
    return cat_agg


def compute_product_analytics(
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Compute product-level performance metrics.

    Returns
    -------
    pd.DataFrame
        top_n products by revenue, with: product_id, name, category, brand,
        total_revenue, order_count, unique_customers, avg_selling_price.
    """
    completed_statuses = {"complete", "shipped", "processing"}
    oi = order_items[order_items["status"].isin(completed_statuses)].copy()
    oi["sale_price"] = pd.to_numeric(oi["sale_price"], errors="coerce")

    prod_agg = oi.groupby("product_id").agg(
        total_revenue=("sale_price", "sum"),
        order_count=("order_id", "nunique"),
        unique_customers=("user_id", "nunique"),
        avg_selling_price=("sale_price", "mean"),
    ).reset_index()

    prod_agg = prod_agg.merge(
        products[["id", "name", "category", "brand"]].rename(columns={"id": "product_id"}),
        on="product_id",
        how="left",
    )
    prod_agg["total_revenue"] = prod_agg["total_revenue"].round(2)
    prod_agg["avg_selling_price"] = prod_agg["avg_selling_price"].round(2)
    prod_agg = prod_agg.sort_values("total_revenue", ascending=False).head(top_n).reset_index(drop=True)

    logger.info("Product analytics computed: top %d products.", len(prod_agg))
    return prod_agg
