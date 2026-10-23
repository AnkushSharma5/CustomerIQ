"""
CustomerIQ — Feature Engineering.

Builds customer-level features from cleaned order data.

Key design decision — churn feature window:
  - Feature observation period: all orders up to the CUTOFF DATE
  - Prediction window: CHURN_FUTURE_DAYS days after the cutoff date
  - The cutoff date is derived as: max(order date) - CHURN_FUTURE_DAYS
    so we have both historical features AND a future label.

This design prevents target leakage: no future purchase information is
used in the feature set.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

from src.config import CHURN_FUTURE_DAYS, CHURN_CUTOFF_DATE

logger = logging.getLogger("customeriq.feature_engineering")

COMPLETED_STATUSES = {"complete", "shipped", "processing"}


def _items_for_completed_orders(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Return completed line items attributed to their parent order owner."""
    completed_orders = orders.loc[
        orders["status"].isin(COMPLETED_STATUSES), ["order_id", "user_id"]
    ].drop_duplicates("order_id")
    items = order_items.loc[
        order_items["status"].isin(COMPLETED_STATUSES)
        & order_items["order_id"].isin(completed_orders["order_id"])
    ].copy()

    # The order header is the source of truth for customer ownership. Validation
    # reports mismatches, and this assignment prevents item-level IDs from
    # silently moving revenue between customers if a caller skips validation.
    order_owners = completed_orders.set_index("order_id")["user_id"]
    items["user_id"] = items["order_id"].map(order_owners)
    return items


# ---------------------------------------------------------------------------
# RFM features
# ---------------------------------------------------------------------------

def build_rfm_features(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    observation_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build RFM (Recency, Frequency, Monetary) features for each customer.

    Parameters
    ----------
    orders : pd.DataFrame
        Cleaned orders table. Must have: user_id, created_at, status.
    order_items : pd.DataFrame
        Cleaned order_items table. Must have: order_id, user_id, sale_price, status.
    observation_date : pd.Timestamp, optional
        The "as-of" date for RFM computation. Defaults to max order date.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        customer_id, recency, frequency, monetary
    """
    # Ensure orders["created_at"] is datetime
    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True)

    completed_orders = orders[orders["status"].isin(COMPLETED_STATUSES)].copy()
    if completed_orders.empty:
        raise ValueError("No completed orders found; RFM cannot be computed.")
    completed_items = _items_for_completed_orders(orders, order_items)

    # Observation date: use max date in dataset if None
    if observation_date is None:
        observation_date = completed_orders["created_at"].max()
    else:
        observation_date = pd.Timestamp(observation_date)

    if observation_date.tzinfo is None:
        observation_date = observation_date.tz_localize("UTC")
    else:
        observation_date = observation_date.tz_convert("UTC")
    latest_order_date = completed_orders["created_at"].max()
    if observation_date < latest_order_date:
        raise ValueError(
            "observation_date cannot be earlier than the latest completed order."
        )
    logger.info("RFM observation date: %s", observation_date)

    # Recency: days since last purchase
    last_purchase = (
        completed_orders.groupby("user_id")["created_at"]
        .max()
        .reset_index()
        .rename(columns={"user_id": "customer_id", "created_at": "last_purchase_date"})
    )
    last_purchase["recency"] = (
        (observation_date - last_purchase["last_purchase_date"])
        .dt.total_seconds() / 86400
    ).round(1)

    # Frequency: number of distinct completed orders
    frequency = (
        completed_orders.groupby("user_id")["order_id"]
        .nunique()
        .reset_index()
        .rename(columns={"user_id": "customer_id", "order_id": "frequency"})
    )

    # Monetary: total revenue from order_items
    monetary = (
        completed_items.groupby("user_id")["sale_price"]
        .sum()
        .reset_index()
        .rename(columns={"user_id": "customer_id", "sale_price": "monetary"})
    )

    # Merge
    rfm = last_purchase[["customer_id", "last_purchase_date", "recency"]].merge(
        frequency, on="customer_id", how="left"
    ).merge(
        monetary, on="customer_id", how="left"
    )

    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["monetary"].fillna(0.0).round(2)

    logger.info("RFM features built for %d customers.", len(rfm))
    return rfm


# ---------------------------------------------------------------------------
# Churn features
# ---------------------------------------------------------------------------

def build_churn_features(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    users: pd.DataFrame,
    future_days: int = CHURN_FUTURE_DAYS,
    cutoff_date_str: Optional[str] = CHURN_CUTOFF_DATE,
) -> pd.DataFrame:
    """
    Build customer features for churn prediction, respecting temporal isolation.

    Methodology
    -----------
    1. Derive CUTOFF DATE = max(order_date) - future_days
       (or use the provided override date)
    2. FEATURE WINDOW = all orders strictly BEFORE the cutoff date
    3. LABEL WINDOW   = orders strictly AFTER the cutoff date
    4. A customer is "churned" if they have ZERO orders in the label window.
    5. Only customers with at least 1 order before the cutoff are included.

    This structure prevents target leakage — features are computed solely
    from historical data; the label is computed from future data.

    Parameters
    ----------
    orders : pd.DataFrame
        Cleaned orders table.
    order_items : pd.DataFrame
        Cleaned order_items table.
    users : pd.DataFrame
        Cleaned users table.
    future_days : int
        Length of the future prediction window in days.
    cutoff_date_str : str, optional
        Override cutoff date as "YYYY-MM-DD". If None, auto-derived.

    Returns
    -------
    pd.DataFrame
        One row per customer with features and 'churned' label (0 or 1).
    """
    # Ensure timezone-aware timestamps
    if future_days <= 0:
        raise ValueError("future_days must be a positive integer.")

    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True)
    completed_statuses = COMPLETED_STATUSES
    completed_orders = orders[orders["status"].isin(completed_statuses)].copy()
    if completed_orders.empty:
        raise ValueError("No completed orders are available to build churn features.")

    max_date = completed_orders["created_at"].max()
    if pd.isna(max_date):
        raise ValueError("Completed orders do not contain valid timestamps.")

    # Cutoff date
    if cutoff_date_str:
        cutoff = pd.Timestamp(cutoff_date_str)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    else:
        cutoff = max_date - pd.Timedelta(days=future_days)

    future_end = cutoff + pd.Timedelta(days=future_days)
    if max_date < future_end:
        raise ValueError(
            f"The completed-order history ends at {max_date.isoformat()}, before the "
            f"full {future_days}-day label window ends at {future_end.isoformat()}."
        )

    logger.info(
        "Churn cutoff: %s | Future window: %d days (until %s)",
        cutoff.date(), future_days, max_date.date(),
    )

    # Historical orders (before cutoff)
    hist = completed_orders[completed_orders["created_at"] < cutoff].copy()
    # Label only purchases in the configured horizon, including its endpoint.
    future = completed_orders[
        (completed_orders["created_at"] >= cutoff)
        & (completed_orders["created_at"] <= future_end)
    ].copy()

    if len(hist) == 0:
        raise ValueError(
            "No historical orders found before the cutoff date. "
            "The dataset may be too small or the cutoff date is incorrect."
        )

    # Customers who have history (eligible for prediction)
    eligible_customers = hist["user_id"].unique()

    # Feature: per-customer aggregations from historical window
    order_counts = hist.groupby("user_id")["order_id"].nunique().rename("frequency")
    last_order = hist.groupby("user_id")["created_at"].max().rename("last_order_date")
    first_order = hist.groupby("user_id")["created_at"].min().rename("first_order_date")

    hist_items = _items_for_completed_orders(hist, order_items)
    hist_items["sale_price"] = pd.to_numeric(hist_items["sale_price"], errors="coerce")

    order_totals = hist_items.groupby(["user_id", "order_id"], as_index=False).agg(
        order_total=("sale_price", "sum")
    )
    monetary = order_totals.groupby("user_id")["order_total"].sum().rename("monetary")
    avg_order_value = order_totals.groupby("user_id")["order_total"].mean().rename("avg_order_value")
    n_unique_products = hist_items.groupby("user_id")["product_id"].nunique().rename("n_unique_products")

    # Merge all features
    feat = (
        pd.DataFrame(index=eligible_customers)
        .join(order_counts, how="left")
        .join(last_order, how="left")
        .join(first_order, how="left")
        .join(monetary, how="left")
        .join(avg_order_value, how="left")
        .join(n_unique_products, how="left")
    )
    feat.index.name = "user_id"
    feat = feat.reset_index()

    feat["recency"] = (cutoff - feat["last_order_date"]).dt.total_seconds() / 86400
    feat["tenure_days"] = (feat["last_order_date"] - feat["first_order_date"]).dt.total_seconds() / 86400

    # Join user demographics
    user_info = users[["id", "age", "gender", "country"]].rename(columns={"id": "user_id"}).copy()
    user_info["age"] = pd.to_numeric(user_info["age"], errors="coerce")
    feat = feat.merge(user_info, on="user_id", how="left")

    # Encode gender
    feat["is_male"] = (feat["gender"].astype(str).str.lower() == "m").astype(int)

    # Churn label: 0 = active (purchased after cutoff), 1 = churned
    future_buyers = set(future["user_id"].unique())
    feat["churned"] = feat["user_id"].apply(lambda uid: 0 if uid in future_buyers else 1)

    # Fill remaining nulls
    numeric_cols = ["frequency", "monetary", "avg_order_value", "n_unique_products",
                    "recency", "tenure_days", "age"]
    for col in numeric_cols:
        if col in feat.columns:
            feat[col] = feat[col].fillna(0)

    churn_rate = feat["churned"].mean()
    logger.info(
        "Churn features built: %d customers | churn rate: %.1f%%",
        len(feat), churn_rate * 100,
    )
    return feat


# ---------------------------------------------------------------------------
# Customer-level aggregation features (for CLV, segmentation, dashboard)
# ---------------------------------------------------------------------------

def build_customer_aggregates(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a comprehensive customer-level feature table for analytics.

    Returns
    -------
    pd.DataFrame
        One row per customer with purchase statistics.
    """
    orders = orders.copy()
    orders["created_at"] = pd.to_datetime(orders["created_at"], utc=True)

    comp_orders = orders[orders["status"].isin(COMPLETED_STATUSES)].copy()
    comp_items = _items_for_completed_orders(orders, order_items)

    # Order-level aggregations
    order_agg = comp_orders.groupby("user_id").agg(
        total_orders=("order_id", "nunique"),
        first_order_date=("created_at", "min"),
        last_order_date=("created_at", "max"),
    ).reset_index()

    # Item-level aggregations
    item_agg = comp_items.groupby("user_id").agg(
        total_revenue=("sale_price", "sum"),
        n_unique_products=("product_id", "nunique"),
    ).reset_index()

    order_totals = comp_items.groupby(["user_id", "order_id"], as_index=False).agg(
        order_total=("sale_price", "sum")
    )
    avg_order_value = order_totals.groupby("user_id")["order_total"].mean().rename("avg_order_value")
    item_agg = item_agg.merge(avg_order_value, on="user_id", how="left")

    # Merge
    agg = order_agg.merge(item_agg, on="user_id", how="left")

    # Observation date
    obs_date = comp_orders["created_at"].max()
    if obs_date.tzinfo is None:
        obs_date = obs_date.tz_localize("UTC")

    agg["recency"] = (
        (obs_date - agg["last_order_date"]).dt.total_seconds() / 86400
    ).round(1)
    agg["tenure_days"] = (
        (agg["last_order_date"] - agg["first_order_date"]).dt.total_seconds() / 86400
    ).round(1)

    # Join user info
    user_cols = [c for c in ["id", "first_name", "last_name", "email", "age",
                              "gender", "country", "state", "city", "created_at"]
                 if c in users.columns]
    user_info = users[user_cols].rename(columns={"id": "user_id"})
    agg = agg.merge(user_info, on="user_id", how="left")
    agg = agg.rename(columns={"user_id": "customer_id"})

    # Fill nulls
    agg["total_orders"] = agg["total_orders"].fillna(0).astype(int)
    agg["total_revenue"] = agg["total_revenue"].fillna(0.0).round(2)
    agg["avg_order_value"] = agg["avg_order_value"].fillna(0.0).round(2)

    logger.info("Customer aggregates built for %d customers.", len(agg))
    return agg
