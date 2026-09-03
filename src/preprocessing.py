"""
CustomerIQ — Data Preprocessing.

Provides reusable cleaning functions that:
- Parse date columns
- Coerce numeric types
- Drop/flag duplicates with logging
- Handle missing values
- Filter invalid records

All operations are logged. Nothing is silently discarded.
"""
import logging

import pandas as pd

logger = logging.getLogger("customeriq.preprocessing")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_all(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply all cleaning steps to every table.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Raw DataFrames keyed by table name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Cleaned DataFrames.
    """
    cleaned: dict[str, pd.DataFrame] = {}

    cleaned["users"] = preprocess_users(tables["users"].copy())
    cleaned["orders"] = preprocess_orders(tables["orders"].copy())
    cleaned["order_items"] = preprocess_order_items(tables["order_items"].copy())
    cleaned["products"] = preprocess_products(tables["products"].copy())
    cleaned["events"] = preprocess_events(tables["events"].copy())
    cleaned["inventory_items"] = preprocess_inventory_items(tables["inventory_items"].copy())
    cleaned["distribution_centers"] = tables["distribution_centers"].copy()  # minimal cleaning

    return cleaned


# ---------------------------------------------------------------------------
# Table-specific preprocessors
# ---------------------------------------------------------------------------

def preprocess_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the users table."""
    original_len = len(df)

    # Parse dates
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Drop rows with missing user id
    df = _drop_missing_key(df, "id", "users")

    # Remove duplicate user ids (keep first occurrence)
    df = _drop_duplicates(df, "id", "users")

    # Standardise string columns
    for col in ["gender", "country", "state", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", pd.NA)

    # Age: coerce to numeric, drop impossible values
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        invalid_age = ((df["age"] < 0) | (df["age"] > 120))
        if invalid_age.sum() > 0:
            logger.warning("users: %d rows with invalid age values set to NaN.", invalid_age.sum())
            df.loc[invalid_age, "age"] = pd.NA

    logger.info("users: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the orders table."""
    original_len = len(df)

    # Parse dates
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "returned_at" in df.columns:
        df["returned_at"] = pd.to_datetime(df["returned_at"], errors="coerce", utc=True)
    if "shipped_at" in df.columns:
        df["shipped_at"] = pd.to_datetime(df["shipped_at"], errors="coerce", utc=True)
    if "delivered_at" in df.columns:
        df["delivered_at"] = pd.to_datetime(df["delivered_at"], errors="coerce", utc=True)

    # Drop rows missing critical IDs
    df = _drop_missing_key(df, "order_id", "orders")
    df = _drop_missing_key(df, "user_id", "orders")

    # Remove duplicate order ids
    df = _drop_duplicates(df, "order_id", "orders")

    # Standardise status
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.lower()

    # Drop rows where created_at could not be parsed
    bad_dates = df["created_at"].isna()
    if bad_dates.sum() > 0:
        logger.warning("orders: dropping %d rows with unparseable created_at.", bad_dates.sum())
        df = df[~bad_dates]

    logger.info("orders: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


def preprocess_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the order_items table."""
    original_len = len(df)

    # Parse dates
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "shipped_at" in df.columns:
        df["shipped_at"] = pd.to_datetime(df["shipped_at"], errors="coerce", utc=True)
    if "delivered_at" in df.columns:
        df["delivered_at"] = pd.to_datetime(df["delivered_at"], errors="coerce", utc=True)
    if "returned_at" in df.columns:
        df["returned_at"] = pd.to_datetime(df["returned_at"], errors="coerce", utc=True)

    # Coerce sale_price
    df["sale_price"] = pd.to_numeric(df["sale_price"], errors="coerce")

    # Drop rows with invalid prices (negative or null)
    bad_price = df["sale_price"].isna() | (df["sale_price"] < 0)
    if bad_price.sum() > 0:
        logger.warning(
            "order_items: dropping %d rows with null/negative sale_price.", bad_price.sum()
        )
        df = df[~bad_price]

    # Standardise status
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.strip().str.lower()

    # Drop rows missing critical IDs
    df = _drop_missing_key(df, "order_id", "order_items")
    df = _drop_missing_key(df, "user_id", "order_items")
    df = _drop_missing_key(df, "product_id", "order_items")

    logger.info("order_items: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


def preprocess_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the products table."""
    original_len = len(df)

    df = _drop_missing_key(df, "id", "products")
    df = _drop_duplicates(df, "id", "products")

    for col in ["cost", "retail_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["name", "category", "brand", "department"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    logger.info("products: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


def preprocess_events(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the events table."""
    original_len = len(df)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    # Drop rows missing user_id or event_type
    df = _drop_missing_key(df, "user_id", "events")

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()

    logger.info("events: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


def preprocess_inventory_items(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the inventory_items table."""
    original_len = len(df)

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if "sold_at" in df.columns:
        df["sold_at"] = pd.to_datetime(df["sold_at"], errors="coerce", utc=True)

    if "cost" in df.columns:
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")

    logger.info("inventory_items: %d → %d rows after cleaning.", original_len, len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_missing_key(df: pd.DataFrame, col: str, table_name: str) -> pd.DataFrame:
    """Drop rows where *col* is null, and log the count."""
    missing = df[col].isna()
    if missing.sum() > 0:
        logger.warning(
            "%s: dropping %d rows with null '%s'.", table_name, missing.sum(), col
        )
        df = df[~missing]
    return df


def _drop_duplicates(df: pd.DataFrame, key_col: str, table_name: str) -> pd.DataFrame:
    """Drop duplicate rows by *key_col*, keeping the first, and log."""
    dupes = df[key_col].duplicated()
    if dupes.sum() > 0:
        logger.warning(
            "%s: dropping %d duplicate rows by '%s'.", table_name, dupes.sum(), key_col
        )
        df = df[~dupes]
    return df
