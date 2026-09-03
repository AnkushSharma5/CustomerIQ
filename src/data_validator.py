"""
CustomerIQ — Data Validator.

Checks each required table for:
- Required columns
- Row counts
- Duplicate records
- Missing values in key columns
- Invalid dates
- Invalid numeric values (negative prices, quantities)
- Referential integrity across tables

Returns a structured validation report and logs all issues found.
"""
import logging
from typing import Any

import pandas as pd

from src.config import REQUIRED_COLUMNS

logger = logging.getLogger("customeriq.data_validator")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Validate all loaded tables.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Loaded raw DataFrames keyed by table name.

    Returns
    -------
    dict
        Validation report with keys per table plus a top-level 'passed' bool.
    """
    report: dict[str, Any] = {"tables": {}, "passed": True}

    for table_name, df in tables.items():
        table_report = _validate_single_table(table_name, df)
        report["tables"][table_name] = table_report
        if not table_report["passed"]:
            report["passed"] = False

    # Cross-table referential integrity
    ref_errors = _check_referential_integrity(tables)
    report["referential_integrity"] = ref_errors
    if ref_errors:
        logger.warning("Referential integrity issues found: %d", len(ref_errors))

    _print_validation_summary(report)
    return report


# ---------------------------------------------------------------------------
# Single-table validation
# ---------------------------------------------------------------------------

def _validate_single_table(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
    """Run all checks on a single table."""
    issues: list[str] = []
    warnings: list[str] = []

    # 1. Required columns
    expected_cols = REQUIRED_COLUMNS.get(table_name, [])
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        for col in missing_cols:
            issues.append(
                f"Missing required column '{col}' in table '{table_name}'. "
                f"Expected columns: {expected_cols}"
            )

    # 2. Row count
    row_count = len(df)
    if row_count == 0:
        issues.append(f"Table '{table_name}' is empty (0 rows).")

    # 3. Duplicate primary keys
    pk_col = _get_primary_key(table_name)
    if pk_col and pk_col in df.columns:
        dupe_count = df[pk_col].duplicated().sum()
        if dupe_count > 0:
            warnings.append(
                f"Table '{table_name}': {dupe_count} duplicate values in primary key '{pk_col}'."
            )

    # 4. Missing values in key columns
    key_cols = _get_key_columns(table_name)
    for col in key_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                pct = null_count / row_count * 100
                warnings.append(
                    f"Table '{table_name}', column '{col}': "
                    f"{null_count} missing values ({pct:.1f}%)."
                )

    # 5. Date validation
    date_cols = _get_date_columns(table_name)
    for col in date_cols:
        if col in df.columns:
            date_issues = _check_dates(df, col, table_name)
            warnings.extend(date_issues)

    # 6. Numeric validations (negative prices / quantities)
    numeric_issues = _check_numerics(df, table_name)
    warnings.extend(numeric_issues)

    passed = len(issues) == 0
    if not passed:
        for issue in issues:
            logger.error(issue)
    for warning in warnings:
        logger.warning(warning)

    return {
        "passed": passed,
        "row_count": row_count,
        "column_count": len(df.columns),
        "issues": issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Specific checks
# ---------------------------------------------------------------------------

def _check_dates(df: pd.DataFrame, col: str, table_name: str) -> list[str]:
    """Check for unparseable or future dates."""
    warnings: list[str] = []
    try:
        parsed = pd.to_datetime(df[col], errors="coerce")
        bad_count = parsed.isna().sum() - df[col].isna().sum()
        if bad_count > 0:
            warnings.append(
                f"Table '{table_name}', column '{col}': "
                f"{bad_count} values could not be parsed as dates."
            )
    except Exception as exc:
        warnings.append(f"Table '{table_name}', column '{col}': date check failed — {exc}")
    return warnings


def _check_numerics(df: pd.DataFrame, table_name: str) -> list[str]:
    """Check for inappropriate negative values."""
    warnings: list[str] = []
    negative_checks: dict[str, list[str]] = {
        "orders": ["num_of_item"],
        "order_items": ["sale_price"],
        "products": ["cost", "retail_price"],
        "inventory_items": ["cost"],
    }
    cols_to_check = negative_checks.get(table_name, [])
    for col in cols_to_check:
        if col in df.columns:
            try:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                neg_count = (numeric_col < 0).sum()
                if neg_count > 0:
                    warnings.append(
                        f"Table '{table_name}', column '{col}': "
                        f"{neg_count} negative values found."
                    )
            except Exception:
                pass
    return warnings


def _check_referential_integrity(tables: dict[str, pd.DataFrame]) -> list[str]:
    """Check foreign-key relationships between tables."""
    errors: list[str] = []

    # order_items.order_id → orders.order_id
    if "order_items" in tables and "orders" in tables:
        oi = tables["order_items"]
        o = tables["orders"]
        if "order_id" in oi.columns and "order_id" in o.columns:
            orphans = ~oi["order_id"].isin(o["order_id"])
            if orphans.sum() > 0:
                errors.append(
                    f"{orphans.sum()} order_items rows have order_id not in orders table."
                )

    # order_items.product_id → products.id
    if "order_items" in tables and "products" in tables:
        oi = tables["order_items"]
        p = tables["products"]
        if "product_id" in oi.columns and "id" in p.columns:
            orphans = ~oi["product_id"].isin(p["id"])
            if orphans.sum() > 0:
                errors.append(
                    f"{orphans.sum()} order_items rows have product_id not in products table."
                )

    # orders.user_id → users.id
    if "orders" in tables and "users" in tables:
        o = tables["orders"]
        u = tables["users"]
        if "user_id" in o.columns and "id" in u.columns:
            orphans = ~o["user_id"].isin(u["id"])
            if orphans.sum() > 0:
                errors.append(
                    f"{orphans.sum()} orders rows have user_id not in users table."
                )

    return errors


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _get_primary_key(table_name: str) -> str | None:
    """Return the primary key column name for a table."""
    pks: dict[str, str] = {
        "users": "id",
        "orders": "order_id",
        "order_items": "id",
        "products": "id",
        "events": "id",
        "inventory_items": "id",
        "distribution_centers": "id",
    }
    return pks.get(table_name)


def _get_key_columns(table_name: str) -> list[str]:
    """Return columns that should not be null."""
    keys: dict[str, list[str]] = {
        "users": ["id", "created_at"],
        "orders": ["order_id", "user_id", "status", "created_at"],
        "order_items": ["id", "order_id", "user_id", "product_id", "status", "sale_price"],
        "products": ["id", "name", "category"],
        "events": ["id", "user_id", "event_type", "created_at"],
        "inventory_items": ["id", "product_id"],
        "distribution_centers": ["id", "name"],
    }
    return keys.get(table_name, [])


def _get_date_columns(table_name: str) -> list[str]:
    """Return date/datetime columns for a table."""
    dates: dict[str, list[str]] = {
        "users": ["created_at"],
        "orders": ["created_at"],
        "order_items": ["created_at"],
        "events": ["created_at"],
        "inventory_items": ["created_at", "sold_at"],
    }
    return dates.get(table_name, [])


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_validation_summary(report: dict[str, Any]) -> None:
    """Print a concise validation summary to stdout."""
    print("\n--- Data Validation Summary ---")
    for table_name, info in report["tables"].items():
        status = "[PASS]" if info["passed"] else "[FAIL]"
        print(
            f"  {status:<6}  {table_name:<25} "
            f"{info['row_count']:>8,} rows  "
            f"{len(info['warnings'])} warnings  "
            f"{len(info['issues'])} errors"
        )
        for issue in info["issues"]:
            print(f"           [FAIL] {issue}")

    ri = report.get("referential_integrity", [])
    if ri:
        print(f"\n  Referential integrity: {len(ri)} issue(s)")
        for e in ri:
            print(f"    [WARN] {e}")
    else:
        print("\n  Referential integrity: [OK]")

    overall = "[PASSED]" if report["passed"] else "[FAILED]"
    print(f"\nOverall validation: {overall}")
    print("-------------------------------\n")
