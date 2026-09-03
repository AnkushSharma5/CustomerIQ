"""
CustomerIQ — Data Loader.

Responsibilities:
- Auto-detect CSV or Parquet format for each required table.
- Load all 7 required tables into a dict of DataFrames.
- Produce clear, actionable error messages when data is missing.
- Never fabricate or silently substitute missing data.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import RAW_DIR, REQUIRED_TABLES

logger = logging.getLogger("customeriq.data_loader")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_all_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Load all required theLook eCommerce tables from *raw_dir*.

    Tries Parquet first, then CSV for each table.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are table names (e.g. "users", "orders").

    Raises
    ------
    SystemExit
        If any required table cannot be found, prints a clear diagnostic
        and exits with code 1 rather than an ugly traceback.
    """
    _check_raw_dir(raw_dir)

    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for table in REQUIRED_TABLES:
        df = _load_single_table(table, raw_dir)
        if df is None:
            missing.append(table)
        else:
            tables[table] = df

    if missing:
        _report_missing(missing, raw_dir)

    logger.info("All %d required tables loaded successfully.", len(tables))
    return tables


def load_table(table_name: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    Load a single table.  Raises FileNotFoundError if not found.
    """
    df = _load_single_table(table_name, raw_dir)
    if df is None:
        raise FileNotFoundError(
            f"Table '{table_name}' not found in {raw_dir}. "
            "Expected '{table_name}.parquet' or '{table_name}.csv'."
        )
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_raw_dir(raw_dir: Path) -> None:
    """Ensure the raw data directory exists."""
    if not raw_dir.exists():
        print("\n" + "=" * 60)
        print("  CustomerIQ Dataset Error")
        print("=" * 60)
        print(f"\nThe raw data directory does not exist:\n  {raw_dir}\n")
        print("Please create the directory and place the theLook eCommerce")
        print("dataset files inside it before running the pipeline.\n")
        print("See: data/raw/README.md for full instructions.\n")
        raise SystemExit(1)


def _load_single_table(table_name: str, raw_dir: Path) -> Optional[pd.DataFrame]:
    """
    Try to load *table_name* from *raw_dir*.

    Tries Parquet first (faster), then CSV.

    Returns None if neither file exists.
    """
    parquet_path = raw_dir / f"{table_name}.parquet"
    csv_path = raw_dir / f"{table_name}.csv"

    if parquet_path.exists():
        logger.info("Loading '%s' from Parquet: %s", table_name, parquet_path.name)
        df = pd.read_parquet(parquet_path)
        logger.info("  → %d rows, %d columns", len(df), len(df.columns))
        return df

    if csv_path.exists():
        logger.info("Loading '%s' from CSV: %s", table_name, csv_path.name)
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info("  → %d rows, %d columns", len(df), len(df.columns))
        return df

    return None


def _report_missing(missing: list[str], raw_dir: Path) -> None:
    """Print a clear error message for missing tables and exit."""
    print("\n" + "=" * 60)
    print("  CustomerIQ Dataset Error")
    print("=" * 60)
    print("\nThe required theLook eCommerce dataset was not found.\n")
    print(f"Expected location:\n  {raw_dir}\n")
    print("Required tables (missing):")
    for table in missing:
        print(f"  ✗ {table}.parquet  OR  {table}.csv")
    print("\nAll required tables:")
    for table in REQUIRED_TABLES:
        print(f"  - {table}")
    print(
        "\nPlease place the dataset files in data/raw/ and run the pipeline again."
        "\nSee data/raw/README.md for detailed instructions.\n"
    )
    raise SystemExit(1)


def detect_format(table_name: str, raw_dir: Path = RAW_DIR) -> Optional[str]:
    """
    Return 'parquet', 'csv', or None depending on what file exists.
    """
    if (raw_dir / f"{table_name}.parquet").exists():
        return "parquet"
    if (raw_dir / f"{table_name}.csv").exists():
        return "csv"
    return None
