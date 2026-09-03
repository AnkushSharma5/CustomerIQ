"""
CustomerIQ — DuckDB Integration.

Responsibilities:
- Create and manage the DuckDB database at database/customeriq.duckdb.
- Load cleaned DataFrames into DuckDB tables.
- Provide a connection factory.
- Execute analytical SQL queries from the sql/ directory.
- Ensure idempotent operation (safe to re-run).
"""
import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DATABASE_PATH, SQL_DIR

logger = logging.getLogger("customeriq.duckdb_manager")


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection(db_path: Path = DATABASE_PATH) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection to the CustomerIQ database.

    The database file and its parent directory are created automatically.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    return conn


# ---------------------------------------------------------------------------
# Load tables into DuckDB
# ---------------------------------------------------------------------------

def load_tables_to_duckdb(
    tables: dict[str, pd.DataFrame],
    db_path: Path = DATABASE_PATH,
) -> None:
    """
    Load all cleaned DataFrames into DuckDB, replacing existing tables.

    This operation is idempotent — running the pipeline multiple times is safe.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Cleaned DataFrames keyed by table name.
    db_path : Path
        Path to the DuckDB database file.
    """
    conn = get_connection(db_path)

    for table_name, df in tables.items():
        logger.info("Loading '%s' into DuckDB (%d rows)...", table_name, len(df))
        # DROP TABLE IF EXISTS guarantees idempotent execution
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register("_temp_df", df)
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM _temp_df")
        conn.unregister("_temp_df")
        logger.info("  [OK] Table '%s' created.", table_name)

    conn.close()
    logger.info("All tables loaded into DuckDB: %s", db_path)


# ---------------------------------------------------------------------------
# Query execution helpers
# ---------------------------------------------------------------------------

def run_query(sql: str, db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """
    Execute a SQL query and return the result as a DataFrame.

    Parameters
    ----------
    sql : str
        SQL statement to execute.
    db_path : Path
        Path to the DuckDB database.

    Returns
    -------
    pd.DataFrame
        Query result.
    """
    conn = get_connection(db_path)
    try:
        df = conn.execute(sql).fetchdf()
    finally:
        conn.close()
    return df


def run_sql_file(filename: str, db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """
    Load and execute a SQL file from the sql/ directory.

    Parameters
    ----------
    filename : str
        SQL filename (e.g. 'rfm.sql').
    db_path : Path
        Path to the DuckDB database.

    Returns
    -------
    pd.DataFrame
        Query result.
    """
    sql_path = SQL_DIR / filename
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    sql = sql_path.read_text(encoding="utf-8")
    logger.info("Executing SQL file: %s", filename)
    return run_query(sql, db_path)


def table_exists(table_name: str, db_path: Path = DATABASE_PATH) -> bool:
    """Return True if *table_name* exists in the DuckDB database."""
    conn = get_connection(db_path)
    try:
        result = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0 if result else False
    finally:
        conn.close()


def get_table_row_count(table_name: str, db_path: Path = DATABASE_PATH) -> int:
    """Return the row count of a DuckDB table."""
    conn = get_connection(db_path)
    try:
        result = conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()
        return result[0] if result else 0
    finally:
        conn.close()
