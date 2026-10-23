"""
CustomerIQ — Regenerate monthly KPI and category Parquet files.

Regenerates the two SQL-backed analytics files from DuckDB. Run the full
pipeline to publish a consistent set of all nine dashboard artifacts.

Usage:
    python scripts/generate_outputs.py

Requires:
    - database/customeriq.duckdb (populated by run_pipeline.py)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUT_DIR, OUTPUT_FILES
from src.duckdb_manager import run_sql_file
from src.utils import save_parquet


def main():
    print("Regenerating monthly KPI and category analytics from DuckDB...\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    # Monthly KPIs
    print("Generating monthly KPIs...")
    try:
        kpis = run_sql_file("monthly_kpis.sql")
        save_parquet(kpis, OUTPUT_DIR / OUTPUT_FILES["monthly_kpis"], "monthly_kpis")
    except Exception as e:
        print(f"  ⚠ monthly_kpis failed: {e}")
        failures.append("monthly_kpis")

    # Category analytics
    print("Generating category analytics...")
    try:
        cat = run_sql_file("category_analytics.sql")
        save_parquet(cat, OUTPUT_DIR / OUTPUT_FILES["category_analytics"], "category_analytics")
    except Exception as e:
        print(f"  ⚠ category_analytics failed: {e}")
        failures.append("category_analytics")

    if failures:
        raise SystemExit(f"Failed to regenerate: {', '.join(failures)}")
    print(
        "\nDone. This helper only updates these two files; run `python run_pipeline.py` "
        "to rebuild a consistent nine-artifact set."
    )


if __name__ == "__main__":
    main()
