"""
Temporary internal test fixture runner.
Verifies end-to-end pipeline execution with synthetic data.
"""
import sys
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_churn import make_synthetic_churn_data
from tests.test_data_validation import make_products, make_events, make_inventory, make_dc


def generate_synthetic_raw_dataset(raw_dir: Path):
    """Generate temporary synthetic parquet files in raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    users, orders, order_items = make_synthetic_churn_data(n_customers=250, seed=42)
    products = make_products(n=30)
    events = make_events(n=100)
    inventory = make_inventory(n=30)
    dc = make_dc(n=5)

    users.to_parquet(raw_dir / "users.parquet", index=False)
    orders.to_parquet(raw_dir / "orders.parquet", index=False)
    order_items.to_parquet(raw_dir / "order_items.parquet", index=False)
    products.to_parquet(raw_dir / "products.parquet", index=False)
    events.to_parquet(raw_dir / "events.parquet", index=False)
    inventory.to_parquet(raw_dir / "inventory_items.parquet", index=False)
    dc.to_parquet(raw_dir / "distribution_centers.parquet", index=False)
    print(f"Generated synthetic raw datasets in {raw_dir}")


def test_pipeline():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    backup_files = {}

    # Backup any existing real raw files if present
    for p in raw_dir.glob("*.parquet"):
        backup_files[p.name] = p.read_bytes()

    try:
        generate_synthetic_raw_dataset(raw_dir)

        # Run pipeline
        from run_pipeline import run_pipeline
        success = run_pipeline()
        assert success is True, "Pipeline failed execution"

        # Verify DuckDB database created
        db_path = PROJECT_ROOT / "database" / "customeriq.duckdb"
        assert db_path.exists(), "DuckDB database file not created"

        # Verify all output parquet files exist and are non-empty
        output_dir = PROJECT_ROOT / "data" / "output"
        from src.config import OUTPUT_FILES
        for key, fname in OUTPUT_FILES.items():
            out_file = output_dir / fname
            assert out_file.exists(), f"Output file missing: {fname}"
            df = pd.read_parquet(out_file)
            assert len(df) > 0, f"Output file empty: {fname}"
            print(f"  ✓ {fname}: {len(df)} rows")

        # Test importing dashboard app
        import dashboard.app
        print("  ✓ Dashboard app imported successfully")

        print("\nPipeline self-test PASSED completely!")

    finally:
        # Cleanup synthetic test parquet files from data/raw
        for name in ["users", "orders", "order_items", "products", "events", "inventory_items", "distribution_centers"]:
            pf = raw_dir / f"{name}.parquet"
            if pf.exists():
                pf.unlink()

        # Restore backups if any existed
        for fname, data in backup_files.items():
            (raw_dir / fname).write_bytes(data)


if __name__ == "__main__":
    test_pipeline()
