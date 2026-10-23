"""End-to-end pipeline test using an isolated temporary project copy."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import validate_output_artifacts
from src.config import OUTPUT_FILES
from tests.test_churn import make_synthetic_churn_data
from tests.test_data_validation import make_products, make_events, make_inventory, make_dc


def _tree_state(path: Path) -> tuple | None:
    """Capture file names, sizes, and timestamps to prove the test is isolated."""
    if not path.exists():
        return None
    return tuple(sorted(
        (item.relative_to(path).as_posix(), item.stat().st_size, item.stat().st_mtime_ns)
        for item in path.rglob("*")
        if item.is_file()
    ))


def generate_synthetic_raw_dataset(raw_dir: Path) -> None:
    """Write labelled synthetic fixtures to the caller-provided directory."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    users, orders, order_items = make_synthetic_churn_data(n_customers=250, seed=42)
    assert users["email"].str.endswith("@example.com").all()
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


def test_pipeline():
    """Run the complete pipeline without touching this checkout's data or DB."""
    protected_paths = [
        PROJECT_ROOT / "data" / "raw",
        PROJECT_ROOT / "data" / "output",
        PROJECT_ROOT / "database",
        PROJECT_ROOT / "models",
    ]
    initial_state = {path: _tree_state(path) for path in protected_paths}

    with tempfile.TemporaryDirectory(prefix="customeriq-e2e-") as temp:
        project_copy = Path(temp) / "project"
        project_copy.mkdir()
        shutil.copytree(PROJECT_ROOT / "src", project_copy / "src")
        shutil.copytree(PROJECT_ROOT / "sql", project_copy / "sql")
        shutil.copy2(PROJECT_ROOT / "run_pipeline.py", project_copy / "run_pipeline.py")
        generate_synthetic_raw_dataset(project_copy / "data" / "raw")

        result = subprocess.run(
            [sys.executable, "run_pipeline.py"],
            cwd=project_copy,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, (
            "Pipeline failed in its isolated fixture project.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        output_dir = project_copy / "data" / "output"
        statuses = validate_output_artifacts(output_dir)
        assert set(statuses) == set(OUTPUT_FILES)
        assert all(item["status"] == "ready" for item in statuses.values()), statuses
        assert len(list(output_dir.glob("*.parquet"))) == len(OUTPUT_FILES)

        for filename in OUTPUT_FILES.values():
            frame = pd.read_parquet(output_dir / filename)
            assert not frame.empty, f"Output file is empty: {filename}"

        assert (project_copy / "database" / "customeriq.duckdb").is_file()

    assert {path: _tree_state(path) for path in protected_paths} == initial_state
