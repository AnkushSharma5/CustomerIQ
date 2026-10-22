from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import pandas as pd

from src.artifacts import StartupRecovery, has_complete_raw_inputs, validate_output_artifacts
from src.config import OUTPUT_DIR, OUTPUT_FILES, PROJECT_ROOT, REQUIRED_TABLES


def test_output_directory_is_derived_from_repository_location():
    assert OUTPUT_DIR == PROJECT_ROOT / "data" / "output"
    assert OUTPUT_DIR.is_absolute()


def test_validate_output_artifacts_reports_ready_missing_unreadable_and_empty(tmp_path: Path):
    pd.DataFrame({"value": [1]}).to_parquet(tmp_path / "ready.parquet", index=False)
    (tmp_path / "corrupt.parquet").write_bytes(b"not a parquet file")
    pd.DataFrame({"value": []}).to_parquet(tmp_path / "empty.parquet", index=False)

    statuses = validate_output_artifacts(
        tmp_path,
        {
            "ready": "ready.parquet",
            "missing": "missing.parquet",
            "corrupt": "corrupt.parquet",
            "empty": "empty.parquet",
        },
    )

    assert statuses["ready"] == {"status": "ready", "rows": 1}
    assert statuses["missing"]["status"] == "missing"
    assert statuses["corrupt"]["status"] == "unreadable"
    assert statuses["empty"] == {"status": "empty", "rows": 0}


def test_has_complete_raw_inputs_accepts_csv_or_parquet(tmp_path: Path):
    for table in REQUIRED_TABLES[:-1]:
        (tmp_path / f"{table}.csv").touch()
    assert not has_complete_raw_inputs(tmp_path)

    (tmp_path / f"{REQUIRED_TABLES[-1]}.parquet").touch()
    assert has_complete_raw_inputs(tmp_path)


def test_startup_recovery_runs_pipeline_once_when_inputs_are_available(tmp_path: Path):
    recovery = StartupRecovery()
    calls = []
    output_files = {"customer_rfm": "customer_rfm.parquet"}

    def pipeline_runner():
        calls.append("run")
        sleep(0.01)
        pd.DataFrame({"customer_id": [1]}).to_parquet(
            tmp_path / "customer_rfm.parquet", index=False
        )
        return True

    def ensure_outputs():
        return recovery.ensure_outputs(
            output_dir=tmp_path,
            output_files=output_files,
            raw_inputs_available=True,
            pipeline_runner=pipeline_runner,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: ensure_outputs(), range(4)))

    assert all(result["customer_rfm"]["status"] == "ready" for result in results)
    assert calls == ["run"]


def test_startup_recovery_does_not_run_without_complete_inputs(tmp_path: Path):
    recovery = StartupRecovery()
    calls = []
    output_files = {"customer_rfm": "customer_rfm.parquet"}

    statuses = recovery.ensure_outputs(
        output_dir=tmp_path,
        output_files=output_files,
        raw_inputs_available=False,
        pipeline_runner=lambda: calls.append("run") or True,
    )

    assert statuses["customer_rfm"]["status"] == "missing"
    assert calls == []


def test_repository_outputs_are_all_readable_and_nonempty():
    statuses = validate_output_artifacts()

    assert set(statuses) == set(OUTPUT_FILES)
    assert len(statuses) == 9
    assert all(item["status"] == "ready" for item in statuses.values())
