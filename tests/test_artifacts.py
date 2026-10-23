from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import pandas as pd

from src.artifacts import (
    StartupRecovery,
    has_complete_raw_inputs,
    publish_output_artifacts,
    validate_output_artifacts,
)
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


def test_validate_output_artifacts_rejects_readable_wrong_schema(tmp_path: Path):
    pd.DataFrame({"customer_id": [1]}).to_parquet(tmp_path / "rfm.parquet", index=False)

    status = validate_output_artifacts(
        tmp_path,
        {"customer_rfm": "rfm.parquet"},
    )

    assert status["customer_rfm"]["status"] == "schema_mismatch"


def test_validate_output_artifacts_rejects_non_numeric_metric_columns(tmp_path: Path):
    pd.DataFrame({
        "year_month": ["2024-01"],
        "total_revenue": ["100.00"],
        "total_orders": ["1"],
        "unique_customers": ["1"],
        "avg_order_value": ["100.00"],
        "new_customers": ["1"],
        "repeat_customers": ["0"],
        "repeat_customer_rate": ["0.0"],
    }).to_parquet(tmp_path / "kpis.parquet", index=False)

    status = validate_output_artifacts(tmp_path, {"monthly_kpis": "kpis.parquet"})

    assert status["monthly_kpis"]["status"] == "schema_mismatch"


def test_publish_output_artifacts_keeps_previous_files_when_staging_fails(tmp_path: Path):
    outputs = {"first": pd.DataFrame({"value": [1]}), "second": pd.DataFrame()}
    filenames = {"first": "first.parquet", "second": "second.parquet"}
    existing_path = tmp_path / "first.parquet"
    pd.DataFrame({"value": [99]}).to_parquet(existing_path, index=False)
    original = existing_path.read_bytes()

    try:
        publish_output_artifacts(outputs, tmp_path, filenames)
    except ValueError as exc:
        assert "empty output artifact" in str(exc)
    else:
        raise AssertionError("Publishing an empty artifact should fail.")

    assert existing_path.read_bytes() == original
    assert not (tmp_path / "second.parquet").exists()


def test_publish_output_artifacts_rolls_back_replacements_on_failure(tmp_path: Path, monkeypatch):
    import src.artifacts as artifacts

    filenames = {"first": "first.parquet", "second": "second.parquet"}
    outputs = {
        "first": pd.DataFrame({"value": [1]}),
        "second": pd.DataFrame({"value": [2]}),
    }
    original = {}
    for filename in filenames.values():
        path = tmp_path / filename
        pd.DataFrame({"value": [99]}).to_parquet(path, index=False)
        original[filename] = path.read_bytes()

    real_replace = artifacts.os.replace
    raised = False

    def fail_second_staged_replace(source, destination):
        nonlocal raised
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not raised
            and source_path.parent.name == "staged"
            and destination_path.name == "second.parquet"
        ):
            raised = True
            raise OSError("simulated replacement failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_second_staged_replace)
    try:
        publish_output_artifacts(outputs, tmp_path, filenames)
    except OSError as exc:
        assert "simulated" in str(exc)
    else:
        raise AssertionError("The simulated file replacement should fail.")

    assert all((tmp_path / name).read_bytes() == content for name, content in original.items())


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
        pd.DataFrame({
            "customer_id": [1],
            "last_purchase_date": ["2024-01-01"],
            "recency": [0],
            "frequency": [1],
            "monetary": [10.0],
            "r_score": [5],
            "f_score": [1],
            "m_score": [3],
            "rfm_score": [9],
            "segment": ["New Customers"],
        }).to_parquet(
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
