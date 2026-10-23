"""Validation and one-time recovery for dashboard output artifacts."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping

import pandas as pd

from src.config import OUTPUT_DIR, OUTPUT_FILES, RAW_DIR, REQUIRED_TABLES

logger = logging.getLogger("customeriq.artifacts")

ArtifactResult = dict[str, str | int | None]

OUTPUT_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "customer_rfm": (
        "customer_id", "last_purchase_date", "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score", "segment",
    ),
    "customer_segments": (
        "customer_id", "total_orders", "first_order_date", "last_order_date",
        "total_revenue", "avg_order_value", "n_unique_products", "tenure_days",
        "annual_frequency", "estimated_clv", "segment", "r_score", "f_score",
        "m_score", "rfm_score", "recency", "frequency", "monetary",
    ),
    "churn_predictions": ("customer_id", "churn_probability", "churn_prediction", "risk_level"),
    "churn_feature_importance": ("feature", "importance"),
    "monthly_kpis": (
        "year_month", "total_revenue", "total_orders", "unique_customers",
        "avg_order_value", "new_customers", "repeat_customers", "repeat_customer_rate",
    ),
    "cohort_retention": ("cohort_month",),
    "category_analytics": (
        "category", "total_revenue", "order_count", "unique_customers",
        "avg_selling_price", "product_count",
    ),
    "recommendations": (
        "customer_id", "product_id", "product_name", "category", "brand",
        "affinity_score", "explanation", "rank",
    ),
    "clv": ("customer_id", "estimated_clv", "annual_frequency"),
}

OUTPUT_NUMERIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "customer_rfm": ("recency", "frequency", "monetary", "r_score", "f_score", "m_score", "rfm_score"),
    "customer_segments": (
        "total_orders", "total_revenue", "avg_order_value", "n_unique_products",
        "tenure_days", "annual_frequency", "estimated_clv", "recency", "frequency",
        "monetary", "r_score", "f_score", "m_score", "rfm_score",
    ),
    "churn_predictions": ("churn_probability", "churn_prediction"),
    "churn_feature_importance": ("importance",),
    "monthly_kpis": (
        "total_revenue", "total_orders", "unique_customers", "avg_order_value",
        "new_customers", "repeat_customers", "repeat_customer_rate",
    ),
    "category_analytics": (
        "total_revenue", "order_count", "unique_customers", "avg_selling_price", "product_count",
    ),
    "recommendations": ("product_id", "affinity_score", "rank"),
    "clv": ("estimated_clv", "annual_frequency"),
}


def validate_output_artifacts(
    output_dir: Path = OUTPUT_DIR,
    output_files: dict[str, str] = OUTPUT_FILES,
    required_columns: Mapping[str, tuple[str, ...]] = OUTPUT_REQUIRED_COLUMNS,
) -> dict[str, ArtifactResult]:
    """Check each configured Parquet artifact's presence, schema, and rows.

    Only the status and row count are returned for display. Detailed I/O errors
    are written to the application log and never need to be shown to visitors.
    """
    results: dict[str, ArtifactResult] = {}

    for key, filename in output_files.items():
        path = output_dir / filename
        if not path.is_file():
            results[key] = {"status": "missing", "rows": None}
            logger.warning("Required output artifact is missing: %s", path)
            continue

        try:
            frame = pd.read_parquet(path)
        except FileNotFoundError:
            results[key] = {"status": "missing", "rows": None}
            logger.warning("Required output artifact disappeared during validation: %s", path)
        except Exception:
            results[key] = {"status": "unreadable", "rows": None}
            logger.exception("Could not read output artifact %s", path)
        else:
            if frame.empty:
                results[key] = {"status": "empty", "rows": 0}
                logger.warning("Output artifact contains no rows: %s", path)
            else:
                missing = sorted(set(required_columns.get(key, ())) - set(frame.columns))
                non_numeric = sorted(
                    column
                    for column in OUTPUT_NUMERIC_COLUMNS.get(key, ())
                    if column in frame.columns
                    and not pd.api.types.is_numeric_dtype(frame[column].dtype)
                )
                missing.extend(f"numeric dtype for {column}" for column in non_numeric)
                if key == "cohort_retention" and not any(
                    str(column).isdigit() for column in frame.columns
                ):
                    missing.append("at least one numeric retention-month column")
                elif key == "cohort_retention":
                    period_columns = [column for column in frame.columns if str(column).isdigit()]
                    non_numeric_periods = [
                        column for column in period_columns
                        if not pd.api.types.is_numeric_dtype(frame[column].dtype)
                    ]
                    missing.extend(
                        f"numeric dtype for retention month {column}"
                        for column in non_numeric_periods
                    )
                if missing:
                    results[key] = {"status": "schema_mismatch", "rows": len(frame)}
                    logger.error(
                        "Output artifact %s is missing required columns: %s",
                        path,
                        ", ".join(missing),
                    )
                else:
                    results[key] = {"status": "ready", "rows": len(frame)}

    return results


def publish_output_artifacts(
    outputs: Mapping[str, pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    output_files: dict[str, str] = OUTPUT_FILES,
) -> None:
    """Stage, validate, and replace a complete artifact set.

    The prior files are retained until every new Parquet file has been written
    and validated. If a replacement fails, already-replaced files are restored.
    """
    if set(outputs) != set(output_files):
        missing = sorted(set(output_files) - set(outputs))
        unexpected = sorted(set(outputs) - set(output_files))
        raise ValueError(
            f"Output set does not match the configured contract; missing={missing}, "
            f"unexpected={unexpected}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".customeriq-stage-", dir=output_dir) as temp:
        stage_dir = Path(temp) / "staged"
        backup_dir = Path(temp) / "backup"
        stage_dir.mkdir()
        backup_dir.mkdir()

        for key, filename in output_files.items():
            frame = outputs[key]
            if frame.empty:
                raise ValueError(f"Refusing to publish empty output artifact: {filename}")
            frame.to_parquet(stage_dir / filename, index=False)

        statuses = validate_output_artifacts(stage_dir, output_files)
        invalid = {key: status for key, status in statuses.items() if status["status"] != "ready"}
        if invalid:
            raise ValueError(f"Staged output artifacts failed validation: {invalid}")

        existing = set()
        for filename in output_files.values():
            final_path = output_dir / filename
            if final_path.exists():
                shutil.copy2(final_path, backup_dir / filename)
                existing.add(filename)

        replaced: list[str] = []
        try:
            for filename in output_files.values():
                os.replace(stage_dir / filename, output_dir / filename)
                replaced.append(filename)
        except Exception:
            for filename in reversed(replaced):
                backup_path = backup_dir / filename
                final_path = output_dir / filename
                if filename in existing:
                    os.replace(backup_path, final_path)
                else:
                    final_path.unlink(missing_ok=True)
            raise


def has_complete_raw_inputs(
    raw_dir: Path = RAW_DIR,
    required_tables: list[str] = REQUIRED_TABLES,
) -> bool:
    """Return whether every required raw table has a Parquet or CSV file."""
    return all(
        (raw_dir / f"{table}.parquet").is_file()
        or (raw_dir / f"{table}.csv").is_file()
        for table in required_tables
    )


class StartupRecovery:
    """Validate outputs and attempt the pipeline at most once per app process."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._attempted = False

    def ensure_outputs(
        self,
        *,
        output_dir: Path = OUTPUT_DIR,
        output_files: dict[str, str] = OUTPUT_FILES,
        raw_inputs_available: bool = False,
        pipeline_runner: Callable[[], bool] | None = None,
    ) -> dict[str, ArtifactResult]:
        """Validate output files, then regenerate once if inputs are available."""
        with self._lock:
            statuses = validate_output_artifacts(output_dir, output_files)
            if all(item["status"] == "ready" for item in statuses.values()):
                return statuses

            if self._attempted:
                return statuses
            self._attempted = True

            if not raw_inputs_available:
                logger.error(
                    "Dashboard outputs are missing or invalid and the complete raw dataset "
                    "is unavailable. Restore the published Parquet artifacts or provide "
                    "approved inputs and regenerate them before deployment."
                )
                return statuses

            if pipeline_runner is None:
                from run_pipeline import run_pipeline

                pipeline_runner = run_pipeline

            try:
                succeeded = pipeline_runner()
            except (Exception, SystemExit):
                logger.exception("Automatic dashboard output recovery failed.")
            else:
                if not succeeded:
                    logger.error("Automatic dashboard output recovery reported failure.")

            return validate_output_artifacts(output_dir, output_files)
