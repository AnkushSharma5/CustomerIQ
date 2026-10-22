"""Validation and one-time recovery for dashboard output artifacts."""
from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import Callable

import pandas as pd

from src.config import OUTPUT_DIR, OUTPUT_FILES, RAW_DIR, REQUIRED_TABLES

logger = logging.getLogger("customeriq.artifacts")

ArtifactResult = dict[str, str | int | None]


def validate_output_artifacts(
    output_dir: Path = OUTPUT_DIR,
    output_files: dict[str, str] = OUTPUT_FILES,
) -> dict[str, ArtifactResult]:
    """Check that each configured Parquet artifact exists and is readable.

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
                results[key] = {"status": "ready", "rows": len(frame)}

    return results


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
