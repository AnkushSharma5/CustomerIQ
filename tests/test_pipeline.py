import pandas as pd

import run_pipeline


def test_failed_validation_stops_before_mutating_duckdb_or_outputs(monkeypatch):
    monkeypatch.setattr(
        run_pipeline,
        "load_all_tables",
        lambda: {"users": pd.DataFrame({"id": [1]})},
    )

    def unexpected_database_write(*args, **kwargs):
        raise AssertionError("DuckDB must not be written after failed validation.")

    def unexpected_output_publish(*args, **kwargs):
        raise AssertionError("Artifacts must not be published after failed validation.")

    monkeypatch.setattr(run_pipeline, "load_tables_to_duckdb", unexpected_database_write)
    monkeypatch.setattr(run_pipeline, "publish_output_artifacts", unexpected_output_publish)

    assert run_pipeline.run_pipeline() is False
