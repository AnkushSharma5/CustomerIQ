import pandas as pd
import pytest

from src.duckdb_manager import get_table_row_count, load_tables_to_duckdb, run_query, table_exists


def test_failed_table_load_rolls_back_and_closes_connection(tmp_path):
    db_path = tmp_path / "customeriq.duckdb"
    load_tables_to_duckdb({"existing": pd.DataFrame({"value": [1]})}, db_path)

    with pytest.raises(TypeError):
        load_tables_to_duckdb(
            {
                "existing": pd.DataFrame({"value": [2]}),
                "invalid": None,
            },
            db_path,
        )

    assert get_table_row_count("existing", db_path) == 1
    assert run_query('SELECT value FROM "existing"', db_path)["value"].tolist() == [1]


def test_table_names_are_quoted_before_sql_interpolation(tmp_path):
    db_path = tmp_path / "customeriq.duckdb"
    load_tables_to_duckdb({"existing": pd.DataFrame({"value": [1]})}, db_path)
    malicious_name = 'users; DROP TABLE existing; --'

    load_tables_to_duckdb({malicious_name: pd.DataFrame({"value": [2]})}, db_path)

    assert table_exists("existing", db_path)
    assert get_table_row_count("existing", db_path) == 1
    assert table_exists(malicious_name, db_path)
    assert get_table_row_count(malicious_name, db_path) == 1
