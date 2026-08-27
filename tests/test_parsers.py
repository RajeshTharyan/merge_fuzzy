from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest

from fuzzy_core.parsers import (
    build_key_series,
    dataframe_to_bytes,
    read_table,
    require_columns,
    shared_columns,
    write_table,
)


def test_require_columns_missing():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(KeyError, match="Missing key column"):
        require_columns(df, ["a", "b"])


def test_require_columns_empty_list():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="At least one key"):
        require_columns(df, [])


def test_shared_columns_sorted_intersection():
    left = pd.DataFrame({"b": [1], "a": [2], "x": [3]})
    right = pd.DataFrame({"a": [1], "b": [2], "z": [3]})
    assert shared_columns(left, right) == ["a", "b"]


def test_build_key_series_normalizes_whitespace_and_case():
    df = pd.DataFrame(
        {
            "first": ["  Jane  ", "JOHN"],
            "last": ["  DOE ", "smith"],
            "id": [10, 20],
        }
    )
    keys = build_key_series(df, ["first", "last"])
    assert list(keys) == ["jane doe", "john smith"]
    # original dtypes / values are left alone
    assert df["id"].tolist() == [10, 20]
    assert df["first"].tolist() == ["  Jane  ", "JOHN"]
    assert pd.api.types.is_integer_dtype(df["id"])


def test_build_key_series_coerces_non_string_without_mutating():
    df = pd.DataFrame({"code": [1001, 1002], "name": ["Acme", "Beta"]})
    keys = build_key_series(df, ["code", "name"])
    assert list(keys) == ["1001 acme", "1002 beta"]
    assert pd.api.types.is_integer_dtype(df["code"])


def test_read_table_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_table(path)


def test_read_table_filelike_requires_name():
    buf = BytesIO(b"a,b\n1,2\n")
    with pytest.raises(ValueError, match="must have a .name"):
        read_table(buf)


def test_csv_roundtrip_path_and_filelike(tmp_path: Path):
    original = pd.DataFrame({"name": ["Acme Corp", "Beta LLC"], "id": [1, 2]})
    path = tmp_path / "firms.csv"
    write_table(original, path)
    from_path = read_table(path)
    pd.testing.assert_frame_equal(from_path, original)

    raw = path.read_bytes()
    buf = BytesIO(raw)
    buf.name = "firms.csv"
    from_buf = read_table(buf)
    pd.testing.assert_frame_equal(from_buf, original)


def test_excel_and_stata_roundtrip(tmp_path: Path):
    original = pd.DataFrame({"name": ["Acme Corp", "Beta LLC"], "id": [1, 2]})
    xlsx = tmp_path / "firms.xlsx"
    dta = tmp_path / "firms.dta"
    write_table(original, xlsx)
    write_table(original, dta)
    pd.testing.assert_frame_equal(read_table(xlsx), original)
    loaded_dta = read_table(dta)
    assert list(loaded_dta.columns) == ["name", "id"]
    assert loaded_dta["name"].tolist() == ["Acme Corp", "Beta LLC"]
    assert loaded_dta["id"].tolist() == [1, 2]


def test_dataframe_to_bytes_csv_and_unknown():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    payload, mime = dataframe_to_bytes(df, "csv")
    assert mime == "text/csv"
    assert b"a,b" in payload
    with pytest.raises(ValueError, match="Unsupported output format"):
        dataframe_to_bytes(df, "parquet")


def test_write_table_rejects_unknown(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported output format"):
        write_table(pd.DataFrame({"a": [1]}), tmp_path / "out.parquet")
