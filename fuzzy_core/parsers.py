"""Read/write tabular files and build the composite match keys."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Union

import pandas as pd

SUPPORTED_INPUT_SUFFIXES = {".csv", ".xls", ".xlsx", ".dta"}
SUPPORTED_OUTPUT_SUFFIXES = {".csv", ".xlsx", ".dta"}

MIME_TYPES = {
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "dta": "application/octet-stream",
}

PathOrBuffer = Union[str, Path, BinaryIO]


def _suffix_of(source: Any) -> str:
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    name = getattr(source, "name", None)
    if not name:
        raise ValueError("File-like object must have a .name so the format can be detected")
    return Path(str(name)).suffix.lower()


def read_table(source: PathOrBuffer) -> pd.DataFrame:
    """Load a CSV, Excel, or Stata file from a path or a file-like object with ``.name``."""
    suffix = _suffix_of(source)
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(source)
    return pd.read_stata(source)


def write_table(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save *df* using the extension on *path* (``.csv``, ``.xlsx``, or ``.dta``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".xlsx":
        df.to_excel(path, index=False)
    elif suffix == ".dta":
        df.to_stata(path, write_index=False)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def dataframe_to_bytes(df: pd.DataFrame, fmt: str) -> tuple[bytes, str]:
    """Serialize *df* for a download button. Returns ``(payload, mime_type)``."""
    fmt = fmt.lower().lstrip(".")
    if fmt not in MIME_TYPES:
        raise ValueError(f"Unsupported output format: {fmt}")
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), MIME_TYPES[fmt]
    buffer = BytesIO()
    if fmt == "xlsx":
        df.to_excel(buffer, index=False)
    else:
        df.to_stata(buffer, write_index=False)
    return buffer.getvalue(), MIME_TYPES[fmt]


def require_columns(df: pd.DataFrame, keys: list[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError("Missing key column(s): " + ", ".join(missing))
    if not keys:
        raise ValueError("At least one key column is required")


def shared_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    return sorted(set(left.columns) & set(right.columns))


def build_key_series(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Concatenate key columns into a single lowercased, whitespace-normalized string."""
    require_columns(df, keys)
    return (
        df.loc[:, keys]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )
