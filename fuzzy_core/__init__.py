"""Importable matching and file-parsing helpers used by the Streamlit and Tkinter UIs."""

from fuzzy_core.matching import (
    DEFAULT_NAME_METRICS,
    best_match_name_matching,
    best_match_rapidfuzz,
    best_match_recordlinkage,
    best_match_textdistance,
    match_best,
    match_compare,
)
from fuzzy_core.parsers import (
    SUPPORTED_INPUT_SUFFIXES,
    SUPPORTED_OUTPUT_SUFFIXES,
    build_key_series,
    dataframe_to_bytes,
    read_table,
    require_columns,
    shared_columns,
    write_table,
)

__all__ = [
    "DEFAULT_NAME_METRICS",
    "SUPPORTED_INPUT_SUFFIXES",
    "SUPPORTED_OUTPUT_SUFFIXES",
    "best_match_name_matching",
    "best_match_rapidfuzz",
    "best_match_recordlinkage",
    "best_match_textdistance",
    "build_key_series",
    "dataframe_to_bytes",
    "match_best",
    "match_compare",
    "read_table",
    "require_columns",
    "shared_columns",
    "write_table",
]
