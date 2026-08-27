"""Streamlit front end. Matching and file I/O live in ``fuzzy_core``."""

from __future__ import annotations

import zipfile

import pandas as pd
import streamlit as st

from fuzzy_core import (
    DEFAULT_NAME_METRICS,
    dataframe_to_bytes,
    match_compare,
    read_table,
    shared_columns,
)

READ_ERRORS = (
    ValueError,
    KeyError,
    OSError,
    UnicodeError,
    zipfile.BadZipFile,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)

st.set_page_config(page_title="Fuzzy Matcher", layout="centered")
st.title("Fuzzy Dataset Matcher")
st.markdown("By: **Prof. Rajesh Tharyan**")

st.markdown(
    """
**What does this app do?**

This app allows you to perform fuzzy matching between two datasets using multiple algorithms. The algorithms used are RapidFuzz, TextDistance, RecordLinkage, and NameMatcher.
You can upload a "MASTER" file and a "USING" file, select the key columns to match on, and compare results from different fuzzy matching methods.

**How to use:**
1. Upload your MASTER and USING files in the sidebar (supported formats: CSV, Excel, Stata).
2. Select the key columns that exist in both datasets for matching.
3. (Optional) Choose distance metrics for the NameMatcher algorithm.
4. Click "Run Fuzzy Match" to see the results.
5. Download the matched results in your preferred format (CSV, Excel, Stata).
"""
)

with st.sidebar:
    st.header("Upload Files")
    master_file = st.file_uploader(
        "Upload MASTER file", type=["csv", "xlsx", "xls", "dta"]
    )
    using_file = st.file_uploader(
        "Upload USING file", type=["csv", "xlsx", "xls", "dta"]
    )
    selected_distance_metrics = st.multiselect(
        "Select distance metrics for NameMatcher (default: all)",
        DEFAULT_NAME_METRICS,
        default=DEFAULT_NAME_METRICS,
    )

if master_file and using_file:
    file_sig = (master_file.name, master_file.size, using_file.name, using_file.size)
    if st.session_state.get("file_sig") != file_sig:
        st.session_state.pop("matched", None)
        st.session_state["file_sig"] = file_sig

    try:
        master_df = read_table(master_file)
        using_df = read_table(using_file)
    except READ_ERRORS as exc:
        st.error(f"Error: {exc}")
    else:
        selected_keys = st.multiselect(
            "Select key variable(s) for matching",
            shared_columns(master_df, using_df),
        )

        if not selected_keys:
            st.info("Please select one or more key variables.")
        else:
            if st.button("Run Fuzzy Match"):
                try:
                    matched = match_compare(
                        master_df,
                        using_df,
                        selected_keys,
                        selected_distance_metrics,
                    )
                except READ_ERRORS as exc:
                    st.error(f"Error: {exc}")
                else:
                    st.session_state["matched"] = matched
                    st.success("Fuzzy matching complete.")

            if "matched" in st.session_state:
                matched = st.session_state["matched"]
                st.dataframe(matched.head(50))
                file_format = st.selectbox(
                    "Choose format to download", ["csv", "xlsx", "dta"]
                )
                payload, mime = dataframe_to_bytes(matched, file_format)
                st.download_button(
                    "Download Results",
                    data=payload,
                    file_name=f"fuzzy_matched.{file_format}",
                    mime=mime,
                )
else:
    st.info("Upload both MASTER and USING files to begin.")
