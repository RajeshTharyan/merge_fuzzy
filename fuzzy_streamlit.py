import streamlit as st
import pandas as pd
from pathlib import Path
from typing import List
from rapidfuzz import fuzz, process
import textdistance
import recordlinkage as rl

st.set_page_config(page_title="Fuzzy Matcher", layout="centered")
st.title("Fuzzy Dataset Matcher")
st.markdown("By: **Prof. Rajesh Tharyan**")

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def _read_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix in (".xls", ".xlsx"):
        return pd.read_excel(uploaded_file)
    elif suffix == ".dta":
        return pd.read_stata(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def _validate_keys(df: pd.DataFrame, keys: List[str]) -> None:
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError("Missing key column(s): " + ", ".join(missing))
    for k in keys:
        if not pd.api.types.is_string_dtype(df[k]):
            df[k] = df[k].astype(str)

def _build_key_series(df: pd.DataFrame, keys: List[str]) -> pd.Series:
    return (
        df[keys]
        .fillna("")
        .apply(lambda r: " ".join(r.astype(str)), axis=1)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

# ─────────────────────────────────────────────────────────────────────────────
# Matching engines
# ─────────────────────────────────────────────────────────────────────────────
def _best_match_rapidfuzz(target: str, universe: pd.Series):
    match, score, idx = process.extractOne(
        target, universe, scorer=fuzz.token_sort_ratio
    )
    return universe.index[idx], score, "rapidfuzz"

def _best_match_textdistance(target: str, universe: pd.Series):
    sims = universe.map(
        lambda x: textdistance.jaro_winkler.normalized_similarity(target, x)
    )
    idx = sims.idxmax()
    return idx, sims.loc[idx] * 100, "textdistance"

def _best_match_recordlinkage(i: int, master_keys: pd.Series, using_keys: pd.Series):
    master_single = master_keys.iloc[[i]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")

    idxer = rl.index.Full()
    pairs = idxer.index(master_single, using_df)

    compare = rl.Compare()
    compare.string("key", "key", method="jaro", label="jw")
    scores_df = compare.compute(pairs, master_single, using_df)
    scores = scores_df["jw"]
    if scores.empty:
        return pd.NA, 0.0, "recordlinkage"
    best_pair = scores.idxmax()
    return best_pair[1], scores.loc[best_pair] * 100, "recordlinkage"

# ─────────────────────────────────────────────────────────────────────────────
# Core fuzzy matcher
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(master_df: pd.DataFrame, using_df: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    _validate_keys(master_df, keys)
    _validate_keys(using_df, keys)

    master_keys = _build_key_series(master_df, keys)
    using_keys = _build_key_series(using_df, keys)

    results = []
    for i, key_string in master_keys.items():
        # compute scores for all three methods
        idx_r, score_r, _ = _best_match_rapidfuzz(key_string, using_keys)
        idx_t, score_t, _ = _best_match_textdistance(key_string, using_keys)
        idx_l, score_l, _ = _best_match_recordlinkage(i, master_keys, using_keys)
        # identify best overall
        using_idx, best_score, method = max(
            [(idx_r, score_r, "rapidfuzz"),
             (idx_t, score_t, "textdistance"),
             (idx_l, score_l, "recordlinkage")],
            key=lambda x: x[1]
        )
        results.append({
            "master_index": i,
            "using_index": using_idx,
            "best_score": round(best_score, 2),
            "method": method,
            "rapid_score": round(score_r, 2),
            "text_score": round(score_t, 2),
            "link_score": round(score_l, 2),
        })

    link = pd.DataFrame(results).set_index("master_index")
    merged = master_df.join(link, how="left")
    merged = merged.merge(
        using_df.add_prefix("using_"),
        left_on="using_index",
        right_index=True,
        how="left"
    )
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit app interface
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Files")
    master_file = st.file_uploader("Upload MASTER file", type=["csv", "xlsx", "xls", "dta"])
    using_file = st.file_uploader("Upload USING file", type=["csv", "xlsx", "xls", "dta"])

if master_file and using_file:
    try:
        master_df = _read_file(master_file)
        using_df = _read_file(using_file)

        shared_columns = sorted(set(master_df.columns) & set(using_df.columns))
        selected_keys = st.multiselect("Select key variable(s) for matching", shared_columns)

        if selected_keys:
            if st.button("Run Fuzzy Match"):
                matched = fuzzy_match(master_df, using_df, selected_keys)
                st.success("Fuzzy matching complete.")
                st.dataframe(matched.head(50))

                file_format = st.selectbox("Choose format to download", ["csv", "xlsx", "dta"])
                filename = f"fuzzy_matched.{file_format}"

                if file_format == "csv":
                    st.download_button("Download CSV", matched.to_csv(index=False), file_name=filename)
                elif file_format == "xlsx":
                    from io import BytesIO
                    buffer = BytesIO()
                    matched.to_excel(buffer, index=False)
                    buffer.seek(0)
                    st.download_button("Download Excel", data=buffer.getvalue(), file_name=filename)
                elif file_format == "dta":
                    from io import BytesIO
                    buffer = BytesIO()
                    matched.to_stata(buffer, write_index=False)
                    buffer.seek(0)
                    st.download_button("Download Stata", data=buffer.getvalue(), file_name=filename)
        else:
            st.info("Please select one or more key variables.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload both MASTER and USING files to begin.")
