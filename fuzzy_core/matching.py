"""Fuzzy matching engines and the two merge strategies used by the UIs.

``match_compare`` (Streamlit) scores every master row with four libraries and
keeps all four answers side by side.

``match_best`` (Tkinter) takes the highest of three scores and joins the
winning USING row. Those three numbers are not calibrated to each other, so
"best" only means "largest raw score."
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import pandas as pd
import recordlinkage as rl
import textdistance
from name_matching.name_matcher import NameMatcher
from rapidfuzz import fuzz, process

from fuzzy_core.parsers import build_key_series, require_columns

logging.getLogger("recordlinkage").setLevel(logging.ERROR)

DEFAULT_NAME_METRICS = ["bag", "typo", "refined_soundex"]

EngineResult = tuple[Any, float, str]


def _empty_result(method: str) -> EngineResult:
    return pd.NA, 0.0, method


def _round_score(score: Any) -> float:
    try:
        if pd.isna(score):
            return 0.0
        return round(float(score), 2)
    except (TypeError, ValueError):
        return 0.0


def _matched_value(universe: pd.Series, idx: Any) -> Any:
    if idx is pd.NA or (not isinstance(idx, (pd.Series, pd.Index)) and pd.isna(idx)):
        return pd.NA
    try:
        return universe.loc[idx]
    except (KeyError, TypeError, ValueError):
        return pd.NA


def best_match_rapidfuzz(target: str, universe: pd.Series) -> EngineResult:
    """Token-sort ratio; RapidFuzz returns a 0–100 score and the Series label."""
    if universe.empty:
        return _empty_result("rapidfuzz")
    result = process.extractOne(target, universe, scorer=fuzz.token_sort_ratio)
    if result is None:
        return _empty_result("rapidfuzz")
    _match, score, idx = result
    return idx, float(score), "rapidfuzz"


def best_match_textdistance(target: str, universe: pd.Series) -> EngineResult:
    """Jaro–Winkler similarity scaled to 0–100 to sit next to RapidFuzz."""
    if universe.empty:
        return _empty_result("textdistance")
    sims = universe.map(
        lambda x: textdistance.jaro_winkler.normalized_similarity(target, "" if pd.isna(x) else str(x))
    )
    if sims.empty or bool(sims.isna().all()):
        return _empty_result("textdistance")
    idx = sims.idxmax()
    return idx, float(sims.loc[idx]) * 100, "textdistance"


def best_match_recordlinkage(
    label: Any, master_keys: pd.Series, using_keys: pd.Series
) -> EngineResult:
    """Jaro similarity via RecordLinkage, scaled to 0–100.

    Uses a full (Cartesian) index of one master row against every USING row.
    That is O(m) per call and O(n×m) for a whole file — fine for demos, not
    for large linkage jobs.
    """
    if using_keys.empty or label not in master_keys.index:
        return _empty_result("recordlinkage")
    master_single = master_keys.loc[[label]].to_frame(name="key")
    using_df = using_keys.to_frame(name="key")
    pairs = rl.index.Full().index(master_single, using_df)
    compare = rl.Compare()
    compare.string("key", "key", method="jaro", label="jw")
    scores_df = compare.compute(pairs, master_single, using_df)
    scores = scores_df["jw"]
    if scores.empty:
        return _empty_result("recordlinkage")
    best_pair = scores.idxmax()
    return best_pair[1], float(scores.loc[best_pair]) * 100, "recordlinkage"


def _name_matcher(distance_metrics: Optional[Iterable[str]] = None) -> NameMatcher:
    matcher = NameMatcher(
        number_of_matches=1,
        top_n=1,
        legal_suffixes=True,
        common_words=False,
        verbose=False,
    )
    metrics = list(distance_metrics) if distance_metrics else []
    if metrics:
        matcher.set_distance_metrics(metrics)
    return matcher


def best_matches_name_matching(
    master_keys: pd.Series,
    using_keys: pd.Series,
    distance_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Match every master key against USING in one NameMatcher pass.

    Scores are used as the library returns them (about 0–100, sometimes
    negative). They are **not** multiplied by 100.
    """
    out = pd.DataFrame(
        {
            "match_index": pd.Series(pd.NA, index=master_keys.index, dtype="object"),
            "score": pd.Series(0.0, index=master_keys.index),
            "match": pd.Series(pd.NA, index=master_keys.index, dtype="object"),
        }
    )
    if master_keys.empty or using_keys.empty:
        return out
    matcher = _name_matcher(distance_metrics)
    using_df = using_keys.to_frame(name="key")
    master_df = master_keys.to_frame(name="key")
    matcher.load_and_process_master_data(
        column="key", df_matching_data=using_df, transform=True
    )
    matches = matcher.match_names(to_be_matched=master_df, column_matching="key")
    if matches is None or matches.empty:
        return out
    for master_idx, row in matches.iterrows():
        uid = row.get("match_index", pd.NA)
        out.at[master_idx, "match_index"] = uid
        out.at[master_idx, "score"] = _round_score(row.get("score", 0.0))
        out.at[master_idx, "match"] = _matched_value(using_keys, uid)
    return out


def best_match_name_matching(
    label: Any,
    master_keys: pd.Series,
    using_keys: pd.Series,
    distance_metrics: Optional[Iterable[str]] = None,
) -> EngineResult:
    """Single-row wrapper around :func:`best_matches_name_matching`."""
    if label not in master_keys.index:
        return _empty_result("name_matching")
    row = best_matches_name_matching(
        master_keys.loc[[label]], using_keys, distance_metrics
    ).loc[label]
    return row["match_index"], float(row["score"]), "name_matching"


def match_compare(
    master_df: pd.DataFrame,
    using_df: pd.DataFrame,
    keys: list[str],
    distance_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Return MASTER rows plus four side-by-side engine results (Streamlit)."""
    require_columns(master_df, keys)
    require_columns(using_df, keys)
    master_keys = build_key_series(master_df, keys)
    using_keys = build_key_series(using_df, keys)

    name_hits = best_matches_name_matching(master_keys, using_keys, distance_metrics)

    records: list[dict[str, Any]] = []
    for label, key_string in master_keys.items():
        idx_r, score_r, _ = best_match_rapidfuzz(key_string, using_keys)
        idx_t, score_t, _ = best_match_textdistance(key_string, using_keys)
        idx_l, score_l, _ = best_match_recordlinkage(label, master_keys, using_keys)
        name_row = name_hits.loc[label]
        records.append(
            {
                "master_index": label,
                "rapid_index": idx_r,
                "rapid_score": _round_score(score_r),
                "rapid_match": _matched_value(using_keys, idx_r),
                "text_index": idx_t,
                "text_score": _round_score(score_t),
                "text_match": _matched_value(using_keys, idx_t),
                "link_index": idx_l,
                "link_score": _round_score(score_l),
                "link_match": _matched_value(using_keys, idx_l),
                "name_index": name_row["match_index"],
                "name_score": _round_score(name_row["score"]),
                "name_match": name_row["match"],
            }
        )

    link = pd.DataFrame.from_records(records).set_index("master_index")
    return master_df.join(link, how="left")


def match_best(
    master_df: pd.DataFrame, using_df: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    """Join each MASTER row to the USING row with the highest of three scores (Tkinter)."""
    require_columns(master_df, keys)
    require_columns(using_df, keys)
    master_keys = build_key_series(master_df, keys)
    using_keys = build_key_series(using_df, keys)

    records: list[dict[str, Any]] = []
    for label, key_string in master_keys.items():
        scores = [
            best_match_rapidfuzz(key_string, using_keys),
            best_match_textdistance(key_string, using_keys),
            best_match_recordlinkage(label, master_keys, using_keys),
        ]
        using_idx, best_score, method = max(scores, key=lambda item: item[1])
        records.append(
            {
                "master_index": label,
                "using_index": using_idx,
                "best_score": _round_score(best_score),
                "method": method,
            }
        )

    link_df = pd.DataFrame.from_records(records).set_index("master_index")
    merged = master_df.join(link_df, how="left")
    return merged.merge(
        using_df.add_prefix("using_"),
        left_on="using_index",
        right_index=True,
        how="left",
    )
