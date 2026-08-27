import pandas as pd
import pytest

from fuzzy_core.matching import (
    best_match_name_matching,
    best_match_rapidfuzz,
    best_match_recordlinkage,
    best_match_textdistance,
    match_best,
    match_compare,
)


def test_rapidfuzz_exact_and_token_sort():
    universe = pd.Series(["alpha beta", "zzz"], index=["keep", "other"])
    idx, score, method = best_match_rapidfuzz("alpha beta", universe)
    assert method == "rapidfuzz"
    assert idx == "keep"
    assert score == pytest.approx(100.0)

    idx2, score2, _ = best_match_rapidfuzz("beta alpha", universe)
    assert idx2 == "keep"
    assert score2 == pytest.approx(100.0)


def test_rapidfuzz_empty_universe():
    idx, score, method = best_match_rapidfuzz("anything", pd.Series(dtype=str))
    assert method == "rapidfuzz"
    assert pd.isna(idx)
    assert score == 0.0


def test_textdistance_exact_is_one_hundred():
    universe = pd.Series(["acme corporation", "other"])
    idx, score, method = best_match_textdistance("acme corporation", universe)
    assert method == "textdistance"
    assert idx == 0
    assert score == pytest.approx(100.0)


def test_textdistance_prefers_closer_string():
    universe = pd.Series(["apple inc", "zzzzzzzz"])
    idx, score, _ = best_match_textdistance("apple", universe)
    assert idx == 0
    assert score > 50


def test_recordlinkage_exact_is_one_hundred():
    master = pd.Series(["acme corporation"], index=[7])
    using = pd.Series(["noise", "acme corporation"], index=[10, 11])
    idx, score, method = best_match_recordlinkage(7, master, using)
    assert method == "recordlinkage"
    assert idx == 11
    assert score == pytest.approx(100.0)


def test_recordlinkage_missing_label():
    master = pd.Series(["acme"])
    using = pd.Series(["acme"])
    idx, score, method = best_match_recordlinkage("nope", master, using)
    assert method == "recordlinkage"
    assert pd.isna(idx)
    assert score == 0.0


def test_name_matching_identical_not_scaled_by_100():
    """NameMatcher already reports ~0–100. Multiplying by 100 was a bug."""
    master = pd.Series(["acme corporation"], index=[0])
    using = pd.Series(["acme corporation", "unrelated zz"])
    idx, score, method = best_match_name_matching(0, master, using)
    assert method == "name_matching"
    assert score == pytest.approx(100.0, abs=5.0)
    assert score < 1000
    assert idx in using.index


def test_match_compare_columns_and_exact_row(master_df, using_df):
    out = match_compare(master_df, using_df, ["company"])
    for col in (
        "rapid_index",
        "rapid_score",
        "rapid_match",
        "text_index",
        "text_score",
        "text_match",
        "link_index",
        "link_score",
        "link_match",
        "name_index",
        "name_score",
        "name_match",
    ):
        assert col in out.columns

    acme = out.loc[2]
    assert acme["rapid_score"] >= 70
    assert "acme" in str(acme["rapid_match"])
    assert acme["name_score"] < 1000
    assert (out["rapid_score"].between(0, 100.0001) | out["rapid_score"].isna()).all()
    assert (out["text_score"].between(0, 100.0001) | out["text_score"].isna()).all()
    assert (out["link_score"].between(0, 100.0001) | out["link_score"].isna()).all()


def test_match_compare_composite_keys():
    master = pd.DataFrame({"name": ["Acme"], "city": ["Boston"]})
    using = pd.DataFrame(
        {"name": ["Acme", "Acme"], "city": ["Boston", "Miami"]}
    )
    out = match_compare(master, using, ["name", "city"])
    assert out.loc[0, "rapid_score"] == pytest.approx(100.0)
    assert out.loc[0, "rapid_index"] == 0


def test_match_best_joins_using_columns(master_df, using_df):
    out = match_best(master_df, using_df, ["company"])
    assert {"using_index", "best_score", "method", "using_company", "using_country"} <= set(
        out.columns
    )
    assert out["best_score"].between(0, 100.0001).all()
    assert out["method"].isin(["rapidfuzz", "textdistance", "recordlinkage"]).all()
    apple = out.loc[0]
    assert apple["using_company"] in {"Apple", "Microsoft Corp", "Acme Corp", "Banana Stand"}


def test_match_compare_missing_key_raises(master_df, using_df):
    with pytest.raises(KeyError, match="Missing key column"):
        match_compare(master_df, using_df, ["not_a_column"])
