import pandas as pd
import pytest

from fuzzy_core.matching import (
    best_match_rapidfuzz,
    best_match_recordlinkage,
    best_match_textdistance,
    match_compare,
)


def test_identical_strings_score_one_hundred_across_string_engines():
    universe = pd.Series(["alpha beta gamma"])
    _, rf, _ = best_match_rapidfuzz("alpha beta gamma", universe)
    _, td, _ = best_match_textdistance("alpha beta gamma", universe)
    master = pd.Series(["alpha beta gamma"])
    _, rl_score, _ = best_match_recordlinkage(0, master, universe)
    assert rf == pytest.approx(100.0)
    assert td == pytest.approx(100.0)
    assert rl_score == pytest.approx(100.0)


def test_compare_scores_are_floats_rounded_to_two_decimals():
    master = pd.DataFrame({"k": ["Apple Inc"]})
    using = pd.DataFrame({"k": ["Apple"]})
    out = match_compare(master, using, ["k"])
    for col in ("rapid_score", "text_score", "link_score", "name_score"):
        value = out.loc[0, col]
        assert isinstance(value, float)
        # two-decimal rounding: 12.345 -> 12.35, so 100*value is near-integer at 0.01
        assert round(value, 2) == pytest.approx(value)
