import pytest

from search.hybrid_retriever import reciprocal_rank_fusion


def test_rrf_matches_formula_with_k60():
    dense = ["a", "b", "c"]
    sparse = ["c", "a", "d"]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_order_rewards_agreement():
    order = [d for d, _ in reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]], k=60)]
    assert order == ["a", "c", "b", "d"]


def test_rrf_consensus_beats_single_top_rank():
    # y is #2 in both lists (2/62) and beats x, which is #1 in only one (1/61).
    assert [d for d, _ in reciprocal_rank_fusion([["x", "y"], ["z", "y"]])][0] == "y"


def test_rrf_exact_ties_are_stable():
    # c appears in both lists (2/62) and outranks a and b (1/61 each), which tie.
    fused = reciprocal_rank_fusion([["a", "c"], ["b", "c"]], k=60)
    assert [d for d, _ in fused] == ["c", "a", "b"]
    assert fused[1][1] == fused[2][1]


def test_rrf_ignores_duplicates_within_a_list_and_handles_empty():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"], []], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)
    assert reciprocal_rank_fusion([[], []]) == []
