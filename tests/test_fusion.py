import pytest

from semantic_search_engine.fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_rrf_rewards_documents_ranked_highly_in_multiple_lists():
    lexical = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    dense = [("b", 0.9), ("a", 0.5), ("d", 0.2)]
    fused = reciprocal_rank_fusion([lexical, dense], k=60)
    fused_ids = [doc_id for doc_id, _ in fused]

    # "a" and "b" each appear near the top of both lists, so both should
    # outrank "c" and "d", which only appear in one list each.
    assert set(fused_ids[:2]) == {"a", "b"}
    assert set(fused_ids[2:]) == {"c", "d"}


def test_rrf_single_list_preserves_relative_order():
    lexical = [("a", 10.0), ("b", 5.0), ("c", 1.0)]
    fused = reciprocal_rank_fusion([lexical])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_rrf_scores_are_positive_and_descending():
    fused = reciprocal_rank_fusion([[("a", 1.0), ("b", 1.0)], [("b", 1.0), ("a", 1.0)]])
    scores = [score for _, score in fused]
    assert all(score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_rrf_rejects_non_positive_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[("a", 1.0)]], k=0)


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_weighted_fusion_normalizes_then_weights_lists():
    lexical = [("a", 100.0), ("b", 0.0)]
    dense = [("b", 1.0), ("a", 0.0)]
    # Weighting dense twice as heavily should let "b" (top of dense) win
    # even though "a" tops the (lower-weighted) lexical list.
    fused = weighted_score_fusion([lexical, dense], weights=[1.0, 2.0])
    assert fused[0][0] == "b"


def test_weighted_fusion_defaults_to_equal_weights():
    identical = [("a", 1.0), ("b", 0.0)]
    fused = weighted_score_fusion([identical, identical])
    assert fused[0][0] == "a"


def test_weighted_fusion_handles_constant_scores_without_division_by_zero():
    constant = [("a", 5.0), ("b", 5.0)]
    fused = weighted_score_fusion([constant])
    assert {doc_id for doc_id, _ in fused} == {"a", "b"}


def test_weighted_fusion_requires_matching_weight_count():
    with pytest.raises(ValueError):
        weighted_score_fusion([[("a", 1.0)]], weights=[1.0, 2.0])
