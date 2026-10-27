import pytest

from semantic_search_engine.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k_counts_relevant_hits_in_top_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}
    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # only "b" in top 2
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # "b" and "d" in top 4


def test_precision_at_k_with_no_results_is_zero():
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_precision_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)


def test_recall_at_k_counts_fraction_of_relevant_found():
    retrieved = ["a", "x", "y", "b"]
    relevant = {"a", "b", "c"}
    assert recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)


def test_recall_at_k_with_no_relevant_documents_is_zero():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_recall_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=0)


def test_reciprocal_rank_of_first_relevant_hit():
    assert reciprocal_rank(["x", "y", "a", "b"], {"a", "b"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_one_when_first_result_is_relevant():
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_relevant_found():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0
