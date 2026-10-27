import pytest

from semantic_search_engine.dense import DenseIndex


def test_dense_index_ranks_matching_document_first(sample_documents, keyword_embedder):
    index = DenseIndex(sample_documents, keyword_embedder)
    results = index.search("what causes ocean tides", top_k=5)
    assert results[0][0] == "d4"


def test_dense_index_scores_are_valid_cosine_similarities(sample_documents, keyword_embedder):
    index = DenseIndex(sample_documents, keyword_embedder)
    results = index.search("polar ice caps", top_k=5)
    for _, score in results:
        assert -1.0001 <= score <= 1.0001


def test_dense_index_respects_top_k(sample_documents, keyword_embedder):
    index = DenseIndex(sample_documents, keyword_embedder)
    results = index.search("tides", top_k=2)
    assert len(results) == 2


def test_dense_index_handles_zero_vector_without_crashing(
    sample_documents, callable_embedder_factory
):
    # An embedder that always returns an all-zero vector would otherwise cause
    # a division-by-zero when normalizing.
    embedder = callable_embedder_factory(lambda _text: [0.0, 0.0])
    index = DenseIndex(sample_documents, embedder)
    results = index.search("anything", top_k=3)
    assert len(results) == 3


def test_dense_index_requires_at_least_one_document(callable_embedder_factory):
    embedder = callable_embedder_factory(lambda _text: [1.0])
    with pytest.raises(ValueError):
        DenseIndex([], embedder)
