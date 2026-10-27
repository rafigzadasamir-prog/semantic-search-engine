import pytest

from semantic_search_engine.lexical import BM25Index, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Cats, Dogs! 123") == ["cats", "dogs", "123"]


def test_tokenize_empty_string_returns_no_tokens():
    assert tokenize("   !!! ") == []


def test_bm25_ranks_matching_document_first(sample_documents):
    index = BM25Index(sample_documents)
    results = index.search("domesticated felines chasing mice", top_k=5)
    assert results[0][0] == "d1"


def test_bm25_scores_are_sorted_descending(sample_documents):
    index = BM25Index(sample_documents)
    results = index.search("domesticated pets animals", top_k=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_excludes_documents_with_no_matching_terms(sample_documents):
    index = BM25Index(sample_documents)
    results = index.search("spacecraft nebula galaxy telescope", top_k=5)
    assert results == []


def test_bm25_search_respects_top_k(sample_documents):
    index = BM25Index(sample_documents)
    results = index.search("domesticated pets animals", top_k=1)
    assert len(results) == 1


def test_bm25_empty_query_returns_no_results(sample_documents):
    index = BM25Index(sample_documents)
    assert index.search("???", top_k=5) == []


def test_bm25_requires_at_least_one_document():
    with pytest.raises(ValueError):
        BM25Index([])
