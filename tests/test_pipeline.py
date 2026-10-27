import pytest

from semantic_search_engine.dense import DenseIndex
from semantic_search_engine.lexical import BM25Index
from semantic_search_engine.pipeline import SearchMode, SearchPipeline


@pytest.fixture
def pipeline(sample_documents, keyword_embedder, word_overlap_cross_encoder):
    lexical_index = BM25Index(sample_documents)
    dense_index = DenseIndex(sample_documents, keyword_embedder)
    return SearchPipeline(
        sample_documents,
        lexical_index,
        dense_index,
        reranker=word_overlap_cross_encoder,
        rrf_k=60,
        rerank_pool_size=5,
    )


@pytest.fixture
def pipeline_without_reranker(sample_documents, keyword_embedder):
    lexical_index = BM25Index(sample_documents)
    dense_index = DenseIndex(sample_documents, keyword_embedder)
    return SearchPipeline(sample_documents, lexical_index, dense_index, reranker=None)


def test_lexical_mode_ranks_by_bm25_and_skips_fusion_rerank(pipeline):
    results = pipeline.search("domesticated felines", top_k=5, mode=SearchMode.LEXICAL)
    assert results
    assert results[0].doc_id == "d1"
    assert results[0].lexical_score is not None
    assert all(r.fused_score is None for r in results)
    assert all(r.rerank_score is None for r in results)


def test_dense_mode_ranks_by_cosine_similarity(pipeline):
    results = pipeline.search("ocean tides", top_k=5, mode=SearchMode.DENSE)
    assert results[0].doc_id == "d4"
    assert results[0].dense_score is not None
    assert all(r.fused_score is None for r in results)
    assert all(r.rerank_score is None for r in results)


def test_hybrid_mode_populates_fused_score_but_not_rerank_score(pipeline):
    results = pipeline.search("domesticated felines", top_k=5, mode=SearchMode.HYBRID)
    assert results
    assert all(r.fused_score is not None for r in results)
    assert all(r.rerank_score is None for r in results)


def test_hybrid_rerank_mode_populates_rerank_score(pipeline):
    results = pipeline.search("domesticated felines", top_k=5, mode=SearchMode.HYBRID_RERANK)
    assert results
    assert any(r.rerank_score is not None for r in results)


def test_hybrid_rerank_without_reranker_configured_raises(pipeline_without_reranker):
    with pytest.raises(ValueError):
        pipeline_without_reranker.search("felines", mode=SearchMode.HYBRID_RERANK)


def test_search_respects_top_k(pipeline):
    results = pipeline.search("domesticated", top_k=2, mode=SearchMode.HYBRID)
    assert len(results) <= 2


def test_search_result_ranks_are_sequential_starting_at_one(pipeline):
    results = pipeline.search("domesticated felines", top_k=5, mode=SearchMode.HYBRID_RERANK)
    assert [r.rank for r in results] == list(range(1, len(results) + 1))


def test_results_carry_document_metadata(pipeline, sample_documents):
    results = pipeline.search("domesticated felines", top_k=1, mode=SearchMode.LEXICAL)
    top = results[0]
    expected = next(doc for doc in sample_documents if doc.id == top.doc_id)
    assert top.title == expected.title
    assert top.text == expected.text
    assert top.topic == expected.topic
