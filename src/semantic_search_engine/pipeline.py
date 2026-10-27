"""The end-to-end hybrid search pipeline: lexical + dense -> fusion -> rerank."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from semantic_search_engine.dense import DenseIndex
from semantic_search_engine.documents import Document
from semantic_search_engine.fusion import reciprocal_rank_fusion
from semantic_search_engine.lexical import BM25Index
from semantic_search_engine.rerank import CrossEncoderLike, rerank


class SearchMode(StrEnum):
    """Which stages of the pipeline to run for a query."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class SearchResult:
    """A single ranked result, carrying the score from every stage it passed through.

    Scores are ``None`` when a stage did not run or did not retrieve the
    document, which is exposed directly in the API response so the behavior
    of each stage is transparent rather than hidden behind a single number.
    """

    doc_id: str
    title: str
    text: str
    topic: str
    rank: int
    lexical_score: float | None = None
    lexical_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None


class SearchPipeline:
    """Runs lexical retrieval, dense retrieval, fusion, and optional reranking.

    ``reranker`` is optional so a pipeline can be constructed for
    lexical/dense/hybrid evaluation without ever loading the cross-encoder
    model; it is only required when ``mode=SearchMode.HYBRID_RERANK``.
    """

    def __init__(
        self,
        documents: list[Document],
        lexical_index: BM25Index,
        dense_index: DenseIndex,
        reranker: CrossEncoderLike | None = None,
        rrf_k: int = 60,
        rerank_pool_size: int = 20,
    ) -> None:
        self.documents_by_id = {doc.id: doc for doc in documents}
        self.lexical_index = lexical_index
        self.dense_index = dense_index
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.rerank_pool_size = rerank_pool_size

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID_RERANK,
        candidate_pool: int = 50,
    ) -> list[SearchResult]:
        """Run the requested pipeline stages and return up to ``top_k`` results.

        ``candidate_pool`` controls how many documents each of the lexical and
        dense stages retrieve before fusion; it should be >= ``top_k`` and
        >= ``rerank_pool_size`` for the later stages to have enough to work
        with.
        """
        lexical_results = self.lexical_index.search(query, top_k=candidate_pool)
        dense_results = self.dense_index.search(query, top_k=candidate_pool)

        lexical_scores = dict(lexical_results)
        dense_scores = dict(dense_results)
        lexical_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(lexical_results, 1)}
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_results, 1)}

        fused_scores: dict[str, float] = {}
        if mode is SearchMode.LEXICAL:
            ordered_ids = [doc_id for doc_id, _ in lexical_results]
        elif mode is SearchMode.DENSE:
            ordered_ids = [doc_id for doc_id, _ in dense_results]
        else:
            fused = reciprocal_rank_fusion([lexical_results, dense_results], k=self.rrf_k)
            fused_scores = dict(fused)
            ordered_ids = [doc_id for doc_id, _ in fused]
        fused_ranks = {doc_id: rank for rank, doc_id in enumerate(ordered_ids, 1)}

        rerank_scores: dict[str, float] = {}
        final_order = ordered_ids
        if mode is SearchMode.HYBRID_RERANK:
            if self.reranker is None:
                raise ValueError("mode=HYBRID_RERANK requires a reranker to be configured")
            candidate_ids = ordered_ids[: self.rerank_pool_size]
            candidates = [
                (doc_id, self.documents_by_id[doc_id].content) for doc_id in candidate_ids
            ]
            reranked = rerank(query, candidates, self.reranker)
            rerank_scores = dict(reranked)
            remainder = [doc_id for doc_id in ordered_ids if doc_id not in rerank_scores]
            final_order = [doc_id for doc_id, _ in reranked] + remainder

        results = []
        for rank, doc_id in enumerate(final_order[:top_k], start=1):
            doc = self.documents_by_id[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=doc.title,
                    text=doc.text,
                    topic=doc.topic,
                    rank=rank,
                    lexical_score=lexical_scores.get(doc_id),
                    lexical_rank=lexical_ranks.get(doc_id),
                    dense_score=dense_scores.get(doc_id),
                    dense_rank=dense_ranks.get(doc_id),
                    fused_score=fused_scores.get(doc_id),
                    fused_rank=fused_ranks.get(doc_id),
                    rerank_score=rerank_scores.get(doc_id),
                )
            )
        return results
