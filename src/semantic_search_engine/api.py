"""FastAPI service exposing the hybrid search pipeline over HTTP."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

from fastapi import Depends, FastAPI, Query
from pydantic import BaseModel

from semantic_search_engine.factory import build_pipeline
from semantic_search_engine.pipeline import SearchMode, SearchPipeline, SearchResult

app = FastAPI(
    title="Semantic Search Engine",
    description=(
        "Hybrid lexical (BM25) + dense (sentence embeddings) search with "
        "reciprocal-rank-fusion and cross-encoder reranking."
    ),
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_pipeline() -> SearchPipeline:
    """Build and cache the real, model-backed pipeline.

    ``lru_cache`` ensures the embedding and reranker models are loaded only
    once per process, on the first request that needs them. FastAPI resolves
    this function as a dependency, so tests replace it wholesale via
    ``app.dependency_overrides`` instead of waiting on real models.
    """
    return build_pipeline()


class SearchResultResponse(BaseModel):
    """One ranked result with per-stage scores, for pipeline transparency."""

    doc_id: str
    title: str
    text: str
    topic: str
    rank: int
    lexical_score: float | None
    lexical_rank: int | None
    dense_score: float | None
    dense_rank: int | None
    fused_score: float | None
    fused_rank: int | None
    rerank_score: float | None

    @classmethod
    def from_result(cls, result: SearchResult) -> SearchResultResponse:
        return cls(**asdict(result))


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[SearchResultResponse]


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return"),
    mode: SearchMode = Query(
        SearchMode.HYBRID_RERANK, description="Which pipeline stages to run"
    ),
    pipeline: SearchPipeline = Depends(get_pipeline),
) -> SearchResponse:
    """Search the corpus and return ranked results with per-stage scores."""
    results = pipeline.search(q, top_k=top_k, mode=mode)
    return SearchResponse(
        query=q,
        mode=mode.value,
        count=len(results),
        results=[SearchResultResponse.from_result(r) for r in results],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
