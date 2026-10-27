"""Tunable pipeline settings, collected in one place."""

from __future__ import annotations

from dataclasses import dataclass

from semantic_search_engine.dense import DEFAULT_DENSE_MODEL_NAME
from semantic_search_engine.rerank import DEFAULT_RERANKER_MODEL_NAME


@dataclass(frozen=True)
class Settings:
    """Configuration for building a real, model-backed ``SearchPipeline``."""

    dense_model_name: str = DEFAULT_DENSE_MODEL_NAME
    reranker_model_name: str = DEFAULT_RERANKER_MODEL_NAME
    rrf_k: int = 60
    """RRF's rank-damping constant; higher values flatten the influence of rank."""
    candidate_pool: int = 50
    """How many documents each of BM25 and dense retrieval fetch before fusion."""
    rerank_pool_size: int = 20
    """How many fused results are passed to the cross-encoder reranker."""
    default_top_k: int = 10
