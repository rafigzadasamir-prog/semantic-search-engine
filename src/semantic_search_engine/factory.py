"""Assembles a fully real, model-backed :class:`SearchPipeline`.

Importing and calling :func:`build_pipeline` loads the sentence-transformers
embedding model and the cross-encoder reranker, which requires those models
to already be cached locally (downloaded during ``pip install`` / first run,
per the project README) and is relatively slow. Keeping this assembly logic
out of ``pipeline.py`` lets the pipeline's ranking/fusion behavior be unit
tested with fast fake embedders and rerankers, never touching this module.
"""

from __future__ import annotations

from semantic_search_engine.config import Settings
from semantic_search_engine.dense import DenseIndex, SentenceTransformerEmbedder
from semantic_search_engine.documents import load_corpus
from semantic_search_engine.lexical import BM25Index
from semantic_search_engine.pipeline import SearchPipeline
from semantic_search_engine.rerank import CrossEncoderReranker


def build_pipeline(settings: Settings | None = None) -> SearchPipeline:
    """Build a ``SearchPipeline`` over the bundled corpus using real models."""
    settings = settings or Settings()
    documents = load_corpus()
    lexical_index = BM25Index(documents)
    embedder = SentenceTransformerEmbedder(settings.dense_model_name)
    dense_index = DenseIndex(documents, embedder)
    reranker = CrossEncoderReranker(settings.reranker_model_name)
    return SearchPipeline(
        documents,
        lexical_index,
        dense_index,
        reranker,
        rrf_k=settings.rrf_k,
        rerank_pool_size=settings.rerank_pool_size,
    )
