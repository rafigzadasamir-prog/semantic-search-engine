"""Cross-encoder reranking of candidate documents.

The reranking stage re-scores a short list of candidates (e.g. the top 20
from hybrid fusion) with a cross-encoder that jointly encodes the query and
each candidate's text, which is more accurate but far more expensive per
document than the independent query/document encoding used for dense
retrieval. Running it only on a small candidate set keeps it affordable.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderLike(Protocol):
    """Anything exposing the same ``predict`` interface as ``CrossEncoder``."""

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        ...


class CrossEncoderReranker:
    """Wraps a ``sentence_transformers.CrossEncoder`` model."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL_NAME) -> None:
        # Imported lazily for the same reason as SentenceTransformerEmbedder:
        # importing this module should not require torch or a model download.
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        return [float(score) for score in self._model.predict(sentence_pairs)]


def rerank(
    query: str,
    candidates: list[tuple[str, str]],
    cross_encoder: CrossEncoderLike,
) -> list[tuple[str, float]]:
    """Re-score ``candidates`` (doc_id, text) pairs for ``query``.

    Returns ``(doc_id, rerank_score)`` pairs sorted by descending score. An
    empty candidate list returns an empty result without calling the model.
    """
    if not candidates:
        return []
    pairs = [(query, text) for _, text in candidates]
    scores = cross_encoder.predict(pairs)
    doc_ids = [doc_id for doc_id, _ in candidates]
    ranked = sorted(zip(doc_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [(doc_id, float(score)) for doc_id, score in ranked]
