"""Dense vector retrieval using sentence embeddings and cosine similarity.

The index itself is a plain NumPy array, no vector database is required. Any
object implementing :class:`Embedder` (a single ``encode`` method returning a
2D array of row vectors) can supply the embeddings, which is what lets the
pipeline be unit-tested with a fake, deterministic embedder instead of
downloading and running the real sentence-transformers model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from semantic_search_engine.documents import Document

DEFAULT_DENSE_MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder(Protocol):
    """Anything that can turn a batch of texts into row vectors."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """Thin wrapper around a local ``sentence-transformers`` model."""

    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL_NAME) -> None:
        # Imported lazily so importing this module never requires the heavy
        # sentence-transformers / torch dependency chain or a model download
        # unless this specific class is actually instantiated.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        return np.asarray(embeddings, dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class DenseIndex:
    """In-memory dense vector index with cosine-similarity search."""

    def __init__(self, documents: list[Document], embedder: Embedder) -> None:
        if not documents:
            raise ValueError("DenseIndex requires at least one document")
        self.documents = documents
        self._doc_ids = [doc.id for doc in documents]
        self.embedder = embedder
        raw = np.asarray(embedder.encode([doc.content for doc in documents]), dtype=np.float32)
        # Embeddings are normalized here regardless of what the embedder did,
        # so cosine similarity reduces to a plain dot product at query time.
        self._embeddings = _normalize_rows(raw)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, cosine_similarity)`` pairs, best first."""
        query_vec = np.asarray(self.embedder.encode([query]), dtype=np.float32)
        query_vec = _normalize_rows(query_vec)[0]
        similarities = self._embeddings @ query_vec
        order = np.argsort(-similarities)[:top_k]
        return [(self._doc_ids[i], float(similarities[i])) for i in order]
