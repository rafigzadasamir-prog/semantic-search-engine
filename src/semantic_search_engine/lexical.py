"""Sparse lexical retrieval using BM25."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from semantic_search_engine.documents import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric tokenizer shared by indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 index over a fixed set of documents.

    Wraps ``rank_bm25.BM25Okapi`` so the rest of the pipeline only deals with
    document ids and scores, not tokenization details.
    """

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("BM25Index requires at least one document")
        self.documents = documents
        self._doc_ids = [doc.id for doc in documents]
        self._tokenized_corpus = [tokenize(doc.content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to ``top_k`` ``(doc_id, score)`` pairs, best score first.

        Documents that score zero (no matching terms) are excluded, matching
        the intuition that BM25 has no opinion about them.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self._doc_ids, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [(doc_id, float(score)) for doc_id, score in ranked if score > 0][:top_k]
