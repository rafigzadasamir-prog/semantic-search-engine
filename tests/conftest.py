"""Shared fixtures and fakes used across the test suite.

None of these fakes touch the network or load a real ML model, which is what
lets the default (non-integration) test run be fast and network-free.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from semantic_search_engine.documents import Document

# Each sample document contains one keyword unique within this small corpus,
# which lets tests build a deterministic "embedding" from simple keyword
# presence instead of depending on a real embedding model.
KEYWORD_DIMS = ["felines", "canines", "polar", "tides", "yeast"]


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            id="d1",
            topic="animals",
            title="Cats",
            text="Cats are small domesticated felines that enjoy chasing mice.",
        ),
        Document(
            id="d2",
            topic="animals",
            title="Dogs",
            text="Dogs are loyal domesticated canines often kept as pets.",
        ),
        Document(
            id="d3",
            topic="space",
            title="Mars",
            text="Mars is a red planet with a thin atmosphere and polar ice caps.",
        ),
        Document(
            id="d4",
            topic="space",
            title="Moon",
            text="The Moon orbits Earth and causes ocean tides through gravity.",
        ),
        Document(
            id="d5",
            topic="food",
            title="Bread",
            text="Bread is baked from flour, water, and yeast that makes dough rise.",
        ),
    ]


class CallableEmbedder:
    """An ``Embedder`` backed by a plain Python function, for deterministic tests."""

    def __init__(self, fn: Callable[[str], list[float]]) -> None:
        self._fn = fn

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array([self._fn(text) for text in texts], dtype=np.float32)


def _keyword_embed(text: str) -> list[float]:
    lowered = text.lower()
    return [1.0 if keyword in lowered else 0.0 for keyword in KEYWORD_DIMS]


@pytest.fixture
def callable_embedder_factory():
    """Factory for building a ``CallableEmbedder`` from any embed function."""
    return CallableEmbedder


@pytest.fixture
def keyword_embedder() -> CallableEmbedder:
    """Embeds ``sample_documents`` content as a one-hot vector of KEYWORD_DIMS.

    A query containing one of those keywords (e.g. "tides") gets the same
    one-hot vector as the matching document, making cosine similarity
    perfectly predictable for tests.
    """
    return CallableEmbedder(_keyword_embed)


class WordOverlapCrossEncoder:
    """Deterministic fake cross-encoder: score = number of shared lowercase words."""

    def predict(self, sentence_pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, text in sentence_pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


@pytest.fixture
def word_overlap_cross_encoder() -> WordOverlapCrossEncoder:
    return WordOverlapCrossEncoder()
