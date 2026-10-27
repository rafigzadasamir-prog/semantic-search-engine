"""End-to-end tests against the real embedding and cross-encoder models.

Marked ``integration`` and excluded from the default test run (see the
``addopts`` setting in pyproject.toml) because they download/load real
sentence-transformers models and are too slow and network-dependent for a
default CI run. Run explicitly with: pytest -m integration
"""

import pytest

from semantic_search_engine.factory import build_pipeline
from semantic_search_engine.pipeline import SearchMode


@pytest.mark.integration
def test_real_pipeline_ranks_a_clearly_relevant_document_first():
    pipeline = build_pipeline()
    results = pipeline.search(
        "How did the solar system form?", top_k=5, mode=SearchMode.HYBRID_RERANK
    )
    assert results
    assert results[0].doc_id == "astro-01"


@pytest.mark.integration
def test_real_pipeline_dense_mode_handles_paraphrased_query():
    # No shared vocabulary with the target document's wording, so this only
    # works if the dense/semantic stage is actually doing something useful.
    pipeline = build_pipeline()
    results = pipeline.search(
        "why do spring tides happen when the sun and moon line up",
        top_k=5,
        mode=SearchMode.DENSE,
    )
    doc_ids = [r.doc_id for r in results]
    assert "astro-05" in doc_ids
