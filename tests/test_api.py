import pytest
from fastapi.testclient import TestClient

from semantic_search_engine.api import app, get_pipeline
from semantic_search_engine.dense import DenseIndex
from semantic_search_engine.lexical import BM25Index
from semantic_search_engine.pipeline import SearchPipeline


@pytest.fixture
def client(sample_documents, keyword_embedder, word_overlap_cross_encoder):
    """A TestClient wired to a fake, fully in-memory pipeline.

    Overriding the ``get_pipeline`` dependency means this test never loads a
    real embedding or cross-encoder model.
    """
    lexical_index = BM25Index(sample_documents)
    dense_index = DenseIndex(sample_documents, keyword_embedder)
    fake_pipeline = SearchPipeline(
        sample_documents,
        lexical_index,
        dense_index,
        reranker=word_overlap_cross_encoder,
    )
    app.dependency_overrides[get_pipeline] = lambda: fake_pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_search_returns_200_with_ranked_results(client):
    response = client.get("/search", params={"q": "domesticated felines"})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "domesticated felines"
    assert body["mode"] == "hybrid_rerank"
    assert body["count"] == len(body["results"])
    assert body["results"][0]["doc_id"] == "d1"


def test_search_result_exposes_every_stage_score(client):
    response = client.get("/search", params={"q": "domesticated felines"})
    result = response.json()["results"][0]
    for field in [
        "doc_id",
        "title",
        "text",
        "topic",
        "rank",
        "lexical_score",
        "lexical_rank",
        "dense_score",
        "dense_rank",
        "fused_score",
        "fused_rank",
        "rerank_score",
    ]:
        assert field in result


def test_search_respects_top_k_param(client):
    response = client.get("/search", params={"q": "domesticated", "top_k": 2})
    assert response.status_code == 200
    assert response.json()["count"] <= 2


def test_search_mode_param_selects_pipeline_stage(client):
    response = client.get(
        "/search", params={"q": "domesticated felines", "mode": "lexical"}
    )
    body = response.json()
    assert body["mode"] == "lexical"
    assert all(r["fused_score"] is None for r in body["results"])
    assert all(r["rerank_score"] is None for r in body["results"])


def test_search_rejects_invalid_mode(client):
    response = client.get("/search", params={"q": "cats", "mode": "not-a-real-mode"})
    assert response.status_code == 422


def test_search_rejects_empty_query(client):
    response = client.get("/search", params={"q": ""})
    assert response.status_code == 422


def test_health_endpoint_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
