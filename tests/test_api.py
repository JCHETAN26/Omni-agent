import pytest
from fastapi.testclient import TestClient

from backend.core.config import get_settings
from tests.conftest import load_samples


@pytest.fixture
def client(monkeypatch):
    for var in ("QDRANT_URL", "ELASTICSEARCH_URL", "REDIS_URL"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hashing")
    get_settings.cache_clear()
    from backend.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def test_health_reports_fallback_backends(client):
    body = client.get("/health").json()
    assert body["backends"] == {
        "dense": "qdrant-in-memory",
        "sparse": "bm25-in-memory",
        "cache": "in-memory",
        "embedder": "hashing-384",
    }


def test_ingest_then_search(client):
    docs = [d.model_dump() for d in load_samples()]
    report = client.post("/ingest", json={"documents": docs}).json()
    assert report["chunks"] > 0

    r = client.post("/search", json={"query": "PSU redundancy lost alert", "top_k": 3})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert hits[0]["chunk"]["section_heading"].endswith("PSU Redundancy Alerts")

    again = client.post("/search", json={"query": "PSU redundancy lost alert", "top_k": 3})
    assert again.json() == r.json()  # served from cache


def test_search_validation(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/ingest", json={"documents": []}).status_code == 422
