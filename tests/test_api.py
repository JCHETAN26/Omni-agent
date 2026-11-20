import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from backend.core.cache import InMemoryCache
from backend.core.services import Services
from backend.main import create_app
from tests.conftest import load_samples, make_settings


@pytest.fixture
def client():
    with TestClient(create_app(make_settings())) as c:
        yield c


def test_health_reports_fallback_backends(client):
    body = client.get("/health").json()
    assert body["backends"] == {
        "dense": "qdrant-in-memory",
        "sparse": "bm25-in-memory",
        "cache": "in-memory",
        "rate_limiter": "in-memory",
        "embedder": "hashing-384",
        "inference": "mock",
        "ranker": "rrf",
        "vision": "off",
    }
    assert body["inference_ready"] is True
    assert body["websocket"]["active"] == 0


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


def test_search_before_ingest_returns_no_hits(client):
    r = client.post("/search", json={"query": "anything"})
    assert r.status_code == 200 and r.json()["hits"] == []


def test_search_invalidates_cached_empty_results_after_ingest(client):
    payload = {"query": "PSU redundancy lost"}
    assert client.post("/search", json=payload).json()["hits"] == []
    client.post("/ingest", json={"documents": [d.model_dump() for d in load_samples()]})
    assert client.post("/search", json=payload).json()["hits"]


def test_local_indexes_do_not_share_cache_entries_between_workers():
    shared_cache = InMemoryCache()
    with (
        TestClient(create_app(make_settings())) as first,
        TestClient(create_app(make_settings())) as second,
    ):
        for api, suffix in ((first, "one"), (second, "two")):
            api.app.state.services.cache = shared_cache
            api.post(
                "/ingest",
                json={
                    "documents": [
                        {"doc_id": suffix, "title": suffix, "text": "Netdev queue timeout."}
                    ]
                },
            ).raise_for_status()
        payload = {"query": "Netdev queue timeout"}
        first_hit = first.post("/search", json=payload).json()["hits"][0]
        second_hit = second.post("/search", json=payload).json()["hits"][0]
        assert [hit["chunk"]["doc_id"] for hit in (first_hit, second_hit)] == ["one", "two"]


def test_partial_ingest_failure_invalidates_cached_results(client, monkeypatch):
    payload = {"query": "PSU redundancy lost"}
    assert client.post("/search", json=payload).json()["hits"] == []
    processor = client.app.state.services.processor
    original = processor._dense_branch

    async def store_then_fail(chunks):
        await original(chunks)
        raise RuntimeError("late dense acknowledgement failure")

    monkeypatch.setattr(processor, "_dense_branch", store_then_fail)
    with pytest.raises(RuntimeError, match="late dense acknowledgement"):
        client.post("/ingest", json={"documents": [d.model_dump() for d in load_samples()]})
    assert client.post("/search", json=payload).json()["hits"]


def test_shared_index_search_bypasses_worker_cache(monkeypatch):
    build = Services.build

    async def build_test_indexes(_):
        return await build(make_settings())

    async def unexpected_cache_lookup(_):
        pytest.fail("a process-local generation can't validate a shared-index cache entry")

    monkeypatch.setattr(Services, "build", build_test_indexes)
    with TestClient(create_app(make_settings(qdrant_url="http://unused.invalid"))) as api:
        monkeypatch.setattr(api.app.state.services.cache, "get", unexpected_cache_lookup)
        assert api.post("/search", json={"query": "netdev timeout"}).status_code == 200


def test_search_validation(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/ingest", json={"documents": []}).status_code == 422


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        ev = json.loads(fields["data"])
        assert fields["event"] == ev["type"]
        events.append(ev)
    return events


def test_sse_streams_complete_response(client):
    client.post("/ingest", json={"documents": [d.model_dump() for d in load_samples()]})
    r = client.get("/sse/stream", params={"q": "EIO-5 controller timeout", "top_k": 2})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert [e["type"] for e in events[:2]] == ["response.start", "response.sources"]
    assert events[1]["sources"][0]["section_heading"].endswith("I/O Timeouts (EIO-5)")
    tokens = [e for e in events if e["type"] == "response.token"]
    end = events[-1]
    assert end["type"] == "response.end" and end["tokens"] == len(tokens)
    text = "".join(t["text"] for t in tokens)
    assert hashlib.sha256(text.encode()).hexdigest() == end["sha256"]


def test_sse_honours_traceparent_header(client, spans):
    trace_id = "0af7651916cd43dd8448eb211c80319c"
    r = client.get(
        "/sse/stream",
        params={"q": "disk"},
        headers={"traceparent": f"00-{trace_id}-b7ad6b7169203331-01"},
    )
    assert _parse_sse(r.text)[0]["trace_id"] == trace_id


@pytest.mark.parametrize("endpoint", ["/sse/stream", "/sse/agent"])
def test_conflicting_limits_are_cited_on_sse(client, endpoint):
    client.post(
        "/ingest",
        json={
            "documents": [
                {
                    "doc_id": str(value),
                    "title": f"Runbook {value}",
                    "text": f"max_connections = {value} for storage-07 in production.",
                }
                for value in (100, 256)
            ]
        },
    ).raise_for_status()
    events = _parse_sse(
        client.get(
            endpoint,
            params={
                "q": "What's the production connection limit for storage-07?",
                "top_k": 1,
            },
        ).text
    )
    end = events[-1]
    assert end["outcome"] == "abstained"
    text = "".join(e["text"] for e in events if e["type"] == "response.token")
    assert hashlib.sha256(text.encode()).hexdigest() == end["sha256"]
    sources = next(e["sources"] for e in events if e["type"] == "response.sources")
    assert len(sources) == 2
    for i, source in enumerate(sources, 1):
        assert f"`{source['document_title'].split()[-1]}` [{i}]" in text


def test_scoped_agent_answer_excludes_other_environments(client):
    documents = [
        {
            "doc_id": stage,
            "title": f"{stage} storage limits",
            "text": f"max_connections = {value} for storage-07 in {stage}.",
        }
        for stage, value in (("production", 900), ("staging", 75))
    ]
    client.post("/ingest", json={"documents": documents}).raise_for_status()
    result = client.get(
        "/sse/agent", params={"q": "What is max_connections for storage-07 in staging?"}
    )
    events = _parse_sse(result.text)
    answer = "".join(e["text"] for e in events if e["type"] == "response.token")
    assert events[-1]["outcome"] == "answered"
    assert "75" in answer and "900" not in answer
    sources = next(e["sources"] for e in events if e["type"] == "response.sources")
    assert {source["doc_id"] for source in sources} == {"staging"}


def test_probes_and_prometheus_metrics(client):
    assert client.get("/livez").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    with client.websocket_connect("/ws/agent") as ws:
        ws.receive_json()
        ws.send_json({"type": "query", "id": "m1", "text": "what's 6*7?"})
        while ws.receive_json()["type"] != "response.end":
            pass
    body = client.get("/metrics").text
    assert 'omniagent_agent_turns_total{outcome="answered",tool="calculator"}' in body
    assert "omniagent_tokens_streamed_total" in body
    assert 'omniagent_turn_seconds_bucket{endpoint="agent"' in body
    assert "omniagent_event_loop_lag_max_seconds" in body


def test_seed_path_ingests_at_startup():
    with TestClient(create_app(make_settings(seed_path="samples"))) as c:
        r = c.post("/search", json={"query": "PSU redundancy lost", "top_k": 1})
        assert r.json()["hits"][0]["chunk"]["section_heading"].endswith("PSU Redundancy Alerts")


def test_seed_cli(capsys):
    import asyncio
    import json
    import os

    from backend.core.config import get_settings
    from backend.ingestion import seed

    get_settings.cache_clear()
    old = os.environ.get("EMBEDDING_PROVIDER")
    os.environ["EMBEDDING_PROVIDER"] = "hashing"
    try:
        asyncio.run(seed.main("samples"))
    finally:
        get_settings.cache_clear()
        if old is None:
            os.environ.pop("EMBEDDING_PROVIDER")
        else:
            os.environ["EMBEDDING_PROVIDER"] = old
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["documents"] == 3 and out["report"]["chunks"] >= 8
