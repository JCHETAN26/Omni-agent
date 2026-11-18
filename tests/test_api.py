import hashlib
import json

import pytest
from fastapi.testclient import TestClient

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
