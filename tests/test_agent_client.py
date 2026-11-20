import hashlib
import io
import json

import pytest

from benchmarks import agent_client


def response(events):
    body = b"".join(
        f"id: {i}\ndata: {json.dumps(event)}\n\n".encode() for i, event in enumerate(events, 1)
    )
    stream = io.BytesIO(body)
    stream.headers = {"x-thread-id": "test-thread"}
    return stream


def test_ask_checks_integrity_and_reuses_thread(monkeypatch):
    text = "Read the runbook. [1]"
    valid = response(
        [
            {"type": "response.sources", "sources": [{"doc_id": "runbooks/example.md"}]},
            {"type": "response.token", "index": 0, "text": text},
            {
                "type": "response.end",
                "tokens": 1,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "outcome": "answered",
                "tool": "knowledge_base",
            },
        ]
    )

    def open_followup(url, timeout):
        assert "thread_id=test-thread" in url
        assert "q=Which+one%3F" in url
        return valid

    monkeypatch.setattr(agent_client, "urlopen", open_followup)
    out = agent_client.ask("http://127.0.0.1:8000", "Which one?", "test-thread")
    assert out["text"] == text
    assert out["sources"][0]["doc_id"] == "runbooks/example.md"
    assert out["total_ms"] >= out["ttft_ms"] >= 0


@pytest.mark.parametrize(
    ("events", "error"),
    [
        ([{"type": "response.token", "index": 1, "text": "bad"}], "token sequence"),
        ([{"type": "response.end", "tokens": 0, "sha256": "bad"}], "integrity"),
        ([], "without a complete answer"),
    ],
)
def test_ask_rejects_incomplete_or_invalid_stream(monkeypatch, events, error):
    monkeypatch.setattr(agent_client, "urlopen", lambda *_args, **_kwargs: response(events))
    with pytest.raises(agent_client.GatewayError, match=error):
        agent_client.ask("http://127.0.0.1:8000", "Question")
