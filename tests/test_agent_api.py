"""/ws/agent and /sse/agent end to end, over a real uvicorn server."""

import asyncio
import json

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from tests.conftest import live_server
from tests.test_websocket import collect_responses, recv


async def ask(ws, rid: str, text: str) -> list[dict]:
    await ws.send(json.dumps({"type": "query", "id": rid, "text": text}))
    _, by_id = await collect_responses(ws, [rid])
    return by_id[rid]


async def test_agent_session_resumes_thread_after_reconnect():
    async with live_server() as (ws_url, http_url, app):
        agent_url = ws_url.replace("/ws/stream", "/ws/agent")
        async with connect(agent_url) as ws:
            hello = await recv(ws)
            assert hello["type"] == "session.created" and hello["resumed"] is False
            thread = hello["thread_id"]
            events = await ask(ws, "q1", "what does EIO-5 mean?")
            assert events[0]["thread_id"] == thread
            assert events[-1]["type"] == "response.end" and events[-1]["outcome"] == "answered"
            assert any(e["type"] == "agent.step" and e["node"] == "route" for e in events)

        async with connect(f"{agent_url}?thread_id={thread}") as ws:  # reconnect
            hello = await recv(ws)
            assert hello["resumed"] is True and hello["history_turns"] == 1
            history = await recv(ws)
            assert history["type"] == "agent.history"
            assert history["messages"][0]["content"] == "what does EIO-5 mean?"
            events = await ask(ws, "q2", "and which firmware version fixes it?")
            ctx = next(e for e in events if e.get("node") == "contextualize")
            assert ctx["changed"] is True
            assert "4.2.1" in "".join(e["text"] for e in events if e["type"] == "response.token")

        async with httpx.AsyncClient() as http:
            health = (await http.get(f"{http_url}/health")).json()
        assert health["agent"] == {"reasoner": "heuristic", "memory": "InMemorySaver"}


async def test_concurrent_agent_sessions_keep_threads_separate():
    async with live_server() as (ws_url, _, _):
        agent_url = ws_url.replace("/ws/stream", "/ws/agent")

        async def convo(n: int) -> tuple[str, list[str]]:
            async with connect(agent_url) as ws:
                thread = (await recv(ws))["thread_id"]
                outcomes = []
                for i, q in enumerate(
                    [
                        "what does EIO-5 mean?",
                        "which firmware fixes it?",
                        "show me errors on storage-07",
                        "what's 64*2?",
                    ]
                ):
                    events = await ask(ws, f"s{n}q{i}", q)
                    outcomes.append(events[-1]["outcome"])
                return thread, outcomes

        results = await asyncio.gather(*(convo(n) for n in range(20)))
        assert len({t for t, _ in results}) == 20
        assert all(o == ["answered"] * 4 for _, o in results)


async def test_invalid_thread_id_is_rejected():
    async with live_server(ingest=False) as (ws_url, _, _):
        url = ws_url.replace("/ws/stream", "/ws/agent") + "?thread_id=bad%20id!"
        async with connect(url) as ws:
            with pytest.raises(ConnectionClosed) as info:
                await recv(ws)
            assert info.value.rcvd.code == 1008


async def test_sse_agent_streams_steps_and_answer():
    async with live_server() as (_, http_url, _):
        async with httpx.AsyncClient(timeout=30) as http:
            r = await http.get(f"{http_url}/sse/agent", params={"q": "what's 15% of 400?"})
    events = [json.loads(b.split("data: ", 1)[1]) for b in r.text.strip().split("\n\n")]
    assert r.headers["x-thread-id"]
    assert events[0]["type"] == "response.start"
    assert any(e.get("node") == "calculate" for e in events)
    assert events[-1]["type"] == "response.end" and events[-1]["tool"] == "calculator"
    assert "60" in "".join(e["text"] for e in events if e["type"] == "response.token")
