"""WebSocket gateway tests against a real uvicorn server and real WS clients."""

import asyncio
import hashlib
import json
import time

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from tests.conftest import live_server

QUERIES = [
    "EIO-5 controller timeout",
    "MTU mismatch VXLAN packet loss",
    "PSU redundancy lost",
    "pods slow DNS ndots",
]


async def recv(ws, timeout: float = 5.0) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def collect_responses(ws, ids: list[str], timeout: float = 15.0) -> tuple[list[dict], dict]:
    """Read frames until every id has a terminal event. Returns (all frames, by-id)."""
    frames: list[dict] = []
    pending = set(ids)
    deadline = time.monotonic() + timeout
    while pending:
        ev = await recv(ws, max(0.1, deadline - time.monotonic()))
        frames.append(ev)
        if ev["type"] == "ping":
            await ws.send(json.dumps({"type": "pong", "ts": ev["ts"]}))
        if ev["type"] in ("response.end", "response.error", "rate_limited") and ev.get("id"):
            pending.discard(ev["id"])
    return frames, {i: [f for f in frames if f.get("id") == i] for i in ids}


def assert_response_integrity(events: list[dict]) -> None:
    types = [e["type"] for e in events]
    assert types[0] == "response.start"
    assert types[1] == "response.sources"
    assert types[-1] == "response.end"
    tokens = [e for e in events if e["type"] == "response.token"]
    assert [t["index"] for t in tokens] == list(range(len(tokens)))  # contiguous, ordered
    end = events[-1]
    assert end["tokens"] == len(tokens)
    text = "".join(t["text"] for t in tokens)
    assert hashlib.sha256(text.encode()).hexdigest() == end["sha256"]


async def test_concurrent_sessions_stream_complete_ordered_responses():
    sessions, per_session = 50, len(QUERIES)
    async with live_server() as (ws_url, _, app):

        async def client(n: int) -> tuple[str, list[dict], dict]:
            async with connect(ws_url) as ws:
                hello = await recv(ws)
                assert hello["type"] == "session.created"
                ids = [f"s{n}-q{i}" for i in range(per_session)]
                for i, q in zip(ids, QUERIES, strict=True):  # pipelined: no waiting
                    await ws.send(json.dumps({"type": "query", "id": i, "text": q, "top_k": 3}))
                frames, by_id = await collect_responses(ws, ids)
                return hello["session_id"], [hello, *frames], by_id

        results = await asyncio.gather(*(client(n) for n in range(sessions)))
        assert len({sid for sid, _, _ in results}) == sessions  # distinct sessions

        for _, frames, by_id in results:
            seqs = [f["seq"] for f in frames]
            assert seqs == list(range(1, len(seqs) + 1))  # no gaps, no reordering
            for events in by_id.values():
                assert_response_integrity(events)
            # requests in one session are answered in arrival order, not interleaved
            ordered = [f["id"] for f in frames if f["type"] == "response.start"]
            assert ordered == list(by_id)

        top = {
            q: by_id[f"s0-q{i}"][1]["sources"][0]["section_heading"]
            for i, q in enumerate(QUERIES)
            for _, _, by_id in results[:1]
        }
        assert top["EIO-5 controller timeout"].endswith("I/O Timeouts (EIO-5)")
        assert top["PSU redundancy lost"].endswith("PSU Redundancy Alerts")

        for _ in range(100):  # all sessions unregistered after clients close
            if not app.state.connections.sessions:
                break
            await asyncio.sleep(0.02)
        assert app.state.connections.snapshot()["active"] == 0
        assert app.state.connections.total_accepted == sessions


async def test_ping_pong():
    async with live_server(ingest=False) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(json.dumps({"type": "ping", "ts": 42}))
            assert (await recv(ws))["type"] == "pong"


async def test_heartbeat_closes_silent_client():
    async with live_server(
        ingest=False, ws_heartbeat_interval_s=0.05, ws_heartbeat_timeout_s=0.3
    ) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            t0 = time.monotonic()
            with pytest.raises(ConnectionClosed) as info:
                while True:  # read server pings but never answer them
                    await recv(ws, timeout=3)
            assert info.value.rcvd.code == 4408
            assert 0.25 < time.monotonic() - t0 < 2.0


async def test_heartbeat_keeps_responsive_client_alive():
    async with live_server(
        ingest=False, ws_heartbeat_interval_s=0.05, ws_heartbeat_timeout_s=0.3
    ) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            pings = 0
            t_end = time.monotonic() + 1.0  # > 3x the timeout
            while time.monotonic() < t_end:
                ev = await recv(ws)
                if ev["type"] == "ping":
                    pings += 1
                    await ws.send(json.dumps({"type": "pong", "ts": ev["ts"]}))
            assert pings >= 5
            await ws.send(json.dumps({"type": "ping"}))
            while (await recv(ws))["type"] != "pong":
                pass


async def test_rate_limit_rejects_burst_beyond_bucket():
    async with live_server(rate_limit_capacity=3, rate_limit_refill_per_s=0.5) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            ids = [f"q{i}" for i in range(10)]
            for i in ids:
                await ws.send(json.dumps({"type": "query", "id": i, "text": "disk"}))
            _, by_id = await collect_responses(ws, ids)
            served = [i for i, ev in by_id.items() if ev[-1]["type"] == "response.end"]
            limited = [i for i, ev in by_id.items() if ev[-1]["type"] == "rate_limited"]
            assert served == ["q0", "q1", "q2"]
            assert len(limited) == 7
            assert all(by_id[i][-1]["retry_after_ms"] > 0 for i in limited)


async def test_ring_buffer_overflow_drops_oldest_queued_query():
    async with live_server(ws_buffer_size=2, rerank_enabled=True) as (ws_url, _, app):
        app.state.services.inference.latency_ms = 150  # make the worker slow
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(json.dumps({"type": "query", "id": "q0", "text": "disk"}))
            assert (await recv(ws))["type"] == "response.start"  # worker now busy on q0
            for i in range(1, 6):  # burst while busy: buffer holds 2, oldest evicted
                await ws.send(json.dumps({"type": "query", "id": f"q{i}", "text": "disk"}))
            frames: list[dict] = []
            while sum(f["type"] == "response.end" for f in frames) < 3:
                frames.append(await recv(ws))
            overflow = [f for f in frames if f["type"] == "buffer.overflow"]
            answered = [f["id"] for f in frames if f["type"] == "response.end"]
            assert [f["dropped_id"] for f in overflow] == ["q1", "q2", "q3"]
            assert answered == ["q0", "q4", "q5"]  # newest survive


async def test_binary_audio_frames_arrive_intact():
    async with live_server(ingest=False) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            chunks = [bytes([i % 256]) * 640 for i in range(25)]  # 25 x 20 ms PCM frames
            for c in chunks:
                await ws.send(c)
            await ws.send(json.dumps({"type": "audio.end", "id": "utt-1"}))
            ev = await recv(ws)
            assert ev["type"] == "audio.received" and ev["id"] == "utt-1"
            assert ev["chunks"] == 25
            assert ev["bytes"] == 25 * 640
            assert ev["sha256"] == hashlib.sha256(b"".join(chunks)).hexdigest()


async def test_session_capacity_rejects_extra_connections():
    async with live_server(ingest=False, ws_max_sessions=2) as (ws_url, _, app):
        async with connect(ws_url) as a, connect(ws_url) as b:
            await recv(a)
            await recv(b)
            async with connect(ws_url) as c:
                with pytest.raises(ConnectionClosed) as info:
                    await recv(c)
                assert info.value.rcvd.code == 1013
        assert app.state.connections.total_rejected == 1


async def test_bad_frames_get_errors_without_closing():
    async with live_server(ingest=False) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send("not json")
            assert (await recv(ws))["code"] == "bad_frame"
            await ws.send(json.dumps({"type": "query", "id": "x", "text": ""}))
            assert (await recv(ws))["code"] == "bad_query"
            await ws.send(json.dumps({"type": "query", "id": "y", "text": "a", "top_k": 0}))
            assert (await recv(ws))["code"] == "bad_query"
            await ws.send(json.dumps({"type": "teleport"}))
            assert (await recv(ws))["code"] == "unknown_type"
            await ws.send(json.dumps({"type": "ping"}))
            assert (await recv(ws))["type"] == "pong"


async def test_oversized_message_closes_with_1009():
    async with live_server(ingest=False, ws_max_message_bytes=1000) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(b"x" * 2000)
            with pytest.raises(ConnectionClosed) as info:
                await recv(ws)
            assert info.value.rcvd.code == 1009


async def test_traceparent_in_frame_propagates_to_server_spans(spans):
    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    traceparent = f"00-{trace_id}-00f067aa0ba902b7-01"
    async with live_server() as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(
                json.dumps(
                    {"type": "query", "id": "t1", "text": "disk", "traceparent": traceparent}
                )
            )
            _, by_id = await collect_responses(ws, ["t1"])
    assert by_id["t1"][0]["trace_id"] == trace_id
    answer = [s for s in spans.get_finished_spans() if s.name == "omniagent.answer"]
    assert len(answer) == 1
    assert format(answer[0].context.trace_id, "032x") == trace_id
    assert format(answer[0].parent.span_id, "016x") == "00f067aa0ba902b7"
    assert answer[0].links  # linked to its ws.session span


async def test_requests_without_traceparent_get_their_own_trace(spans):
    async with live_server() as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            for i in ("a", "b"):
                await ws.send(json.dumps({"type": "query", "id": i, "text": "disk"}))
            _, by_id = await collect_responses(ws, ["a", "b"])
    trace_a, trace_b = by_id["a"][0]["trace_id"], by_id["b"][0]["trace_id"]
    assert trace_a != trace_b
    session = next(s for s in spans.get_finished_spans() if s.name == "ws.session")
    answers = [s for s in spans.get_finished_spans() if s.name == "omniagent.answer"]
    assert all(s.parent is None for s in answers)  # roots, not children of the session
    assert all(s.links[0].context.span_id == session.context.span_id for s in answers)


async def test_slow_reranker_falls_back_to_rrf_order_instead_of_failing():
    async with live_server(rerank_enabled=True, rerank_timeout_s=0.05) as (ws_url, _, app):
        app.state.services.inference.latency_ms = 500  # far over the 50 ms budget
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(json.dumps({"type": "query", "id": "r", "text": "EIO-5 timeout"}))
            _, by_id = await collect_responses(ws, ["r"])
    events = by_id["r"]
    assert_response_integrity(events)
    sources = events[1]
    assert sources["reranked"] is False and sources["rerank_skipped"] == "timeout"
    assert all(s["rerank_score"] is None for s in sources["sources"])


async def test_fast_reranker_reorders_and_scores():
    async with live_server(rerank_enabled=True) as (ws_url, _, _):
        async with connect(ws_url) as ws:
            await recv(ws)
            await ws.send(json.dumps({"type": "query", "id": "r", "text": "PSU redundancy"}))
            _, by_id = await collect_responses(ws, ["r"])
    sources = by_id["r"][1]
    assert sources["reranked"] is True and sources["rerank_skipped"] is None
    scores = [s["rerank_score"] for s in sources["sources"]]
    assert scores == sorted(scores, reverse=True)


async def test_health_reports_event_loop_lag():
    import httpx

    async with live_server(ingest=False) as (_, http_url, _):
        await asyncio.sleep(0.2)
        async with httpx.AsyncClient() as http:
            lag = (await http.get(f"{http_url}/health")).json()["event_loop_lag"]
    assert lag["samples"] >= 5 and lag["max_ms"] >= 0
