"""Collaborative incident rooms: protocol, ordering, catch-up, agent and
vision broadcast, backpressure, and cross-replica fan-out through Redis."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.rooms.hub import RoomSession
from backend.rooms.store import seq_of, with_seq
from tests.conftest import make_settings


def _jpeg() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((64, 64, 3), 128, np.uint8))
    assert ok
    return buf.tobytes()


def recv_until(ws, kind: str, limit: int = 200) -> list[dict]:
    """Frames up to and including the first of type `kind`."""
    out = []
    for _ in range(limit):
        ev = ws.receive_json()
        out.append(ev)
        if ev["type"] == kind:
            return out
    raise AssertionError(f"no {kind} in {[e['type'] for e in out]}")


def recv_all(ws, *kinds: str, limit: int = 200) -> dict[str, dict]:
    """Read until one frame of each kind has arrived (in any order)."""
    got: dict[str, dict] = {}
    for _ in range(limit):
        ev = ws.receive_json()
        if ev["type"] in kinds and ev["type"] not in got:
            got[ev["type"]] = ev
            if len(got) == len(kinds):
                return got
    raise AssertionError(f"missing {set(kinds) - set(got)}")


def seqs(events: list[dict]) -> list[int]:
    return [e["seq"] for e in events if "seq" in e]


@pytest.fixture
def client():
    with TestClient(create_app(make_settings(vision_backend="mock"))) as c:
        yield c


def new_room(c) -> str:
    r = c.post("/rooms", json={"title": "storage-07 EIO"})
    assert r.status_code == 201
    return r.json()["room_id"]


def test_seq_helpers_keep_exact_numbers():
    raw = with_seq(42, '{"type":"message","client_ts":1795287528836.1234}')
    assert raw == '{"seq":42,"type":"message","client_ts":1795287528836.1234}'
    assert seq_of(raw) == 42 and seq_of('{"type":"agent.token"}') is None


def test_room_lifecycle_broadcast_ack_and_idempotency(client):
    room = new_room(client)
    assert client.get("/rooms/nope-404").status_code == 404
    with client.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a:
        snap = a.receive_json()
        assert snap["type"] == "room.snapshot" and snap["meta"]["title"] == "storage-07 EIO"
        recv_until(a, "member.joined")
        with client.websocket_connect(f"/ws/rooms/{room}?name=Bo") as b:
            assert [m["name"] for m in b.receive_json()["members"]] == ["Ana", "Bo"]
            recv_until(b, "member.joined")
            recv_until(a, "member.joined")  # Bo's join, seen by Ana

            a.send_json(
                {
                    "type": "message",
                    "text": "disk 3 is throwing EIO",
                    "client_msg_id": "m1",
                    "client_ts": 1.5,
                }
            )
            mine = recv_all(a, "message", "ack")  # the direct ack can overtake the broadcast copy
            got_b = recv_until(b, "message")[-1]
            assert got_b["text"] == "disk 3 is throwing EIO" and got_b["name"] == "Ana"
            assert got_b["client_ts"] == 1.5
            assert mine["ack"]["acked_seq"] == got_b["seq"] == mine["message"]["seq"]

            a.send_json(
                {"type": "message", "text": "disk 3 is throwing EIO", "client_msg_id": "m1"}
            )
            assert (
                recv_until(a, "ack")[-1]["duplicate"] is True
            )  # resend after a reconnect: not re-broadcast

            b.send_json({"type": "typing", "active": True})
            typing = recv_until(a, "presence.typing")[-1]
            assert typing["name"] == "Bo" and "seq" not in typing  # ephemeral

    snap = client.get(f"/rooms/{room}").json()
    types = [e["type"] for e in snap["events"]]
    assert types.count("message") == 1 and types[-1] == "member.left"
    assert seqs(snap["events"]) == list(range(1, len(snap["events"]) + 1))  # gap-free, ordered


def test_reconnect_with_last_seq_replays_exactly_the_missed_events(client):
    room = new_room(client)
    with client.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a:
        a.receive_json()
        joined = recv_until(a, "member.joined")[-1]["seq"]
    last_seen = joined  # Ana disconnects after seeing her own join
    with client.websocket_connect(f"/ws/rooms/{room}?name=Bo") as b:
        b.receive_json()
        for i in range(5):
            b.send_json({"type": "message", "text": f"update {i}", "client_msg_id": f"b{i}"})
            recv_all(b, "ack", "message")
    with client.websocket_connect(f"/ws/rooms/{room}?name=Ana&last_seq={last_seen}") as a:
        snap = a.receive_json()
        assert snap["resync"] is False
        replay = [a.receive_json() for _ in range(snap["replay"])]
        # Missed: Ana left, Bo joined, 5 messages, Bo left. Nothing she had seen.
        assert seqs(replay) == list(range(last_seen + 1, last_seen + 1 + len(replay)))
        assert [e["text"] for e in replay if e["type"] == "message"] == [
            f"update {i}" for i in range(5)
        ]
        live = recv_until(a, "member.joined")[-1]  # her own new join continues the sequence
        assert live["seq"] == replay[-1]["seq"] + 1


def test_pins_are_shared_state(client):
    room = new_room(client)
    with client.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a:
        a.receive_json()
        a.send_json(
            {
                "type": "pin",
                "kind": "log",
                "title": "storage-07 kernel log",
                "snippet": "EIO on sdc",
            }
        )
        pin = recv_until(a, "pin.added")[-1]["pin"]
        assert client.get(f"/rooms/{room}").json()["pins"][0]["title"] == "storage-07 kernel log"
        a.send_json({"type": "pin", "kind": "bogus", "title": "x"})
        assert recv_until(a, "error")[-1]["error"].startswith("pin needs")
        a.send_json({"type": "unpin", "pin_id": pin["pin_id"]})
        recv_until(a, "pin.removed")
        assert client.get(f"/rooms/{room}").json()["pins"] == []


def test_agent_turn_streams_to_every_participant(client):
    room = new_room(client)
    with (
        client.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a,
        client.websocket_connect(f"/ws/rooms/{room}?name=Bo") as b,
    ):
        a.receive_json()
        b.receive_json()
        a.send_json({"type": "ask", "question": "what's 6*7?"})
        for ws in (a, b):
            events = recv_until(ws, "agent.answer")
            start = next(e for e in events if e["type"] == "agent.start")
            tokens = [e["text"] for e in events if e["type"] == "agent.token"]
            answer = events[-1]
            assert start["name"] == "Ana" and answer["turn_id"] == start["turn_id"]
            assert tokens and "".join(tokens) == answer["text"] and "42" in answer["text"]


def test_frame_becomes_the_rooms_visual_context(client):
    room = new_room(client)
    with (
        client.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a,
        client.websocket_connect(f"/ws/rooms/{room}?name=Bo") as b,
    ):
        a.receive_json()
        b.receive_json()
        a.send_bytes(_jpeg())
        vc = recv_until(b, "visual_context")[-1]
        assert vc["name"] == "Ana" and len(vc["labels"]) == 5
        b.send_json({"type": "ask", "question": "what is on the camera?"})
        answer = recv_until(a, "agent.answer")[-1]
        assert answer["visual_context"] and "frame #" in answer["visual_context"]


def test_unknown_room_is_rejected(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/rooms/doesnotexist") as ws:
            ws.receive_json()
    assert exc.value.code == 4404


def test_slow_consumer_overflows_without_blocking_and_is_marked():
    s = RoomSession(room="r", name="slow", ws=None, queue_size=4)  # type: ignore[arg-type]
    for i in range(1, 50):
        s.offer(with_seq(i, '{"type":"message"}'))  # never blocks, never raises
    assert s.slow
    assert asyncio.run(_drain(s)) == "stopped"  # the sender returns so the session can close (4409)


async def _drain(s: RoomSession) -> str:
    sent = []

    class WS:
        async def send_text(self, raw):
            sent.append(raw)

    s.ws = WS()
    await asyncio.wait_for(s.run_sender(), timeout=1)
    return "stopped"


# --- two replicas sharing a real Redis ----------------------------------------


@pytest.fixture(scope="module")
def redis_url():
    exe = shutil.which("redis-server")
    if not exe:
        pytest.skip("redis-server not installed")
    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    proc = subprocess.Popen(
        [exe, "--port", str(port), "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.1).close()
            break
        except OSError:
            time.sleep(0.1)
    yield f"redis://127.0.0.1:{port}/0"
    proc.terminate()
    proc.wait(timeout=5)


def test_two_replicas_fan_out_through_redis_in_one_order(redis_url):
    settings = make_settings(redis_url=redis_url)
    with TestClient(create_app(settings)) as r1, TestClient(create_app(settings)) as r2:
        room = new_room(r1)
        assert r2.get(f"/rooms/{room}").status_code == 200  # state lives in Redis
        with (
            r1.websocket_connect(f"/ws/rooms/{room}?name=Ana") as a,
            r2.websocket_connect(f"/ws/rooms/{room}?name=Bo") as b,
        ):
            a.receive_json()
            b.receive_json()
            for i in range(10):
                (a if i % 2 else b).send_json(
                    {"type": "message", "text": f"m{i}", "client_msg_id": f"x{i}"}
                )
            got = {}
            for name, ws in (("a", a), ("b", b)):
                msgs = []
                while len(msgs) < 10:
                    ev = ws.receive_json()
                    if ev["type"] == "message":
                        msgs.append(ev)
                got[name] = msgs
            # Both replicas delivered the same events in the same Redis-assigned order.
            assert [m["seq"] for m in got["a"]] == [m["seq"] for m in got["b"]]
            assert sorted(m["text"] for m in got["a"]) == [f"m{i}" for i in range(10)]
            assert [m["seq"] for m in got["a"]] == sorted(m["seq"] for m in got["a"])
