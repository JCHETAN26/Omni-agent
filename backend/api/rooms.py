"""Collaborative incident rooms: REST + WebSocket.

    POST /rooms                      {"title"?}            -> {"room_id", "title", "created_at"}
    GET  /rooms/{room_id}                           -> snapshot (members, pins, recent events)
    GET  /rooms/{room_id}/events?after=N&limit=M           -> sequenced events after N
    WS   /ws/rooms/{room_id}?name=<display name>&last_seq=N

WebSocket protocol (JSON text frames; binary frames are camera/screen images):

  client -> server
    {"type": "message", "text", "client_msg_id"?, "client_ts"?}   idempotent per client_msg_id
    {"type": "typing", "active": bool}                            ephemeral
    {"type": "focus", "pane": str}                                ephemeral + presence
    {"type": "pin", "kind": "search"|"log"|"citation"|"note", "title", "ref"?, "snippet"?}
    {"type": "unpin", "pin_id"}
    {"type": "ask", "question"}                                   shared agent turn
    {"type": "ping", "ts"?}                                       heartbeat -> pong
    <binary JPEG/PNG>                                             -> visual_context for the room

  server -> client
    room.snapshot (first frame: members, pins, head_seq, replay count, resync flag)
    then the sequenced events after `last_seq` (or the latest 50), then live events.
    Sequenced (stored, "seq" strictly +1 per room): member.joined, member.left,
      message, pin.added, pin.removed, agent.start, agent.answer, agent.error,
      visual_context
    Ephemeral (no seq): presence.typing, presence.focus, agent.token, pong,
      rate_limited, error, and ack ({"client_msg_id", "acked_seq"} to the sender
      only; "seq" is reserved for the room's order)

Close codes: 4404 unknown room, 4408 heartbeat timeout, 4409 too slow (reconnect
with last_seq to resync), 1013 at capacity.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.core import metrics
from backend.rooms.hub import RoomHub, RoomSession
from backend.rooms.store import RoomStore, now_ms
from backend.vision import context as vision_context
from model_server.vision_client import VisionUnavailable

router = APIRouter()

ROOM_RE = re.compile(r"^[a-z0-9][a-z0-9-]{3,39}$")
CLOSE_UNKNOWN_ROOM, CLOSE_HEARTBEAT, CLOSE_SLOW, CLOSE_FULL = 4404, 4408, 4409, 1013
PIN_KINDS = {"search", "log", "citation", "note"}
HISTORY_DEFAULT, HISTORY_MAX = 50, 1000


def room_thread(room: str) -> str:
    """The agent conversation (memory + visual context) shared by a room."""
    return f"room-{room}"[:64]


class CreateRoom(BaseModel):
    title: str = Field(default="", max_length=120)


def _store(request_or_ws) -> RoomStore:
    return request_or_ws.app.state.rooms.store


@router.post("/rooms", status_code=201)
async def create_room(body: CreateRoom, request: Request) -> dict:
    room = uuid.uuid4().hex[:10]
    title = body.title.strip() or f"Incident {room[:6]}"
    meta = {"title": title, "created_at": now_ms()}
    await _store(request).create(room, meta)
    return {"room_id": room, **meta}


async def _members(store: RoomStore, room: str, ttl_s: float) -> list[dict]:
    cutoff = time.time() - ttl_s
    return [
        {"session": sid, **p}
        for sid, p in sorted(
            (await store.presence(room)).items(), key=lambda kv: kv[1].get("joined_at", 0)
        )
        if p.get("last_seen", 0) >= cutoff
    ]


@router.get("/rooms/{room_id}")
async def room_snapshot(room_id: str, request: Request) -> dict:
    store, s = _store(request), request.app.state.services.settings
    meta = await store.meta(room_id) if ROOM_RE.match(room_id) else None
    if meta is None:
        raise HTTPException(404, "room not found")
    raw, _ = await store.events_after(room_id, 0, HISTORY_DEFAULT)
    return {
        "room_id": room_id,
        "meta": meta,
        "head_seq": await store.last_seq(room_id),
        "members": await _members(store, room_id, s.rooms_presence_ttl_s),
        "pins": list((await store.pins(room_id)).values()),
        "events": [json.loads(e) for e in raw],
    }


@router.get("/rooms/{room_id}/events")
async def room_events(
    room_id: str,
    request: Request,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=HISTORY_MAX),
) -> dict:
    store = _store(request)
    if not ROOM_RE.match(room_id) or await store.meta(room_id) is None:
        raise HTTPException(404, "room not found")
    raw, trimmed = await store.events_after(room_id, after, limit)
    return {
        "events": [json.loads(e) for e in raw],
        "trimmed": trimmed,
        "head_seq": await store.last_seq(room_id),
    }


class _Bucket:
    """Token bucket for one session's inbound messages."""

    def __init__(self, rate: float):
        self.rate, self.cap = rate, 2 * rate
        self.tokens, self.at = self.cap, time.monotonic()

    def take(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.cap, self.tokens + (now - self.at) * self.rate)
        self.at = now
        if self.tokens < 1:
            return False
        self.tokens -= 1
        return True


@router.websocket("/ws/rooms/{room_id}")
async def ws_room(ws: WebSocket, room_id: str, name: str = "", last_seq: int | None = None) -> None:
    rooms = ws.app.state.rooms
    store: RoomStore = rooms.store
    hub: RoomHub = rooms.hub
    services = ws.app.state.services
    s = services.settings
    await ws.accept()
    if not ROOM_RE.match(room_id) or await store.meta(room_id) is None:
        await ws.close(code=CLOSE_UNKNOWN_ROOM, reason="unknown room")
        return
    if hub.local_sessions >= s.rooms_max_sessions:
        await ws.close(code=CLOSE_FULL, reason="server at room capacity")
        return
    display = re.sub(r"\s+", " ", name).strip()[:40] or "guest"
    sess = RoomSession(room=room_id, name=display, ws=ws, queue_size=s.rooms_queue_size)

    # Subscribe before reading history: live events published meanwhile are
    # queued, and the sender drops the ones the replay already covered.
    await hub.join(sess)
    sender: asyncio.Task | None = None
    try:
        after = last_seq if last_seq is not None else None
        if after is None:
            head = await store.last_seq(room_id)
            raw, trimmed = await store.events_after(
                room_id, max(head - HISTORY_DEFAULT, 0), HISTORY_DEFAULT
            )
            resync = False
        else:
            raw, trimmed = await store.events_after(room_id, after, HISTORY_MAX)
            resync = trimmed  # the client missed more than the room keeps: start over
        presence = {
            "name": display,
            "focus": "timeline",
            "joined_at": time.time(),
            "last_seen": time.time(),
        }
        await store.set_presence(room_id, sess.id, presence)
        await ws.send_text(
            json.dumps(
                {
                    "type": "room.snapshot",
                    "room_id": room_id,
                    "session": sess.id,
                    "name": display,
                    "meta": await store.meta(room_id),
                    "members": await _members(store, room_id, s.rooms_presence_ttl_s),
                    "pins": list((await store.pins(room_id)).values()),
                    "replay": len(raw),
                    "resync": resync,
                    "heartbeat_interval_s": s.ws_heartbeat_interval_s,
                }
            )
        )
        for e in raw:
            await ws.send_text(e)
        if raw:
            sess.sent_seq = json.loads(raw[-1])["seq"]
        elif after is not None:
            sess.sent_seq = after
        sender = asyncio.create_task(sess.run_sender(), name=f"room-sender-{sess.id}")
        await _append(
            store,
            room_id,
            {"type": "member.joined", "session": sess.id, "name": display, "at": now_ms()},
        )
        await _receive(ws, sess, store, rooms, services, sender)
    except WebSocketDisconnect:
        pass
    finally:
        await hub.leave(sess)
        if sender:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await sender
        with contextlib.suppress(Exception):
            await store.remove_presence(room_id, sess.id)
            await _append(
                store,
                room_id,
                {"type": "member.left", "session": sess.id, "name": display, "at": now_ms()},
            )


async def _append(store: RoomStore, room: str, event: dict) -> int:
    seq, _ = await store.append(room, event)
    metrics.ROOM_EVENTS.labels(event["type"]).inc()
    return seq


async def _receive(
    ws: WebSocket, sess: RoomSession, store: RoomStore, rooms, services, sender
) -> None:
    s = services.settings
    bucket = _Bucket(s.rooms_msgs_per_s)
    room = sess.room
    last_presence = 0.0
    while True:
        recv = asyncio.create_task(ws.receive())
        done, _ = await asyncio.wait(
            {recv, sender}, timeout=s.ws_heartbeat_timeout_s, return_when=asyncio.FIRST_COMPLETED
        )
        if sender in done:  # the sender stopped: slow consumer (or the socket died)
            recv.cancel()
            if sess.slow:
                await ws.close(
                    code=CLOSE_SLOW, reason="too slow: reconnect with last_seq to resync"
                )
            return
        if not done:
            recv.cancel()
            await ws.close(code=CLOSE_HEARTBEAT, reason="heartbeat timeout")
            return
        msg = recv.result()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        now = time.time()
        if now - last_presence > 5:  # refresh last_seen at most every 5 s
            last_presence = now
            p = (await store.presence(room)).get(sess.id) or {"name": sess.name, "joined_at": now}
            p["last_seen"] = now
            await store.set_presence(room, sess.id, p)
        if msg.get("bytes") is not None:
            await _frame(ws, sess, store, rooms, services, msg["bytes"])
            continue
        try:
            data = json.loads(msg.get("text") or "")
            kind = data.get("type")
        except (ValueError, AttributeError):
            await ws.send_text(json.dumps({"type": "error", "error": "invalid JSON"}))
            continue
        if kind == "ping":
            await ws.send_text(json.dumps({"type": "pong", "ts": data.get("ts")}))
            continue
        if not bucket.take():
            await ws.send_text(
                json.dumps({"type": "rate_limited", "retry_after_s": 1 / bucket.rate})
            )
            continue
        await _handle(ws, sess, store, rooms, services, kind, data)


async def _handle(
    ws, sess: RoomSession, store: RoomStore, rooms, services, kind, data: dict
) -> None:
    room, who = sess.room, {"session": sess.id, "name": sess.name}
    err = lambda e: ws.send_text(json.dumps({"type": "error", "error": e}))  # noqa: E731
    if kind == "message":
        text = str(data.get("text", "")).strip()
        if not text or len(text) > 4000:
            return await err("message text must be 1-4000 characters")
        cid = str(data.get("client_msg_id", ""))[:64]
        if cid and not await store.claim_once(room, f"msg:{cid}"):
            return await ws.send_text(
                json.dumps({"type": "ack", "client_msg_id": cid, "duplicate": True})
            )
        ev = {"type": "message", **who, "text": text, "client_msg_id": cid, "at": now_ms()}
        if isinstance(data.get("client_ts"), int | float):
            ev["client_ts"] = data["client_ts"]
        seq = await _append(store, room, ev)
        await ws.send_text(json.dumps({"type": "ack", "client_msg_id": cid, "acked_seq": seq}))
    elif kind == "typing":
        await store.publish_ephemeral(
            room, {"type": "presence.typing", **who, "active": bool(data.get("active"))}
        )
    elif kind == "focus":
        pane = str(data.get("pane", ""))[:40]
        p = (await store.presence(room)).get(sess.id) or {"name": sess.name}
        p.update(focus=pane, last_seen=time.time())
        await store.set_presence(room, sess.id, p)
        await store.publish_ephemeral(room, {"type": "presence.focus", **who, "pane": pane})
    elif kind == "pin":
        if data.get("kind") not in PIN_KINDS or not str(data.get("title", "")).strip():
            return await err(f"pin needs a kind in {sorted(PIN_KINDS)} and a title")
        pin = {
            "pin_id": uuid.uuid4().hex[:12],
            "kind": data["kind"],
            "title": str(data["title"])[:200],
            "ref": str(data.get("ref", ""))[:500],
            "snippet": str(data.get("snippet", ""))[:1000],
            "by": sess.name,
            "at": now_ms(),
        }
        await store.add_pin(room, pin["pin_id"], pin)
        await _append(store, room, {"type": "pin.added", **who, "pin": pin})
    elif kind == "unpin":
        pin_id = str(data.get("pin_id", ""))
        if await store.remove_pin(room, pin_id):
            await _append(store, room, {"type": "pin.removed", **who, "pin_id": pin_id})
    elif kind == "ask":
        question = str(data.get("question", "")).strip()
        if not question or len(question) > 2000:
            return await err("question must be 1-2000 characters")
        turn = uuid.uuid4().hex[:12]
        await _append(
            store,
            room,
            {"type": "agent.start", **who, "turn_id": turn, "question": question, "at": now_ms()},
        )
        task = asyncio.create_task(
            _agent_turn(rooms, ws.app.state.agent, room, turn, question, who)
        )
        rooms.tasks.add(task)
        task.add_done_callback(rooms.tasks.discard)
    else:
        await err(f"unknown message type {kind!r}")


async def _agent_turn(rooms, agent, room: str, turn: str, question: str, who: dict) -> None:
    """Run one agent turn for the room: tokens stream to everyone (ephemeral),
    the full answer is stored as one sequenced event for late joiners."""
    store: RoomStore = rooms.store
    tokens: list[str] = []
    result: dict = {"sources": [], "visual_context": None}

    async def emit(ev: dict) -> None:
        t = ev.get("type")
        if t == "response.token":
            tokens.append(ev["text"])
            await store.publish_ephemeral(
                room,
                {"type": "agent.token", "turn_id": turn, "index": ev["index"], "text": ev["text"]},
            )
        elif t == "response.sources":
            result["sources"] = ev.get("sources", [])
        elif t == "agent.step" and ev.get("node") == "visual_context":
            result["visual_context"] = ev.get("observation")
        elif t == "response.end":
            result.update(outcome=ev.get("outcome"), tool=ev.get("tool"))
        elif t == "response.error":
            result["error"] = ev.get("error", "agent failed")

    try:
        await agent.run_turn(room_thread(room), question, emit, f"room-{turn}", None)
    except Exception as exc:  # noqa: BLE001 - the room must hear about it
        result["error"] = str(exc)
    if "error" in result:
        await _append(
            store, room, {"type": "agent.error", "turn_id": turn, "error": result["error"], **who}
        )
        return
    await _append(
        store,
        room,
        {
            "type": "agent.answer",
            "turn_id": turn,
            "question": question,
            "asked_by": who["name"],
            "text": "".join(tokens),
            "outcome": result.get("outcome"),
            "tool": result.get("tool"),
            "sources": result["sources"][:5],
            "visual_context": result["visual_context"],
            "at": now_ms(),
        },
    )


async def _frame(ws, sess: RoomSession, store: RoomStore, rooms, services, data: bytes) -> None:
    """A participant's camera/screen frame becomes the room's visual context
    (seen by the room's agent turns) and is announced to everyone."""
    s = services.settings
    if services.vision is None or services.preprocessor is None:
        return await ws.send_text(json.dumps({"type": "error", "error": "vision is disabled"}))
    if len(data) > s.vision_max_frame_bytes:
        return await ws.send_text(json.dumps({"type": "error", "error": "frame too large"}))
    try:
        pixels = await asyncio.to_thread(services.preprocessor.encoded, data)
        labels = await services.vision.classify(pixels)
    except (ValueError, VisionUnavailable) as exc:
        return await ws.send_text(json.dumps({"type": "error", "error": f"frame: {exc}"}))
    rooms.frames += 1
    await vision_context.record(
        services.cache, room_thread(sess.room), rooms.frames, labels, s.vision_context_ttl_s
    )
    await _append(
        store,
        sess.room,
        {
            "type": "visual_context",
            "session": sess.id,
            "name": sess.name,
            "labels": [{"label": lb.label, "score": lb.score} for lb in labels[:5]],
            "at": now_ms(),
        },
    )


class Rooms:
    """App-wide room state: the store, this replica's hub, background tasks."""

    def __init__(self, store: RoomStore):
        self.store = store
        self.hub = RoomHub(store)
        self.tasks: set[asyncio.Task] = set()
        self.frames = 0
        self._sweeper: asyncio.Task | None = None

    def start(self, presence_ttl_s: float) -> None:
        self.hub.start()
        self._sweeper = asyncio.create_task(self.hub.sweep(presence_ttl_s, presence_ttl_s / 3))

    async def stop(self) -> None:
        for t in [*self.tasks, self._sweeper]:
            if t:
                t.cancel()
        await self.hub.stop()
        await self.store.close()
