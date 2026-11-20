"""WebSocket gateway: /ws/stream

Protocol (JSON text frames unless noted):

  client -> server
    {"type": "query", "id": "q1", "text": "...", "top_k": 5, "traceparent"?: "00-..."}
    {"type": "ping", "ts": 123}             -> answered immediately with pong
    {"type": "pong", "ts": 123}             -> reply to a server heartbeat
    <binary frame>                          -> one audio chunk for the current utterance
    {"type": "audio.end", "id": "a1"}       -> closes the utterance

  server -> client (every frame has a per-session, strictly increasing "seq")
    session.created, pong, ping (heartbeat), rate_limited, buffer.overflow,
    error, audio.received, response.start / .sources / .token / .end / .error

Per connection, three tasks run until one finishes:
  reader    receive -> validate -> rate-limit (token bucket) -> ring buffer
  worker    ring buffer -> handle -> stream events back (in arrival order)
  heartbeat send ping every interval; close 4408 if the client goes silent

Close codes: 1009 message too big, 1013 server at session capacity,
4408 heartbeat timeout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import anyio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from opentelemetry import context as otel_context
from opentelemetry import propagate
from opentelemetry.trace import SpanKind
from starlette.websockets import WebSocketState

from backend.api.ring_buffer import RingBuffer
from backend.api.streaming import stream_answer
from backend.core import metrics
from backend.core.services import Services
from backend.core.telemetry import get_tracer

log = logging.getLogger(__name__)
router = APIRouter()

CLOSE_TOO_BIG = 1009
CLOSE_TRY_AGAIN = 1013
CLOSE_HEARTBEAT_TIMEOUT = 4408


@dataclass
class QueryItem:
    id: str
    text: str
    top_k: int
    traceparent: str | None


@dataclass
class AudioChunk:
    data: bytes


@dataclass
class AudioEnd:
    id: str


@dataclass
class Session:
    id: str
    ws: WebSocket
    buffer: RingBuffer
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    seq: int = 0
    messages_in: int = 0
    messages_out: int = 0
    rate_limited: int = 0
    span: object = None
    services: Services | None = None
    handle_query: object = None
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _audio_chunks: int = 0
    _audio_bytes: int = 0
    _audio_hash: object = field(default_factory=hashlib.sha256)

    async def send(self, event: dict) -> None:
        async with self._send_lock:  # starlette sends are not safe to interleave
            self.seq += 1
            await self.ws.send_text(json.dumps({**event, "seq": self.seq}, separators=(",", ":")))
            self.messages_out += 1
            metrics.WS_FRAMES.labels("out").inc()


class ConnectionManager:
    def __init__(self, max_sessions: int):
        self.max_sessions = max_sessions
        self.sessions: dict[str, Session] = {}
        self.total_accepted = 0
        self.total_rejected = 0

    @property
    def full(self) -> bool:
        return len(self.sessions) >= self.max_sessions

    def register(self, ws: WebSocket, buffer_size: int) -> Session:
        session = Session(id=uuid.uuid4().hex, ws=ws, buffer=RingBuffer(buffer_size))
        self.sessions[session.id] = session
        self.total_accepted += 1
        return session

    def unregister(self, session: Session) -> None:
        self.sessions.pop(session.id, None)

    def snapshot(self) -> dict:
        return {
            "active": len(self.sessions),
            "max": self.max_sessions,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
        }


QueryHandler = Callable[["Session", "QueryItem"], Awaitable[None]]


async def _plain_query(session: Session, item: QueryItem) -> None:
    await stream_answer(
        session.services,
        item.text,
        item.top_k,
        item.id,
        session.send,
        item.traceparent,
        session.span,
    )


@router.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await serve_session(ws, _plain_query)


async def serve_session(
    ws: WebSocket,
    handle_query: QueryHandler,
    hello: dict | None = None,
    on_open: Callable[[Session], Awaitable[None]] | None = None,
) -> None:
    """Run one WebSocket session: accept, register, then reader/worker/heartbeat.

    `handle_query` answers each query (in arrival order). `hello` adds fields to
    `session.created`; `on_open` runs right after it (for example to replay history).
    """
    services: Services = ws.app.state.services
    manager: ConnectionManager = ws.app.state.connections
    s = services.settings

    await ws.accept()
    if manager.full:
        manager.total_rejected += 1
        await ws.close(code=CLOSE_TRY_AGAIN, reason="server at session capacity")
        return

    session = manager.register(ws, s.ws_buffer_size)
    endpoint = ws.url.path
    metrics.WS_SESSIONS.labels(endpoint).inc()
    session.services = services
    session.handle_query = handle_query
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "ws.session",
        context=propagate.extract(ws.headers, context=otel_context.get_current()),
        kind=SpanKind.SERVER,
        attributes={"http.route": endpoint, "ws.session_id": session.id},
    ) as span:
        session.span = span
        log.info("ws session opened", extra={"session_id": session.id})
        try:
            await session.send(
                {
                    "type": "session.created",
                    "session_id": session.id,
                    "heartbeat_interval_s": s.ws_heartbeat_interval_s,
                    "heartbeat_timeout_s": s.ws_heartbeat_timeout_s,
                    "rate_limit": {
                        "capacity": s.rate_limit_capacity,
                        "refill_per_s": s.rate_limit_refill_per_s,
                    },
                    "buffer_size": s.ws_buffer_size,
                    **(hello or {}),
                }
            )
            if on_open is not None:
                await on_open(session)
            # anyio task group: integrates with Starlette's cancel scopes, and the
            # first task to finish (normally the reader, on disconnect) stops the rest.
            async with anyio.create_task_group() as tg:

                async def run(fn) -> None:
                    try:
                        await fn(session, services)
                    except Exception as exc:  # noqa: BLE001
                        if not _is_disconnect(exc):
                            log.error(
                                "ws task %s failed",
                                fn.__name__,
                                exc_info=exc,
                                extra={"session_id": session.id},
                            )
                    finally:
                        tg.cancel_scope.cancel()

                tg.start_soon(run, _reader)
                tg.start_soon(run, _worker)
                tg.start_soon(run, _heartbeat)
        except Exception as exc:  # noqa: BLE001
            if not _is_disconnect(exc):
                log.exception("ws session error", extra={"session_id": session.id})
        finally:
            manager.unregister(session)
            metrics.WS_SESSIONS.labels(endpoint).dec()
            span.set_attributes(
                {
                    "ws.messages_in": session.messages_in,
                    "ws.messages_out": session.messages_out,
                    "ws.rate_limited": session.rate_limited,
                    "ws.dropped": session.buffer.dropped,
                }
            )
            if (
                ws.client_state == WebSocketState.CONNECTED
                and ws.application_state == WebSocketState.CONNECTED
            ):
                with anyio.CancelScope(shield=True):
                    try:
                        await ws.close()
                    except Exception:  # noqa: BLE001 - peer already gone
                        pass
            log.info(
                "ws session closed",
                extra={
                    "session_id": session.id,
                    "messages_in": session.messages_in,
                    "messages_out": session.messages_out,
                    "duration_s": round(time.monotonic() - session.created_at, 3),
                },
            )


def _is_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, WebSocketDisconnect | ConnectionError | asyncio.CancelledError) or (
        type(exc).__name__ in {"ClientDisconnected", "ConnectionClosed", "ConnectionClosedOK"}
    )


async def _reader(session: Session, services: Services) -> None:
    s = services.settings
    limiter = services.rate_limiter
    while True:
        msg = await session.ws.receive()
        if msg["type"] == "websocket.disconnect":
            return
        session.last_seen = time.monotonic()
        session.messages_in += 1
        metrics.WS_FRAMES.labels("in").inc()
        raw_bytes, raw_text = msg.get("bytes"), msg.get("text")
        size = len(raw_bytes) if raw_bytes is not None else len(raw_text or "")
        if size > s.ws_max_message_bytes:
            await session.ws.close(code=CLOSE_TOO_BIG, reason="message too big")
            return

        if raw_bytes is not None:
            item, ref, cost = AudioChunk(raw_bytes), None, s.rate_limit_audio_cost
        else:
            try:
                data = json.loads(raw_text or "")
                if not isinstance(data, dict):
                    raise ValueError("frame must be a JSON object")
            except ValueError as exc:
                await session.send({"type": "error", "code": "bad_frame", "error": str(exc)})
                continue
            kind = data.get("type")
            if kind == "ping":
                await session.send({"type": "pong", "ts": data.get("ts")})
                continue
            if kind == "pong":
                continue
            ref = data.get("id")
            if kind == "query":
                text = data.get("text")
                top_k = data.get("top_k", 5)
                if not isinstance(text, str) or not text.strip():
                    await session.send(
                        {
                            "type": "error",
                            "id": ref,
                            "code": "bad_query",
                            "error": "query.text must be a non-empty string",
                        }
                    )
                    continue
                if not isinstance(top_k, int) or not 1 <= top_k <= 50:
                    await session.send(
                        {
                            "type": "error",
                            "id": ref,
                            "code": "bad_query",
                            "error": "top_k must be an integer in [1, 50]",
                        }
                    )
                    continue
                item = QueryItem(ref or uuid.uuid4().hex, text, top_k, data.get("traceparent"))
                cost = 1.0
            elif kind == "audio.end":
                item, cost = AudioEnd(ref or uuid.uuid4().hex), 0.0
            else:
                await session.send(
                    {
                        "type": "error",
                        "id": ref,
                        "code": "unknown_type",
                        "error": f"unknown message type {kind!r}",
                    }
                )
                continue

        if cost > 0:
            decision = await limiter.acquire(f"ws:{session.id}", cost)
            if not decision.allowed:
                session.rate_limited += 1
                metrics.WS_RATE_LIMITED.inc()
                await session.send(
                    {
                        "type": "rate_limited",
                        "id": ref,
                        "retry_after_ms": round(decision.retry_after_s * 1000, 1),
                    }
                )
                continue

        evicted = session.buffer.push(item)
        if evicted is not None:
            metrics.WS_BUFFER_DROPPED.inc()
            await session.send(
                {
                    "type": "buffer.overflow",
                    "dropped": type(evicted).__name__,
                    "dropped_id": getattr(evicted, "id", None),
                    "total_dropped": session.buffer.dropped,
                }
            )


async def _worker(session: Session, services: Services) -> None:
    while True:
        item = await session.buffer.get()
        if isinstance(item, AudioChunk):
            session._audio_chunks += 1
            session._audio_bytes += len(item.data)
            session._audio_hash.update(item.data)
        elif isinstance(item, AudioEnd):
            # Placeholder until speech-to-text lands: acknowledge what arrived so
            # clients can verify nothing was lost in transit.
            await session.send(
                {
                    "type": "audio.received",
                    "id": item.id,
                    "chunks": session._audio_chunks,
                    "bytes": session._audio_bytes,
                    "sha256": session._audio_hash.hexdigest(),
                }
            )
            session._audio_chunks = session._audio_bytes = 0
            session._audio_hash = hashlib.sha256()
        elif isinstance(item, QueryItem):
            await session.handle_query(session, item)


async def _heartbeat(session: Session, services: Services) -> None:
    s = services.settings
    while True:
        await asyncio.sleep(s.ws_heartbeat_interval_s)
        silent_for = time.monotonic() - session.last_seen
        if silent_for > s.ws_heartbeat_timeout_s:
            log.info(
                "ws heartbeat timeout",
                extra={"session_id": session.id, "silent_s": round(silent_for, 2)},
            )
            await session.ws.close(code=CLOSE_HEARTBEAT_TIMEOUT, reason="heartbeat timeout")
            return
        await session.send({"type": "ping", "ts": time.time()})
