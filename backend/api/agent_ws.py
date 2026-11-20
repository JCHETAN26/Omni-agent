"""Agent endpoints: /ws/agent (WebSocket) and /sse/agent (SSE).

Sessions reuse the Phase 2 gateway (heartbeats, ring buffer, rate limiting,
per-session seq numbers). Each connection is bound to a conversation
`thread_id`:

    ws://host/ws/agent                     -> new conversation (thread_id in session.created)
    ws://host/ws/agent?thread_id=<id>      -> resume: memory restored from the checkpointer,
                                              prior turns replayed as agent.history

Thread ids are not authenticated yet: anyone holding an id can resume that
conversation. Bind them to a user once auth lands.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Query, Request, WebSocket
from fastapi.responses import StreamingResponse

from backend.agent.runner import AgentRunner
from backend.api.websocket import QueryItem, Session, serve_session

router = APIRouter()

_THREAD_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
CLOSE_POLICY = 1008


@router.websocket("/ws/agent")
async def ws_agent(ws: WebSocket, thread_id: str | None = None) -> None:
    agent: AgentRunner = ws.app.state.agent
    if thread_id is not None and not _THREAD_RE.match(thread_id):
        await ws.accept()
        await ws.close(code=CLOSE_POLICY, reason="invalid thread_id")
        return
    thread = thread_id or uuid.uuid4().hex
    history = await agent.history(thread) if thread_id is not None else []

    async def on_open(session: Session) -> None:
        if history:
            await session.send(
                {
                    "type": "agent.history",
                    "thread_id": thread,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in history],
                }
            )

    async def handle(session: Session, item: QueryItem) -> None:
        await agent.run_turn(
            thread, item.text, session.send, item.id, item.traceparent, session.span
        )

    await serve_session(
        ws,
        handle,
        hello={
            "thread_id": thread,
            "resumed": bool(history),
            "history_turns": len(history) // 2,
            "reasoner": agent.reasoner.name,
        },
        on_open=on_open,
    )


@router.get("/sse/agent")
async def sse_agent(
    request: Request,
    q: str = Query(min_length=1),
    thread_id: str | None = Query(default=None, pattern=_THREAD_RE.pattern),
):
    agent: AgentRunner = request.app.state.agent
    thread = thread_id or uuid.uuid4().hex
    rid = f"sse-{time.time_ns()}"

    async def body() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)

        async def produce() -> None:
            try:
                await agent.run_turn(thread, q, queue.put, rid, request.headers.get("traceparent"))
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        n = 0
        try:
            while (event := await queue.get()) is not None:
                n += 1
                data = json.dumps(event, separators=(",", ":"))
                yield f"id: {n}\nevent: {event['type']}\ndata: {data}\n\n".encode()
        finally:
            producer.cancel()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Thread-Id": thread},
    )
