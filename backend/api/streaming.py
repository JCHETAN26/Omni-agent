"""Response event stream shared by the WebSocket and SSE transports.

Event sequence for one request id:
    response.start -> response.sources -> response.token* -> response.end
or  response.start -> response.error

Integrity: tokens carry a contiguous `index`, and `response.end` carries the
token count plus a SHA-256 of the concatenated token text, so clients can
detect lost, duplicated or reordered frames.

The answer text is a PLACEHOLDER (extractive: best source passage, streamed
word by word) until LLM generation lands. The transport and event contract
stay the same when a real generator replaces it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from opentelemetry import context as otel_context
from opentelemetry.trace import Link, SpanKind, get_current_span

from backend.core.models import SearchHit
from backend.core.services import Services
from backend.core.telemetry import current_trace_id, extract_context, get_tracer
from search.pipeline import retrieve_with_rerank

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\S+\s*")


def compose_answer(hits: list[SearchHit]) -> str:
    if not hits:
        return "I couldn't find anything relevant in the knowledge base."
    top = hits[0].chunk
    return f'From "{top.section_heading}": {top.text}'


Emit = Callable[[dict], Awaitable[None]]


class TokenEmitter:
    """Emits `response.token` frames with contiguous indices and a running SHA-256.

    `extra` fields (e.g. the agent's generation `attempt`) are added to every frame.
    """

    def __init__(self, emit: Emit, request_id: str, **extra):
        self.emit = emit
        self.request_id = request_id
        self.extra = extra
        self.count = 0
        self._digest = hashlib.sha256()
        self._parts: list[str] = []

    async def __call__(self, text: str) -> None:
        if not text:
            return
        self._digest.update(text.encode())
        self._parts.append(text)
        await self.emit(
            {
                "type": "response.token",
                "id": self.request_id,
                "index": self.count,
                "text": text,
                **self.extra,
            }
        )
        self.count += 1
        await asyncio.sleep(0)  # let other sessions interleave

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def text(self) -> str:
        return "".join(self._parts)


async def stream_answer(
    services: Services,
    query: str,
    top_k: int,
    request_id: str,
    emit: Emit,
    traceparent: str | None = None,
    session_span=None,
) -> None:
    """Produce the event sequence for one request through `emit`.

    A callback, not an async generator, so the tracing span opens and closes
    in one task even when the consumer is cancelled mid-stream.
    """
    tracer = get_tracer()
    # Parent: the client's traceparent if sent; else for WebSocket requests a new
    # root trace (linked to the long-lived session span, so sessions don't become
    # one giant trace); else (SSE/HTTP) the current request span.
    if traceparent:
        parent = extract_context({"traceparent": traceparent})
    elif session_span is not None:
        parent = otel_context.Context()
    else:
        parent = None
    links = [Link(session_span.get_span_context())] if session_span is not None else []
    with tracer.start_as_current_span(
        "omniagent.answer",
        context=parent,
        kind=SpanKind.SERVER,
        links=links,
        attributes={"omniagent.request_id": request_id, "omniagent.top_k": top_k},
    ):
        t0 = time.perf_counter()
        await emit({"type": "response.start", "id": request_id, "trace_id": current_trace_id()})
        try:
            kb = await retrieve_with_rerank(
                services.retriever,
                services.inference,
                services.settings,
                query,
                top_k,
                request_id,
            )
            hits = kb.hits
            await emit(
                {
                    "type": "response.sources",
                    "id": request_id,
                    "sources": kb.sources(),
                    "retrieval_ms": kb.retrieval_ms,
                    "reranked": kb.reranked,
                    "rerank_skipped": kb.rerank_skipped,
                }
            )

            tokens = TokenEmitter(emit, request_id)
            first_token_ms = None
            for token in _TOKEN_RE.findall(compose_answer(hits)):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t0) * 1000, 2)
                await tokens(token)
            await emit(
                {
                    "type": "response.end",
                    "id": request_id,
                    "tokens": tokens.count,
                    "sha256": tokens.sha256,
                    "first_token_ms": first_token_ms,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            )
        except Exception as exc:
            log.exception("answer failed", extra={"request_id": request_id})
            get_current_span().record_exception(exc)
            await emit({"type": "response.error", "id": request_id, "error": str(exc)})


# ----------------------------------------------------------------------------
# SSE transport
# ----------------------------------------------------------------------------

router = APIRouter()


@router.get("/sse/stream")
async def sse_stream(
    request: Request,
    q: str = Query(min_length=1),
    top_k: int = Query(default=5, ge=1, le=50),
    request_id: str | None = None,
):
    services: Services = request.app.state.services
    rid = request_id or f"sse-{time.time_ns()}"
    traceparent = request.headers.get("traceparent")

    async def body() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)

        async def produce() -> None:
            try:
                await stream_answer(services, q, top_k, rid, queue.put, traceparent)
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
            producer.cancel()  # client went away: stop retrieval/generation work

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
