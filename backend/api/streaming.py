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
from model_server.inference_client import InferenceError

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\S+\s*")


def _source(hit: SearchHit, rerank_score: float | None) -> dict:
    c = hit.chunk
    return {
        "chunk_id": c.chunk_id,
        "document_title": c.document_title,
        "section_heading": c.section_heading,
        "rrf_score": round(hit.score, 6),
        "rerank_score": None if rerank_score is None else round(rerank_score, 6),
        "dense_rank": hit.dense_rank,
        "sparse_rank": hit.sparse_rank,
    }


def compose_answer(hits: list[SearchHit]) -> str:
    if not hits:
        return "I couldn't find anything relevant in the knowledge base."
    top = hits[0].chunk
    return f'From "{top.section_heading}": {top.text}'


Emit = Callable[[dict], Awaitable[None]]


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
            s = services.settings
            depth = max(top_k, s.rerank_candidates) if s.rerank_enabled else top_k
            result = await services.retriever.search(query, depth)
            hits = result.hits
            rerank: list[float | None] = [None] * len(hits)
            reranked, rerank_skipped = False, None
            if s.rerank_enabled and hits:
                try:
                    scores = await asyncio.wait_for(
                        services.inference.rerank(query, [h.chunk.text for h in hits]),
                        s.rerank_timeout_s,
                    )
                    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
                    hits = [hits[i] for i in order]
                    rerank = [scores[i] for i in order]
                    reranked = True
                except (TimeoutError, InferenceError) as exc:
                    # Reranking is an enhancement: degrade to RRF order, never fail.
                    rerank_skipped = "timeout" if isinstance(exc, TimeoutError) else "error"
                    get_current_span().add_event("rerank.skipped", {"reason": rerank_skipped})
                    log.warning(
                        "rerank skipped (%s), using RRF order",
                        rerank_skipped,
                        extra={"request_id": request_id},
                    )
            hits, rerank = hits[:top_k], rerank[:top_k]
            await emit(
                {
                    "type": "response.sources",
                    "id": request_id,
                    "sources": [_source(h, r) for h, r in zip(hits, rerank, strict=True)],
                    "retrieval_ms": result.timings.total_ms,
                    "reranked": reranked,
                    "rerank_skipped": rerank_skipped,
                }
            )

            digest = hashlib.sha256()
            count = 0
            first_token_ms = None
            for token in _TOKEN_RE.findall(compose_answer(hits)):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t0) * 1000, 2)
                digest.update(token.encode())
                await emit(
                    {"type": "response.token", "id": request_id, "index": count, "text": token}
                )
                count += 1
                await asyncio.sleep(0)  # let other sessions interleave
            await emit(
                {
                    "type": "response.end",
                    "id": request_id,
                    "tokens": count,
                    "sha256": digest.hexdigest(),
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
