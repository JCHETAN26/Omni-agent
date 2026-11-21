"""FastAPI gateway: REST (health, ingest, search), SSE and WebSocket streaming.

See the README for the standalone worker launch command.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from anyio.lowlevel import checkpoint
from fastapi import FastAPI, Request, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from backend.agent.runner import AgentRunner
from backend.api import agent_ws, frames, streaming, websocket
from backend.api.websocket import ConnectionManager
from backend.core import metrics
from backend.core.config import Settings, get_settings
from backend.core.models import Document
from backend.core.services import Services
from backend.core.telemetry import LoopLagMonitor, setup_logging, setup_tracing
from backend.ingestion.processor import IngestReport
from search.hybrid_retriever import HybridResult
from search.pipeline import retrieve_with_rerank

log = logging.getLogger(__name__)


class IngestRequest(BaseModel):
    documents: list[Document] = Field(min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings)
    setup_tracing(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = await Services.build(settings)
        if settings.seed_path:
            from backend.ingestion.loader import load_markdown_dir

            docs = load_markdown_dir(settings.seed_path)
            if docs:
                report = await app.state.services.processor.ingest(docs)
                app.state.services.index_version += 1
                log.info(
                    "seeded knowledge base",
                    extra={"path": settings.seed_path, "chunks": report.chunks},
                )
        app.state.connections = ConnectionManager(settings.ws_max_sessions)
        app.state.agent = AgentRunner.from_services(app.state.services)
        # Load AnyIO's asyncio backend before the first WebSocket creates a task group.
        await checkpoint()
        app.state.loop_lag = LoopLagMonitor()
        app.state.loop_lag.start()
        log.info("omniagent started", extra={"backends": app.state.services.backends})
        yield
        await app.state.loop_lag.stop()
        await app.state.agent.close()
        await app.state.services.close()

    app = FastAPI(title="OmniAgent", version="0.3.0", lifespan=lifespan)
    app.include_router(websocket.router)
    app.include_router(streaming.router)
    app.include_router(agent_ws.router)
    app.include_router(frames.router)

    @app.get("/livez", include_in_schema=False)
    async def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request, response: Response) -> dict:
        state = request.app.state
        ready = hasattr(state, "agent") and await state.services.inference.ready()
        response.status_code = 200 if ready else 503
        return {"ready": ready}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics(request: Request) -> Response:
        lag = request.app.state.loop_lag.snapshot()
        metrics.LOOP_LAG.set(lag.get("max_ms", 0) / 1000)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health(request: Request) -> dict:
        services: Services = request.app.state.services
        return {
            "status": "ok",
            "backends": services.backends,
            "inference_ready": await services.inference.ready(),
            "agent": {
                "reasoner": request.app.state.agent.reasoner.name,
                "memory": type(request.app.state.agent.checkpointer).__name__,
            },
            "websocket": request.app.state.connections.snapshot(),
            "event_loop_lag": request.app.state.loop_lag.snapshot(),
        }

    @app.post("/ingest", response_model=IngestReport)
    async def ingest(req: IngestRequest, request: Request) -> IngestReport:
        services: Services = request.app.state.services
        services.index_writes += 1
        try:
            return await services.processor.ingest(req.documents)
        finally:
            services.index_version += 1
            services.index_writes -= 1

    @app.post("/search/ranked")
    async def search_ranked(req: SearchRequest, request: Request) -> dict:
        """Full ranking pipeline: hybrid retrieval, cross-encoder rerank and the
        learning-to-rank stage, each as configured, with per-stage timings."""
        services: Services = request.app.state.services
        t0 = time.perf_counter()
        kb = await retrieve_with_rerank(
            services.retriever,
            services.inference,
            services.settings,
            req.query,
            req.top_k,
            ltr=services.ltr,
        )
        return {
            "hits": kb.sources(),
            "reranked": kb.reranked,
            "rerank_skipped": kb.rerank_skipped,
            "ltr_applied": kb.ltr_applied,
            "ltr_skipped": kb.ltr_skipped,
            "conflict": kb.conflict,
            "timings": {
                "retrieval_ms": kb.retrieval_ms,
                "rerank_ms": kb.rerank_ms,
                "ltr_ms": kb.ltr_ms,
                "total_ms": round((time.perf_counter() - t0) * 1000, 2),
            },
        }

    @app.post("/search", response_model=HybridResult)
    async def search(req: SearchRequest, request: Request) -> HybridResult:
        services: Services = request.app.state.services
        # External indexes can be written by other workers or ingestion jobs.
        if settings.qdrant_url or settings.elasticsearch_url or services.index_writes:
            return await services.retriever.search(req.query, req.top_k)
        digest = hashlib.sha256(f"{req.top_k}:{req.query}".encode()).hexdigest()[:24]
        version = services.index_version
        key = f"search:{services.search_cache_namespace}:v{version}:{digest}"
        cached = await services.cache.get(key)
        if cached is not None and version == services.index_version and not services.index_writes:
            return HybridResult.model_validate(cached)
        result = await services.retriever.search(req.query, req.top_k)
        if version == services.index_version and not services.index_writes:
            await services.cache.set(key, result.model_dump(), settings.cache_ttl_seconds)
        return result

    # WebSocket sessions have their own server span. Skip the generic ASGI
    # span and its extra route lookup for each upgrade.
    http_exclusions = os.environ.get(
        "OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", os.environ.get("OTEL_PYTHON_EXCLUDED_URLS", "")
    )
    exclusions = f"{http_exclusions},^wss?://" if http_exclusions else "^wss?://"
    FastAPIInstrumentor.instrument_app(
        app, excluded_urls=exclusions, exclude_spans=["receive", "send"]
    )
    return app


app = create_app()


def _presize_linux_fd_table(capacity: int) -> None:
    if sys.platform != "linux":
        return

    import fcntl
    import resource

    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    last_fd = capacity - 1
    if soft_limit != resource.RLIM_INFINITY:
        last_fd = min(last_fd, soft_limit - 1)
    if last_fd < 0:
        return

    # Linux retains the grown table after the duplicate is closed.
    try:
        with open(os.devnull, "rb", buffering=0) as source:
            os.close(fcntl.fcntl(source, fcntl.F_DUPFD_CLOEXEC, last_fd))
    except (OSError, OverflowError) as exc:
        log.warning("could not pre-size the worker file descriptor table: %s", exc)


def server_app() -> FastAPI:
    """Prepare the dedicated Uvicorn worker before it serves traffic."""
    # Leave room for the listener, upstream connections and other worker descriptors.
    _presize_linux_fd_table(get_settings().ws_max_sessions + 64)
    # Avoid scanning process-lifetime dependency objects when a connection burst
    # triggers a full collection. Lifespan and request allocations remain collectible.
    gc.collect()
    gc.freeze()
    return app
