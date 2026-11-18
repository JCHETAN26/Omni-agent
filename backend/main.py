"""FastAPI gateway: REST (health, ingest, search), SSE and WebSocket streaming.

Run:  uvicorn backend.main:app --reload
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from backend.agent.runner import AgentRunner
from backend.api import agent_ws, streaming, websocket
from backend.api.websocket import ConnectionManager
from backend.core import metrics
from backend.core.config import Settings, get_settings
from backend.core.models import Document
from backend.core.services import Services
from backend.core.telemetry import LoopLagMonitor, setup_logging, setup_tracing
from backend.ingestion.processor import IngestReport
from search.hybrid_retriever import HybridResult

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
        report = await services.processor.ingest(req.documents)
        services.index_version += 1
        return report

    @app.post("/search", response_model=HybridResult)
    async def search(req: SearchRequest, request: Request) -> HybridResult:
        services: Services = request.app.state.services
        digest = hashlib.sha256(f"{req.top_k}:{req.query}".encode()).hexdigest()[:24]
        key = f"search:v{services.index_version}:{digest}"
        if (cached := await services.cache.get(key)) is not None:
            return HybridResult.model_validate(cached)
        result = await services.retriever.search(req.query, req.top_k)
        await services.cache.set(key, result.model_dump(), settings.cache_ttl_seconds)
        return result

    # HTTP spans for REST/SSE. Per-frame ASGI send/receive spans are excluded:
    # WebSocket frames get their own explicit spans in backend/api/websocket.py.
    FastAPIInstrumentor.instrument_app(app, exclude_spans=["receive", "send"])
    return app


app = create_app()
