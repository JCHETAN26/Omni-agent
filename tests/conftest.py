from pathlib import Path

import pytest

from backend.core.models import Document
from backend.core.stores import InMemoryBM25Index, QdrantDenseStore
from backend.ingestion.embedder import HashingEmbedder
from backend.ingestion.processor import IngestionProcessor
from search.hybrid_retriever import HybridRetriever

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def load_samples() -> list[Document]:
    from backend.ingestion.loader import load_markdown_dir

    return load_markdown_dir(SAMPLES)


@pytest.fixture
def sample_docs() -> list[Document]:
    return load_samples()


@pytest.fixture
async def stack():
    """Offline stack: hashing embedder + in-memory Qdrant + in-process BM25."""
    from qdrant_client import AsyncQdrantClient

    embedder = HashingEmbedder()
    dense = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "test_chunks")
    sparse = InMemoryBM25Index()
    processor = IngestionProcessor(embedder, dense, sparse, chunk_size=400, chunk_overlap=60)
    retriever = HybridRetriever(embedder, dense, sparse, rrf_k=60, candidates=20)
    yield processor, retriever
    await dense.close()


# ----------------------------------------------------------------------------
# Phase 2 fixtures
# ----------------------------------------------------------------------------

import asyncio  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from backend.core.config import Settings  # noqa: E402
from backend.core.telemetry import setup_tracing  # noqa: E402

_SPAN_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="session", autouse=True)
def _tracing():
    provider = setup_tracing(Settings(_env_file=None))
    provider.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
    yield


@pytest.fixture
def spans() -> InMemorySpanExporter:
    _SPAN_EXPORTER.clear()
    return _SPAN_EXPORTER


def make_settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        embedding_provider="hashing",
        inference_backend="mock",
        log_format="text",
        log_level="WARNING",
    )
    return Settings(**{**base, **overrides})


@asynccontextmanager
async def live_server(ingest: bool = True, **overrides):
    """Run the real app under uvicorn on a free port. Yields (ws_url, http_url, app)."""
    import uvicorn

    from backend.main import create_app

    app = create_app(make_settings(**overrides))
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    )
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    if ingest:
        await app.state.services.processor.ingest(load_samples())
    try:
        yield f"ws://127.0.0.1:{port}/ws/stream", f"http://127.0.0.1:{port}", app
    finally:
        server.should_exit = True
        await task
