import asyncio

from backend.core.stores import InMemoryBM25Index
from backend.ingestion.embedder import HashingEmbedder


async def test_ingest_populates_both_indexes(stack, sample_docs):
    processor, retriever = stack
    report = await processor.ingest(sample_docs)
    assert report.documents == 3
    assert report.chunks >= 8
    assert report.embedding_dim == 384

    count = await processor.dense.client.count(processor.dense.collection)
    assert count.count == report.chunks
    assert len(processor.sparse._chunks) == report.chunks


async def test_dense_payload_carries_metadata(stack, sample_docs):
    processor, _ = stack
    await processor.ingest(sample_docs)
    points, _ = await processor.dense.client.scroll(processor.dense.collection, limit=1)
    payload = points[0].payload
    for key in ("chunk_id", "document_title", "section_heading", "text"):
        assert payload[key]


async def test_hybrid_search_finds_exact_error_code(stack, sample_docs):
    processor, retriever = stack
    await processor.ingest(sample_docs)
    result = await retriever.search("EIO-5 controller timeout", top_k=5)
    top = result.hits[0]
    assert "EIO-5" in top.chunk.section_heading
    assert top.dense_rank is not None and top.sparse_rank is not None
    assert top.score == max(h.score for h in result.hits)


async def test_hybrid_search_returns_ranks_from_each_retriever(stack, sample_docs):
    processor, retriever = stack
    await processor.ingest(sample_docs)
    result = await retriever.search("MTU mismatch VXLAN packet loss", top_k=3)
    assert result.hits[0].chunk.section_heading.endswith("Packet Loss on Edge Routers")
    assert result.timings.total_ms >= 0


async def test_dense_and_sparse_branches_run_concurrently(sample_docs):
    from qdrant_client import AsyncQdrantClient

    from backend.core.stores import QdrantDenseStore
    from backend.ingestion.processor import IngestionProcessor

    events: list[str] = []

    class SlowEmbedder(HashingEmbedder):
        async def embed(self, texts):
            events.append("embed_start")
            await asyncio.sleep(0.2)
            events.append("embed_end")
            return await super().embed(texts)

    class RecordingBM25(InMemoryBM25Index):
        async def index(self, chunks):
            await super().index(chunks)
            events.append("sparse_done")

    dense = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "c")
    processor = IngestionProcessor(SlowEmbedder(), dense, RecordingBM25(), embed_batch_size=1000)
    await processor.ingest(sample_docs)
    assert events == ["embed_start", "sparse_done", "embed_end"]
    await dense.close()
