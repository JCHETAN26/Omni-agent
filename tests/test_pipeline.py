import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.core.models import Document, SearchHit
from backend.core.stores import InMemoryBM25Index
from backend.ingestion.chunker import chunk_document
from backend.ingestion.embedder import HashingEmbedder
from search.hybrid_retriever import HybridResult, RetrievalTimings
from search.pipeline import retrieve_with_rerank
from tests.conftest import make_settings


@pytest.mark.parametrize(
    "question,needle_rank",
    [
        ("why the network queue timed out around 0xD15EA5E07F2B", 26),
        ("why the network queue timed out", 4),
    ],
)
async def test_reranking_keeps_protected_lexical_evidence(question, needle_rank):
    hits = []
    for i in range(1, 28):
        text = "network queue timeout"
        if i == needle_rank:
            text += " exact recorder trace=0xD15EA5E07F2B"
        (chunk,) = chunk_document(Document(doc_id=str(i), title=f"Incident {i}", text=text))
        hits.append(
            SearchHit(chunk=chunk, score=1 / i, sparse_rank=1 if i == needle_rank else None)
        )
    retriever = SimpleNamespace(
        search=AsyncMock(
            return_value=HybridResult(
                hits=hits,
                timings=RetrievalTimings(dense_ms=0, sparse_ms=0, fusion_ms=0, total_ms=0),
            )
        )
    )
    reranker = AsyncMock(
        side_effect=lambda _, passages: [
            -99.0 if "exact recorder" in text else 10.0 for text in passages
        ]
    )
    result = await retrieve_with_rerank(
        retriever,
        SimpleNamespace(rerank=reranker),
        make_settings(rerank_enabled=True, rerank_candidates=8),
        question,
        4,
    )
    assert result.reranked and len(result.hits) == 4
    assert len(reranker.await_args.args[1]) == 8
    assert result.hits[-1] == hits[needle_rank - 1]
    assert result.rerank_scores[-1] == -99.0


@pytest.mark.parametrize("scope", ["storage-08 in production", "storage-07 in staging"])
def test_setting_conflict_does_not_mix_hosts_or_environments(scope):
    from search.evidence import setting_conflict

    chunks = [
        chunk_document(
            Document(
                doc_id=str(i), title=f"Runbook {i}", text=f"max_connections = {value} for {s}."
            )
        )[0]
        for i, (value, s) in enumerate([(256, "storage-07 in production"), (100, scope)])
    ]
    assert setting_conflict("What is the max_connections setting?", chunks) is None


def test_conflicting_policy_flags_preserve_source_citations():
    from search.evidence import setting_conflict

    chunks = [
        chunk_document(
            Document(doc_id=value, title="production storage-07", text=f"egress_policy: {value}.")
        )[0]
        for value in ("allow", "deny")
    ]
    message, ids = setting_conflict("What is the egress policy for storage-07?", chunks)
    assert "`allow` [1] and `deny` [2]" in message
    assert ids == [chunk.chunk_id for chunk in chunks]


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


async def test_dense_upsert_is_split_into_bounded_requests(sample_docs):
    """Large ingest batches must not become one long wait=True upsert."""
    from qdrant_client import AsyncQdrantClient

    from backend.core.stores import QdrantDenseStore
    from backend.ingestion.chunker import chunk_document

    client = AsyncQdrantClient(location=":memory:")
    sizes: list[int] = []
    original = client.upsert

    async def recording_upsert(*, collection_name, points, wait):
        sizes.append(len(points))
        return await original(collection_name=collection_name, points=points, wait=wait)

    client.upsert = recording_upsert
    dense = QdrantDenseStore(client, "bounded", upsert_batch=3)
    chunks = [c for d in sample_docs for c in chunk_document(d)][:7]
    await dense.ensure_collection(4)
    await dense.upsert(chunks, [[1.0, 0.0, 0.0, float(i)] for i in range(len(chunks))])
    assert sizes == [3, 3, 1]
    assert (await client.count("bounded")).count == 7
    await dense.close()


def test_remote_qdrant_client_gets_configured_timeout():
    from backend.core.config import Settings
    from backend.core.stores import QdrantDenseStore

    store = QdrantDenseStore.from_settings(
        Settings(qdrant_url="http://127.0.0.1:6333", qdrant_timeout_s=45, qdrant_upsert_batch=256)
    )
    assert store.client._client._timeout == 45
    assert store.upsert_batch == 256


async def test_elasticsearch_client_gets_configured_timeout_and_retries():
    from backend.core.config import Settings
    from backend.core.stores import sparse_index_from_settings

    index = sparse_index_from_settings(
        Settings(elasticsearch_url="http://127.0.0.1:9200", elasticsearch_timeout_s=45)
    )
    assert index.client._request_timeout == 45
    assert index.client._retry_on_timeout is True
    assert index.client._max_retries == 2
    await index.close()


async def test_elasticsearch_400_on_index_check_explains_client_server_mismatch():
    from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
    from elasticsearch import BadRequestError

    from backend.core.stores import ElasticsearchSparseIndex

    index = ElasticsearchSparseIndex("http://127.0.0.1:9200", None, "omni")
    meta = ApiResponseMeta(400, "1.1", HttpHeaders(), 0.0, NodeConfig("http", "127.0.0.1", 9200))
    index.client.indices.exists = AsyncMock(side_effect=BadRequestError("None", meta, None))
    with pytest.raises(RuntimeError, match="newer major version than the server"):
        await index.ensure_index()
    await index.close()


async def test_qdrant_int8_quantization_keeps_exact_top_hits():
    import numpy as np
    from qdrant_client import AsyncQdrantClient

    from backend.core.config import Settings
    from backend.core.stores import QdrantDenseStore

    settings = Settings(_env_file=None, qdrant_quantization="int8", qdrant_hnsw_ef=64)
    store = QdrantDenseStore.from_settings(settings)
    assert store.search_params.hnsw_ef == 64 and store.search_params.quantization.rescore
    assert store.quantization_config().scalar.always_ram is True
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(300, 16)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    docs = [Document(doc_id=f"d{i}", title="t", text=f"text {i}") for i in range(300)]
    chunks = [chunk_document(d)[0] for d in docs]
    await store.ensure_collection(16)
    await store.upsert(chunks, vectors.tolist())
    hits = await store.search(vectors[7].tolist(), 5)
    assert hits[0][0].doc_id == "d7" and hits[0][1] > 0.99
    plain = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "plain")
    assert plain.search_params is None and plain.quantization_config() is None  # "auto"
    bypass = QdrantDenseStore(AsyncQdrantClient(location=":memory:"), "b", quantization="none")
    assert bypass.search_params.quantization.ignore is True
    await store.close()
