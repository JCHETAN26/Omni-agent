import asyncio

import numpy as np
import pytest

from backend.core.faiss_store import FaissDenseStore
from backend.core.models import Chunk

DIM = 32


def _chunk(i: int, text: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"00000000-0000-0000-0000-{i:012d}",
        doc_id=f"doc-{i}",
        document_title=f"title {i}",
        section_heading="s",
        position=0,
        text=text or f"text {i}",
    )


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


async def test_hnsw_matches_exact_search_on_random_vectors():
    rng = np.random.default_rng(7)
    vectors = _unit(rng.normal(size=(2000, DIM)).astype("float32"))
    store = FaissDenseStore(ef_search=256)
    await store.upsert([_chunk(i) for i in range(2000)], vectors.tolist())
    queries = _unit(rng.normal(size=(50, DIM)).astype("float32"))
    overlap = 0
    for q in queries:
        exact = set(np.argsort(-(vectors @ q))[:10].tolist())
        hits = await store.search(q.tolist(), 10)
        assert len(hits) == 10
        assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)
        overlap += len(exact & {int(c.doc_id.split("-")[1]) for c, _ in hits})
    assert overlap / 500 >= 0.95  # recall@10 of the approximate graph


async def test_reupsert_overwrites_instead_of_duplicating():
    store = FaissDenseStore()
    a = [1.0] + [0.0] * (DIM - 1)
    b = [0.0, 1.0] + [0.0] * (DIM - 2)
    await store.upsert([_chunk(1, "old"), _chunk(2)], [a, b])
    await store.upsert([_chunk(1, "new")], [b])  # same id, new text and vector
    assert store.count == 2
    hits = await store.search(b, 5)
    assert sorted(c.text for c, _ in hits) == ["new", "text 2"]
    # Searching for the retired vector must still return `limit` live hits.
    assert len(await store.search(a, 2)) == 2


async def test_index_persists_across_restarts(tmp_path):
    rng = np.random.default_rng(1)
    vectors = _unit(rng.normal(size=(100, DIM)).astype("float32"))
    store = FaissDenseStore(path=str(tmp_path / "idx"))
    await store.upsert([_chunk(i) for i in range(100)], vectors.tolist())
    before = await store.search(vectors[3].tolist(), 5)
    await store.close()

    reopened = FaissDenseStore(path=str(tmp_path / "idx"))
    assert reopened.count == 100
    after = await reopened.search(vectors[3].tolist(), 5)
    assert [c.chunk_id for c, _ in after] == [c.chunk_id for c, _ in before]
    assert after[0][0].doc_id == "doc-3"


async def test_dimension_mismatch_is_rejected():
    store = FaissDenseStore()
    await store.ensure_collection(DIM)
    with pytest.raises(ValueError):
        await store.ensure_collection(DIM + 1)


async def test_searches_interleaved_with_writes_stay_consistent():
    rng = np.random.default_rng(3)
    store = FaissDenseStore()
    first = _unit(rng.normal(size=(500, DIM)).astype("float32"))
    await store.upsert([_chunk(i) for i in range(500)], first.tolist())

    async def writer():
        for batch in range(5):
            rows = _unit(rng.normal(size=(200, DIM)).astype("float32"))
            base = 500 + batch * 200
            await store.upsert([_chunk(base + i) for i in range(200)], rows.tolist())

    async def reader():
        for _ in range(40):
            hits = await store.search(first[0].tolist(), 10)
            assert len(hits) == 10
            assert len({c.chunk_id for c, _ in hits}) == 10

    await asyncio.gather(writer(), *(reader() for _ in range(4)))
    assert store.count == 1500


async def test_hybrid_retriever_runs_on_faiss(sample_docs):
    from backend.core.stores import InMemoryBM25Index
    from backend.ingestion.embedder import HashingEmbedder
    from backend.ingestion.processor import IngestionProcessor
    from search.hybrid_retriever import HybridRetriever

    embedder, dense, sparse = HashingEmbedder(), FaissDenseStore(), InMemoryBM25Index()
    await IngestionProcessor(embedder, dense, sparse).ingest(sample_docs)
    result = await HybridRetriever(embedder, dense, sparse).search("sk_buff ring buffer", 5)
    assert result.hits
    top = result.hits[0]
    assert top.dense_score is not None or top.sparse_score is not None
    for hit in result.hits:
        assert (hit.dense_rank is None) == (hit.dense_score is None)
        assert (hit.sparse_rank is None) == (hit.sparse_score is None)
