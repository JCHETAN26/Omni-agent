"""Storage adapters: dense vectors (Qdrant) and sparse BM25 (Elasticsearch).

Both expose a tiny async interface so ingestion and retrieval never touch
client libraries directly, and the fallbacks are drop-in replacements.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    VectorParams,
)

from backend.core.config import Settings
from backend.core.models import Chunk

# ----------------------------------------------------------------------------
# Dense stores
# ----------------------------------------------------------------------------


class DenseStore(Protocol):
    async def ensure_collection(self, dim: int) -> None: ...
    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...
    async def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]: ...
    async def close(self) -> None: ...


def dense_store_from_settings(settings: Settings) -> DenseStore:
    if settings.dense_backend == "faiss":
        from backend.core.faiss_store import FaissDenseStore

        return FaissDenseStore.from_settings(settings)
    return QdrantDenseStore.from_settings(settings)


class QdrantDenseStore:
    """Qdrant-backed ANN store. Uses embedded in-memory mode when no URL is set."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection: str,
        upsert_batch: int = 512,
        *,
        quantization: str = "auto",
        hnsw_ef: int | None = None,
        oversampling: float = 2.0,
    ):
        self.client = client
        self.collection = collection
        self.upsert_batch = upsert_batch
        self.quantization = quantization
        self._exists = False
        # int8 scalar quantization keeps compressed vectors in RAM for the graph
        # walk, then rescores the top `limit * oversampling` with full vectors.
        # Qdrant searches a quantized collection with quantization by default, so
        # "none" must opt out explicitly to measure an unquantized baseline.
        quant = {
            "int8": QuantizationSearchParams(rescore=True, oversampling=oversampling),
            "none": QuantizationSearchParams(ignore=True),
        }.get(quantization)
        self.search_params = (
            SearchParams(hnsw_ef=hnsw_ef, quantization=quant) if hnsw_ef or quant else None
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantDenseStore:
        if settings.qdrant_url:
            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=settings.qdrant_timeout_s,
            )
        else:
            client = AsyncQdrantClient(location=":memory:")
        return cls(
            client,
            settings.qdrant_collection,
            settings.qdrant_upsert_batch,
            quantization=settings.qdrant_quantization,
            hnsw_ef=settings.qdrant_hnsw_ef,
            oversampling=settings.qdrant_oversampling,
        )

    def quantization_config(self) -> ScalarQuantization | None:
        if self.quantization != "int8":
            return None
        return ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True)
        )

    async def apply_quantization(self) -> None:
        """Enable the configured quantization on an existing collection (Qdrant rebuilds it)."""
        config = self.quantization_config()
        if config is not None:
            await self.client.update_collection(self.collection, quantization_config=config)

    async def ensure_collection(self, dim: int) -> None:
        if not await self.client.collection_exists(self.collection):
            await self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                quantization_config=self.quantization_config(),
            )
        self._exists = True

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(id=c.chunk_id, vector=v, payload=c.payload())
            for c, v in zip(chunks, vectors, strict=True)
        ]
        # Bounded requests keep each wait=True call well inside the client timeout
        # while Qdrant's optimizer is rebuilding segments on a large collection.
        for start in range(0, len(points), self.upsert_batch):
            await self.client.upsert(
                collection_name=self.collection,
                points=points[start : start + self.upsert_batch],
                wait=True,
            )

    async def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        if not self._exists:  # nothing ingested yet (or created by another process)
            self._exists = await self.client.collection_exists(self.collection)
            if not self._exists:
                return []
        resp = await self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
            search_params=self.search_params,
        )
        return [(Chunk(**p.payload), p.score) for p in resp.points]

    async def close(self) -> None:
        await self.client.close()


# ----------------------------------------------------------------------------
# Sparse index (BM25)
# ----------------------------------------------------------------------------


class SparseIndex(Protocol):
    async def ensure_index(self) -> None: ...
    async def index(self, chunks: list[Chunk]) -> None: ...
    async def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]: ...
    async def close(self) -> None: ...


class ElasticsearchSparseIndex:
    """Elasticsearch inverted index scored with BM25 (k1=1.2, b=0.75)."""

    def __init__(self, url: str, api_key: str | None, index_name: str, timeout_s: float = 60.0):
        from elasticsearch import AsyncElasticsearch

        # Chunk IDs are deterministic document _ids, so a retried bulk overwrites
        # rather than duplicates.
        self.client = AsyncElasticsearch(
            url, api_key=api_key, request_timeout=timeout_s, retry_on_timeout=True, max_retries=2
        )
        self.index_name = index_name

    async def ensure_index(self) -> None:
        from elasticsearch import BadRequestError
        from elasticsearch import __version__ as client_version

        try:
            exists = await self.client.indices.exists(index=self.index_name)
        except BadRequestError as exc:
            # An index-existence HEAD has no body to be malformed. A bare 400 here
            # almost always means the client is a newer major version than the
            # server: it sends "compatible-with=<major>" headers the server rejects.
            client = ".".join(map(str, client_version))
            raise RuntimeError(
                f"Elasticsearch rejected the existence check for {self.index_name!r} with HTTP "
                f"400. This usually means the Python client ({client}) is a newer major version "
                f"than the server; install elasticsearch=={client_version[0] - 1}.* for an older "
                "server, or match the client's major version to the server's."
            ) from exc
        if exists:
            return
        await self.client.indices.create(
            index=self.index_name,
            settings={
                "number_of_shards": 1,
                "similarity": {"omni_bm25": {"type": "BM25", "k1": 1.2, "b": 0.75}},
            },
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "document_title": {"type": "text", "similarity": "omni_bm25"},
                    "section_heading": {"type": "text", "similarity": "omni_bm25"},
                    "position": {"type": "integer"},
                    "text": {"type": "text", "similarity": "omni_bm25"},
                }
            },
        )

    async def index(self, chunks: list[Chunk]) -> None:
        from elasticsearch.helpers import async_bulk

        actions = [
            {"_index": self.index_name, "_id": c.chunk_id, "_source": c.payload()} for c in chunks
        ]
        await async_bulk(self.client, actions, refresh="wait_for")

    async def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        resp = await self.client.search(
            index=self.index_name,
            size=limit,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["text", "section_heading^0.5", "document_title^0.3"],
                }
            },
        )
        return [(Chunk(**h["_source"]), h["_score"]) for h in resp["hits"]["hits"]]

    async def close(self) -> None:
        await self.client.close()


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer, close to Elasticsearch's `standard` analyzer."""
    return _TOKEN_RE.findall(text.lower())


class InMemoryBM25Index:
    """Pure-Python Okapi BM25 with the same defaults as Elasticsearch.

    Development fallback when ELASTICSEARCH_URL is not configured. Supports
    incremental indexing and re-indexing a chunk_id (overwrite semantics).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self._chunks: dict[str, Chunk] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._len: dict[str, int] = {}
        self._df: Counter[str] = Counter()

    async def ensure_index(self) -> None:
        return None

    async def index(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.chunk_id in self._tf:  # overwrite: retract old stats first
                self._df.subtract(self._tf[c.chunk_id].keys())
            tf = Counter(tokenize(c.text))
            self._chunks[c.chunk_id] = c
            self._tf[c.chunk_id] = tf
            self._len[c.chunk_id] = sum(tf.values())
            self._df.update(tf.keys())

    async def search(self, query: str, limit: int) -> list[tuple[Chunk, float]]:
        n = len(self._chunks)
        if n == 0:
            return []
        avgdl = sum(self._len.values()) / n
        terms = set(tokenize(query))
        scores: dict[str, float] = {}
        for term in terms:
            df = self._df.get(term, 0)
            if df <= 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))  # Lucene's BM25 idf
            for cid, tf in self._tf.items():
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self._len[cid] / avgdl)
                scores[cid] = scores.get(cid, 0.0) + idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [(self._chunks[cid], s) for cid, s in ranked]

    async def close(self) -> None:
        return None


def sparse_index_from_settings(settings: Settings) -> SparseIndex:
    if settings.elasticsearch_url:
        return ElasticsearchSparseIndex(
            settings.elasticsearch_url,
            settings.elasticsearch_api_key,
            settings.elasticsearch_index,
            settings.elasticsearch_timeout_s,
        )
    return InMemoryBM25Index()
