"""In-process Faiss HNSW dense store, a drop-in alternative to Qdrant.

Vectors are L2-normalised by every embedder, so inner product equals cosine
similarity and scores are comparable with Qdrant's COSINE distance.

HNSW cannot delete vectors. Re-ingesting a chunk therefore retires its previous
internal id and adds a fresh one; retired ids are filtered at search time and
counted so the search can over-fetch enough to still return ``limit`` hits.
That keeps the same overwrite semantics as the deterministic Qdrant/ES ids.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from backend.core.models import Chunk


class _ReadWriteLock:
    """Many concurrent searches, or one writer. Faiss HNSW is not safe to
    search while vectors are being added."""

    def __init__(self) -> None:
        self._readers = 0
        self._cond = asyncio.Condition()
        self._writing = False

    async def acquire_read(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writing)
            self._readers += 1

    async def release_read(self) -> None:
        async with self._cond:
            self._readers -= 1
            self._cond.notify_all()

    async def acquire_write(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: not self._writing and self._readers == 0)
            self._writing = True

    async def release_write(self) -> None:
        async with self._cond:
            self._writing = False
            self._cond.notify_all()


class FaissDenseStore:
    """HNSW (inner product) index with in-memory payloads and optional disk persistence."""

    def __init__(
        self,
        path: str | None = None,
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.path = Path(path) if path else None
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None  # faiss.IndexIDMap2 over IndexHNSWFlat, created on first use
        self._dim: int | None = None
        self._next_id = 0
        self._internal: dict[str, int] = {}  # chunk_id -> live internal id
        self._payloads: dict[int, Chunk] = {}  # live internal id -> chunk
        self._retired = 0  # vectors still in the graph but no longer live
        self._lock = _ReadWriteLock()
        if self.path and (self.path / "index.faiss").exists():
            self._load()

    @classmethod
    def from_settings(cls, settings) -> FaissDenseStore:
        return cls(
            settings.faiss_path,
            settings.faiss_hnsw_m,
            settings.faiss_ef_construction,
            settings.faiss_ef_search,
        )

    # -- lifecycle -------------------------------------------------------------

    def _new_index(self, dim: int):
        import faiss

        hnsw = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        hnsw.hnsw.efConstruction = self.ef_construction
        hnsw.hnsw.efSearch = self.ef_search
        return faiss.IndexIDMap2(hnsw)

    async def ensure_collection(self, dim: int) -> None:
        if self._index is None:
            self._index = self._new_index(dim)
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(f"Faiss index has dimension {self._dim}, got {dim}")

    @property
    def count(self) -> int:
        return len(self._payloads)

    # -- writes ----------------------------------------------------------------

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if not chunks:
            return
        matrix = np.asarray(vectors, dtype="float32")
        await self.ensure_collection(matrix.shape[1])
        await self._lock.acquire_write()
        try:
            ids = np.arange(self._next_id, self._next_id + len(chunks), dtype="int64")
            self._next_id += len(chunks)
            await asyncio.to_thread(self._index.add_with_ids, matrix, ids)
            for chunk, internal in zip(chunks, ids.tolist(), strict=True):
                previous = self._internal.get(chunk.chunk_id)
                if previous is not None:
                    self._payloads.pop(previous, None)
                    self._retired += 1
                self._internal[chunk.chunk_id] = internal
                self._payloads[internal] = chunk
        finally:
            await self._lock.release_write()

    # -- reads -----------------------------------------------------------------

    async def search(self, vector: list[float], limit: int) -> list[tuple[Chunk, float]]:
        if self._index is None or not self._payloads:
            return []
        query = np.asarray([vector], dtype="float32")
        await self._lock.acquire_read()
        try:
            # Over-fetch by the number of retired vectors so filtering them out
            # can never leave fewer than `limit` live hits.
            k = min(limit + self._retired, self._index.ntotal)
            scores, ids = await asyncio.to_thread(self._index.search, query, k)
        finally:
            await self._lock.release_read()
        hits: list[tuple[Chunk, float]] = []
        for internal, score in zip(ids[0].tolist(), scores[0].tolist(), strict=True):
            chunk = self._payloads.get(internal)
            if chunk is not None:
                hits.append((chunk, float(score)))
                if len(hits) == limit:
                    break
        return hits

    # -- persistence -----------------------------------------------------------

    def _save(self) -> None:
        import faiss

        self.path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.path / "index.faiss"))
        meta = {
            "dim": self._dim,
            "next_id": self._next_id,
            "retired": self._retired,
            "chunks": {str(i): c.payload() for i, c in self._payloads.items()},
        }
        (self.path / "payloads.json").write_text(json.dumps(meta))

    def _load(self) -> None:
        import faiss

        self._index = faiss.read_index(str(self.path / "index.faiss"))
        faiss.downcast_index(self._index.index).hnsw.efSearch = self.ef_search
        meta = json.loads((self.path / "payloads.json").read_text())
        self._dim = meta["dim"]
        self._next_id = meta["next_id"]
        self._retired = meta["retired"]
        self._payloads = {int(i): Chunk(**c) for i, c in meta["chunks"].items()}
        self._internal = {c.chunk_id: i for i, c in self._payloads.items()}

    async def close(self) -> None:
        if self.path and self._index is not None:
            await self._lock.acquire_write()
            try:
                await asyncio.to_thread(self._save)
            finally:
                await self._lock.release_write()
