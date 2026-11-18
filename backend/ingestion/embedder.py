"""Embedding providers. All return L2-normalised vectors (cosine-ready)."""

from __future__ import annotations

import asyncio
import hashlib
import math
from itertools import pairwise
from typing import Protocol

import httpx

from backend.core.config import Settings
from backend.core.stores import tokenize


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local Hugging Face model (default: all-MiniLM-L6-v2, 384-d).

    The model is loaded lazily and inference runs in a worker thread so the
    event loop stays free while PyTorch is busy.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        self.name = model_name
        self.batch_size = batch_size
        self._model = None
        self._lock = asyncio.Lock()

    async def _load(self):
        async with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = await asyncio.to_thread(SentenceTransformer, self.name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._load()
        vecs = await asyncio.to_thread(
            model.encode, texts, batch_size=self.batch_size, normalize_embeddings=True
        )
        return vecs.tolist()


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, batch_size: int = 256):
        self.name = model
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        async def one(batch: list[str]) -> list[list[float]]:
            r = await self._client.post("/embeddings", json={"model": self.name, "input": batch})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]

        results = await asyncio.gather(*(one(b) for b in batches))
        return [v for batch in results for v in batch]


class HashingEmbedder:
    """Deterministic, dependency-free embedder (token + bigram feature hashing).

    Only lexical similarity: meant for offline tests and CI, not quality.
    """

    def __init__(self, dim: int = 384):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
        v = [0.0] * self.dim
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def embedder_from_settings(settings: Settings) -> Embedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIEmbedder(settings.openai_api_key, settings.openai_embedding_model)
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)
