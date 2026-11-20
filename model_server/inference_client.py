"""Async inference clients for embeddings and cross-encoder reranking.

`TritonInferenceClient` talks to Triton Inference Server over gRPC
(`tritonclient.grpc.aio`, native asyncio). The FastAPI event loop never runs
model code:
* tensor compute happens in Triton (another process or host),
* tokenisation, padding, and input encoding run in worker threads,
* a semaphore bounds preparation and in-flight RPCs together.

Sequences are padded to a fixed length bucket (32/64/128/256). Triton's
dynamic batcher only merges requests whose input shapes match, so bucketing
lets concurrent requests share a batch.

`MockInferenceClient` keeps the same interface with no server, for tests and
local development.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from opentelemetry.trace import SpanKind, Status, StatusCode

from backend.core import metrics
from backend.core.config import DEFAULT_TRITON_MAX_IN_FLIGHT, Settings
from backend.core.telemetry import get_tracer, inject_carrier

log = logging.getLogger(__name__)

LENGTH_BUCKETS = (32, 64, 128, 256)


class InferenceError(RuntimeError):
    pass


class InferenceClient(Protocol):
    name: str

    async def ready(self) -> bool: ...
    async def warmup(self) -> None: ...
    async def embed(self, texts: Sequence[str]) -> np.ndarray: ...
    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]: ...
    async def close(self) -> None: ...


def bucket_length(n: int, buckets: Sequence[int] = LENGTH_BUCKETS) -> int:
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def _pad(rows: list[list[int]], length: int) -> np.ndarray:
    out = np.zeros((len(rows), length), dtype=np.int64)
    for i, row in enumerate(rows):
        row = row[:length]
        out[i, : len(row)] = row
    return out


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


class TritonInferenceClient:
    def __init__(
        self,
        url: str,
        embed_model: str = "minilm_embedder",
        rerank_model: str = "cross_encoder_reranker",
        embed_tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_tokenizer: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_batch_size: int = 64,
        max_in_flight: int = DEFAULT_TRITON_MAX_IN_FLIGHT,
        timeout_s: float = 10.0,
    ):
        import tritonclient.grpc.aio as grpcclient

        self._grpc = grpcclient
        self.name = f"triton:{url}"
        self.url = url
        self.client = grpcclient.InferenceServerClient(url=url)
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self._tokenizer_names = {"embed": embed_tokenizer, "rerank": rerank_tokenizer}
        self._tokenizers: dict[str, object] = {}
        self._tok_lock = asyncio.Lock()
        self.max_batch_size = max_batch_size
        self.timeout_s = timeout_s
        self._sem = asyncio.Semaphore(max_in_flight)

    @classmethod
    def from_settings(cls, s: Settings) -> TritonInferenceClient:
        return cls(
            s.triton_grpc_url,
            s.triton_embed_model,
            s.triton_rerank_model,
            s.triton_embed_tokenizer,
            s.triton_rerank_tokenizer,
            max_in_flight=s.triton_max_in_flight,
            timeout_s=s.triton_timeout_s,
        )

    async def _tokenizer(self, kind: str):
        if kind not in self._tokenizers:
            async with self._tok_lock:
                if kind not in self._tokenizers:
                    # The import itself takes ~2 s: keep it off the event loop too.
                    self._tokenizers[kind] = await asyncio.to_thread(
                        _load_tokenizer, self._tokenizer_names[kind]
                    )
        return self._tokenizers[kind]

    async def warmup(self) -> None:
        """Load tokenizers before serving traffic (no server round-trip needed)."""
        await asyncio.gather(self._tokenizer("embed"), self._tokenizer("rerank"))

    async def ready(self) -> bool:
        try:
            return bool(
                await self.client.is_server_ready()
                and await self.client.is_model_ready(self.embed_model)
            )
        except Exception:  # noqa: BLE001 - readiness probe must not raise
            return False

    def _prepare_batch(self, tokenizer, texts: Sequence[str], query: str | None):
        if query is None:
            encoded = tokenizer(list(texts), truncation=True, max_length=256)
            names = ("input_ids", "attention_mask")
        else:
            encoded = tokenizer([query] * len(texts), list(texts), truncation=True, max_length=256)
            names = ("input_ids", "attention_mask", "token_type_ids")
        length = bucket_length(max(map(len, encoded["input_ids"])))
        inputs = []
        for name in names:
            arr = _pad(encoded[name], length)
            inp = self._grpc.InferInput(name, list(arr.shape), "INT64")
            inp.set_data_from_numpy(arr)
            inputs.append(inp)
        return inputs, length

    async def _infer(
        self,
        model: str,
        tokenizer,
        texts: Sequence[str],
        output: str,
        query: str | None,
    ) -> np.ndarray:
        t0 = time.perf_counter()
        async with self._sem:
            inputs, length = await asyncio.to_thread(self._prepare_batch, tokenizer, texts, query)
            tracer = get_tracer()
            with tracer.start_as_current_span(
                "triton.infer",
                kind=SpanKind.CLIENT,
                attributes={
                    "triton.model": model,
                    "triton.batch_size": len(texts),
                    "triton.seq_len": length,
                },
            ) as span:
                try:
                    result = await self.client.infer(
                        model,
                        inputs,
                        outputs=[self._grpc.InferRequestedOutput(output)],
                        headers=inject_carrier(),
                        client_timeout=self.timeout_s,
                    )
                except Exception as exc:
                    metrics.TRITON_ERRORS.labels(model).inc()
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise InferenceError(f"{model}: {exc}") from exc
                elapsed = time.perf_counter() - t0
                metrics.TRITON_SECONDS.labels(model).observe(elapsed)
                span.set_attribute("triton.latency_ms", elapsed * 1000)
                return result.as_numpy(output)

    async def _infer_batches(
        self, kind: str, model: str, output: str, texts: Sequence[str], query: str | None = None
    ) -> np.ndarray:
        tokenizer = await self._tokenizer(kind)
        batches = await asyncio.gather(
            *(
                self._infer(model, tokenizer, texts[i : i + self.max_batch_size], output, query)
                for i in range(0, len(texts), self.max_batch_size)
            )
        )
        return np.concatenate(batches, axis=0) if len(batches) > 1 else batches[0]

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return await self._infer_batches("embed", self.embed_model, "sentence_embedding", texts)

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        logits = await self._infer_batches("rerank", self.rerank_model, "logits", passages, query)
        return logits.reshape(-1).astype(float).tolist()

    async def close(self) -> None:
        await self.client.close()


class MockInferenceClient:
    """In-process stand-in: hashing embeddings + lexical-overlap rerank scores."""

    def __init__(self, dim: int = 384, latency_ms: float = 0.0):
        from backend.ingestion.embedder import HashingEmbedder

        self.name = "mock"
        self.latency_ms = latency_ms
        self._hasher = HashingEmbedder(dim)

    async def ready(self) -> bool:
        return True

    async def warmup(self) -> None:
        return None

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        with get_tracer().start_as_current_span("mock.embed", attributes={"batch": len(texts)}):
            if self.latency_ms:
                await asyncio.sleep(self.latency_ms / 1000)
            return np.asarray(await self._hasher.embed(list(texts)), dtype=np.float32)

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        from backend.core.stores import tokenize

        with get_tracer().start_as_current_span("mock.rerank", attributes={"batch": len(passages)}):
            if self.latency_ms:
                await asyncio.sleep(self.latency_ms / 1000)
            q = set(tokenize(query))
            scores = []
            for p in passages:
                toks = tokenize(p)
                overlap = sum(1 for t in toks if t in q) / (len(toks) ** 0.5 or 1.0)
                # tiny deterministic tie-breaker so equal-overlap passages keep a stable order
                tie = int(hashlib.blake2b(p.encode(), digest_size=2).hexdigest(), 16) / 1e9
                scores.append(overlap + tie)
            return scores

    async def close(self) -> None:
        return None


class TritonEmbedder:
    """Adapts an InferenceClient to the Phase 1 `Embedder` protocol."""

    def __init__(self, client: InferenceClient):
        self.client = client
        self.name = f"{client.name}:embed"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return (await self.client.embed(texts)).tolist()


def inference_client_from_settings(settings: Settings) -> InferenceClient:
    if settings.inference_backend == "triton":
        return TritonInferenceClient.from_settings(settings)
    return MockInferenceClient()
