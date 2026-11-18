"""Async inference clients for embeddings and cross-encoder reranking.

`TritonInferenceClient` talks to Triton Inference Server over gRPC
(`tritonclient.grpc.aio`, native asyncio). The FastAPI event loop never runs
model code:
* tensor compute happens in Triton (another process or host),
* tokenisation runs in a worker thread (`asyncio.to_thread`),
* a semaphore caps in-flight requests so bursts queue in the client instead
  of overloading the server.

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

from backend.core.config import Settings
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
        max_in_flight: int = 128,
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

    async def _infer(self, model: str, tensors: dict[str, np.ndarray], output: str) -> np.ndarray:
        batch, seq = next(iter(tensors.values())).shape
        tracer = get_tracer()
        with tracer.start_as_current_span(
            "triton.infer",
            kind=SpanKind.CLIENT,
            attributes={
                "rpc.system": "grpc",
                "rpc.service": "inference.GRPCInferenceService",
                "rpc.method": "ModelInfer",
                "triton.model": model,
                "triton.batch_size": batch,
                "triton.seq_len": seq,
            },
        ) as span:
            inputs = []
            for name, arr in tensors.items():
                inp = self._grpc.InferInput(name, list(arr.shape), "INT64")
                inp.set_data_from_numpy(arr)
                inputs.append(inp)
            outputs = [self._grpc.InferRequestedOutput(output)]
            t0 = time.perf_counter()
            try:
                async with self._sem:
                    result = await self.client.infer(
                        model,
                        inputs,
                        outputs=outputs,
                        headers=inject_carrier(),
                        client_timeout=self.timeout_s,
                    )
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise InferenceError(f"{model}: {exc}") from exc
            span.set_attribute("triton.latency_ms", (time.perf_counter() - t0) * 1000)
            return result.as_numpy(output)

    async def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        tok = await self._tokenizer("embed")
        enc = await asyncio.to_thread(tok, list(texts), truncation=True, max_length=256)
        chunks = []
        for i in range(0, len(texts), self.max_batch_size):
            ids = enc["input_ids"][i : i + self.max_batch_size]
            length = bucket_length(max(len(r) for r in ids))
            chunks.append(
                self._infer(
                    self.embed_model,
                    {
                        "input_ids": _pad(ids, length),
                        "attention_mask": _pad(enc["attention_mask"][i : i + len(ids)], length),
                    },
                    "sentence_embedding",
                )
            )
        return np.concatenate(await asyncio.gather(*chunks), axis=0)

    async def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        tok = await self._tokenizer("rerank")
        enc = await asyncio.to_thread(
            tok, [query] * len(passages), list(passages), truncation=True, max_length=256
        )
        chunks = []
        for i in range(0, len(passages), self.max_batch_size):
            sl = slice(i, i + self.max_batch_size)
            ids = enc["input_ids"][sl]
            length = bucket_length(max(len(r) for r in ids))
            chunks.append(
                self._infer(
                    self.rerank_model,
                    {
                        "input_ids": _pad(ids, length),
                        "attention_mask": _pad(enc["attention_mask"][sl], length),
                        "token_type_ids": _pad(enc["token_type_ids"][sl], length),
                    },
                    "logits",
                )
            )
        logits = np.concatenate(await asyncio.gather(*chunks), axis=0)
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
