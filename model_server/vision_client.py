"""Image classification clients for live frames.

`TritonVisionClient` sends one preprocessed frame per request to a ViT model
on a real Triton Inference Server over gRPC. Requests from concurrent sessions
are separate RPCs; Triton's dynamic batcher (when the model config enables it)
merges them into one batch on the server. Top-k labels come back through
Triton's classification extension (`class_count`), as "score:index:label".

`MockVisionClient` returns deterministic labels with no server.

Callers treat `VisionUnavailable` as "no visual context": the agent keeps
answering from text alone when Triton is down.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.core import metrics
from backend.core.config import Settings


class VisionUnavailable(RuntimeError):
    """Triton is unreachable, the model is not ready, or the call timed out."""


@dataclass(frozen=True)
class Label:
    label: str
    score: float


class VisionClient(Protocol):
    name: str

    async def ready(self) -> bool: ...
    async def classify(self, pixels: np.ndarray) -> list[Label]: ...
    async def close(self) -> None: ...


def parse_classification(raw: np.ndarray) -> list[Label]:
    """Triton classification output: BYTES entries "score:index[:label]"."""
    out = []
    for item in np.asarray(raw).reshape(-1):
        text = item.decode() if isinstance(item, bytes) else str(item)
        parts = text.split(":", 2)
        if len(parts) < 2:
            raise ValueError(f"unexpected classification entry {text!r}")
        label = parts[2] if len(parts) == 3 and parts[2] else f"class_{parts[1]}"
        out.append(Label(label, float(parts[0])))
    return out


class TritonVisionClient:
    def __init__(self, url: str, model: str, top_k: int = 5, timeout_s: float = 5.0):
        import tritonclient.grpc.aio as grpcclient

        self._grpc = grpcclient
        self.client = grpcclient.InferenceServerClient(url=url)
        self.url, self.model, self.top_k, self.timeout_s = url, model, top_k, timeout_s
        self.name = f"triton:{url}:{model}"

    @classmethod
    def from_settings(cls, s: Settings) -> TritonVisionClient:
        return cls(s.triton_grpc_url, s.triton_vision_model, s.vision_top_k, s.vision_timeout_s)

    async def ready(self) -> bool:
        try:
            return bool(
                await self.client.is_server_ready() and await self.client.is_model_ready(self.model)
            )
        except Exception:  # noqa: BLE001 - readiness must not raise
            return False

    async def classify(self, pixels: np.ndarray) -> list[Label]:
        if pixels.ndim != 4 or pixels.shape[0] != 1:
            raise ValueError("classify takes one (1, 3, H, W) frame")
        inp = self._grpc.InferInput("pixel_values", list(pixels.shape), "FP32")
        inp.set_data_from_numpy(np.ascontiguousarray(pixels, dtype=np.float32))
        out = self._grpc.InferRequestedOutput("logits", class_count=self.top_k)
        t0 = time.perf_counter()
        try:
            result = await self.client.infer(
                self.model, [inp], outputs=[out], client_timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - any transport/server failure
            metrics.TRITON_ERRORS.labels(self.model).inc()
            raise VisionUnavailable(f"{self.model}: {exc}") from exc
        metrics.TRITON_SECONDS.labels(self.model).observe(time.perf_counter() - t0)
        return parse_classification(result.as_numpy("logits"))

    async def close(self) -> None:
        await self.client.close()


class MockVisionClient:
    """Deterministic labels derived from the pixels; no model, no server."""

    LABELS = ("monitor", "screen", "desktop computer", "notebook", "web site", "switch")

    def __init__(self, top_k: int = 5, latency_ms: float = 0.0):
        self.name, self.top_k, self.latency_ms = "mock", top_k, latency_ms

    async def ready(self) -> bool:
        return True

    async def classify(self, pixels: np.ndarray) -> list[Label]:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        seed = int(hashlib.blake2b(pixels.tobytes()[:4096], digest_size=2).hexdigest(), 16)
        names = [self.LABELS[(seed + i) % len(self.LABELS)] for i in range(self.top_k)]
        return [Label(n, round(1.0 / (i + 2), 4)) for i, n in enumerate(names)]

    async def close(self) -> None:
        return None


def vision_client_from_settings(s: Settings) -> VisionClient | None:
    if s.vision_backend == "triton":
        return TritonVisionClient.from_settings(s)
    if s.vision_backend == "mock":
        return MockVisionClient(s.vision_top_k)
    return None
