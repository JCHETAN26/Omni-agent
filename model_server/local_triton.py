"""A local Triton stand-in that speaks Triton's real gRPC protocol.

This is NOT Triton. It exists so the real `tritonclient.grpc.aio` client can be
exercised over real gRPC on machines where Triton cannot run (macOS, Docker-free
Codespaces), in tests and in load benchmarks. It implements:

* the KServe v2 / Triton `GRPCInferenceService` (ServerLive, ServerReady,
  ModelReady, ModelConfig, ModelInfer), using Triton's own protobufs
* model loading from the same `model_repository/*/config.pbtxt` files
* Triton-style dynamic batching: wait up to `max_queue_delay_microseconds`
  for more requests, merge requests with identical input shapes, cap at
  `max_batch_size`, run up to `instance_group.count` batches concurrently
* onnxruntime execution of `1/model.onnx` when present, otherwise a
  deterministic synthetic model with a configurable compute cost
* W3C `traceparent` extraction from request metadata into a server span

Run standalone:  python -m model_server.local_triton --port 8001
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import grpc
import numpy as np
from google.protobuf import text_format
from opentelemetry.trace import SpanKind
from tritonclient.grpc import model_config_pb2, service_pb2, service_pb2_grpc

from backend.core.telemetry import extract_context, get_tracer

log = logging.getLogger(__name__)

DEFAULT_REPOSITORY = Path(__file__).resolve().parent / "model_repository"
_DTYPES = {"INT64": np.int64, "INT32": np.int32, "FP32": np.float32, "FP16": np.float16}

Runner = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


# ----------------------------------------------------------------------------
# Model runners
# ----------------------------------------------------------------------------


class OnnxRunner:
    def __init__(self, path: Path, intra_op_threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return dict(zip(self.output_names, self.session.run(None, inputs), strict=True))


class SyntheticRunner:
    """Deterministic fake model with a CPU-bound cost of base + per_item ms."""

    def __init__(self, cfg: model_config_pb2.ModelConfig, base_ms: float, per_item_ms: float):
        self.output = cfg.output[0]
        self.dim = int(self.output.dims[0])
        self.base_s = base_ms / 1000
        self.per_item_s = per_item_ms / 1000

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        n = ids.shape[0]
        _busy_wait(self.base_s + self.per_item_s * n)  # holds the worker thread, like real compute
        if "token_type_ids" in inputs:  # cross-encoder: query/passage token overlap
            tt = inputs["token_type_ids"]
            out = np.zeros((n, self.dim), dtype=np.float32)
            for i in range(n):
                q = set(ids[i][(tt[i] == 0) & (mask[i] == 1)].tolist())
                p = ids[i][(tt[i] == 1) & (mask[i] == 1)].tolist()
                out[i, 0] = sum(t in q for t in p) / (len(p) ** 0.5 or 1.0)
            return {self.output.name: out}
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i in range(n):
            for t in ids[i][mask[i] == 1].tolist():
                out[i, (t * 2654435761) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return {self.output.name: out / np.where(norms == 0, 1, norms)}


def _busy_wait(seconds: float) -> None:
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


# ----------------------------------------------------------------------------
# Dynamic batcher
# ----------------------------------------------------------------------------


@dataclass
class _Request:
    tensors: dict[str, np.ndarray]
    future: asyncio.Future
    size: int
    signature: tuple = field(init=False)

    def __post_init__(self) -> None:
        self.signature = tuple(sorted((k, v.shape[1:]) for k, v in self.tensors.items()))


@dataclass
class BatchStats:
    requests: int = 0
    items: int = 0
    batches: int = 0
    batch_sizes: Counter = field(default_factory=Counter)
    compute_ns: Counter = field(default_factory=Counter)  # batch_size -> total ns

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "items": self.items,
            "batches": self.batches,
            "mean_batch_size": round(self.mean_batch_size, 2),
            "max_batch_size": max(self.batch_sizes, default=0),
        }


class DynamicBatcher:
    def __init__(
        self,
        runner: Runner,
        max_batch_size: int,
        max_queue_delay_s: float,
        instances: int,
        executor: ThreadPoolExecutor,
    ):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.max_queue_delay_s = max_queue_delay_s
        self.executor = executor
        self.stats = BatchStats()
        self._queue: asyncio.Queue[_Request] = asyncio.Queue()
        self._slots = asyncio.Semaphore(instances)
        self._task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        for t in list(self._inflight):
            t.cancel()

    async def submit(self, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        size = next(iter(tensors.values())).shape[0]
        req = _Request(tensors, asyncio.get_running_loop().create_future(), size)
        self.stats.requests += 1
        self._queue.put_nowait(req)
        return await req.future

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            group = [await self._queue.get()]
            total = group[0].size
            deadline = loop.time() + self.max_queue_delay_s
            while total < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self._queue.get(), remaining)
                except TimeoutError:
                    break
                group.append(req)
                total += req.size

            by_shape: dict[tuple, list[_Request]] = defaultdict(list)
            for req in group:
                by_shape[req.signature].append(req)
            for reqs in by_shape.values():
                batch: list[_Request] = []
                size = 0
                for req in reqs:
                    if batch and size + req.size > self.max_batch_size:
                        await self._dispatch(batch)
                        batch, size = [], 0
                    batch.append(req)
                    size += req.size
                if batch:
                    await self._dispatch(batch)

    async def _dispatch(self, batch: list[_Request]) -> None:
        await self._slots.acquire()  # one slot per model instance
        task = asyncio.create_task(self._execute(batch))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _execute(self, batch: list[_Request]) -> None:
        batch = [r for r in batch if not r.future.done()]  # caller gave up: skip compute
        if not batch:
            self._slots.release()
            return
        try:
            names = list(batch[0].tensors)
            merged = {n: np.concatenate([r.tensors[n] for r in batch], axis=0) for n in names}
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter_ns()
            outputs = await loop.run_in_executor(self.executor, self.runner, merged)
            size = sum(r.size for r in batch)
            self.stats.compute_ns[size] += time.perf_counter_ns() - t0
            self.stats.batches += 1
            self.stats.items += size
            self.stats.batch_sizes[size] += 1
            offset = 0
            for r in batch:
                part = {k: v[offset : offset + r.size] for k, v in outputs.items()}
                offset += r.size
                if not r.future.done():
                    r.future.set_result(part)
        except Exception as exc:  # noqa: BLE001 - surface to every waiting request
            for r in batch:
                if not r.future.done():
                    r.future.set_exception(exc)
        finally:
            self._slots.release()


# ----------------------------------------------------------------------------
# Model repository + gRPC service
# ----------------------------------------------------------------------------


@dataclass
class LoadedModel:
    config: model_config_pb2.ModelConfig
    batcher: DynamicBatcher
    backend: str  # "onnxruntime" or "synthetic"


def load_config(model_dir: Path) -> model_config_pb2.ModelConfig:
    cfg = model_config_pb2.ModelConfig()
    text_format.Parse((model_dir / "config.pbtxt").read_text(), cfg)
    if not cfg.name:
        cfg.name = model_dir.name
    return cfg


class _Servicer(service_pb2_grpc.GRPCInferenceServiceServicer):
    def __init__(self, server: LocalTritonServer):
        self.server = server

    async def ServerLive(self, request, context):
        return service_pb2.ServerLiveResponse(live=True)

    async def ServerReady(self, request, context):
        return service_pb2.ServerReadyResponse(ready=True)

    async def ModelReady(self, request, context):
        return service_pb2.ModelReadyResponse(ready=request.name in self.server.models)

    async def ModelConfig(self, request, context):
        model = self.server.models.get(request.name)
        if model is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model '{request.name}'")
        return service_pb2.ModelConfigResponse(config=model.config)

    async def ModelStatistics(self, request, context):
        names = [request.name] if request.name else list(self.server.models)
        resp = service_pb2.ModelStatisticsResponse()
        for name in names:
            model = self.server.models.get(name)
            if model is None:
                await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model '{name}'")
            st = model.batcher.stats
            ms = resp.model_stats.add(
                name=name, version="1", inference_count=st.items, execution_count=st.batches
            )
            for size, count in sorted(st.batch_sizes.items()):
                b = ms.batch_stats.add(batch_size=size)
                b.compute_infer.count = count
                b.compute_infer.ns = st.compute_ns[size]
        return resp

    async def ModelInfer(self, request, context):
        metadata = {k: v for k, v in context.invocation_metadata()}
        if "traceparent" in metadata:
            self.server.received_traceparents.append(metadata["traceparent"])
        model = self.server.models.get(request.model_name)
        if model is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, f"unknown model '{request.model_name}'")

        with get_tracer().start_as_current_span(
            "triton.server.ModelInfer",
            context=extract_context(metadata),
            kind=SpanKind.SERVER,
            attributes={"triton.model": request.model_name},
        ):
            tensors: dict[str, np.ndarray] = {}
            for i, inp in enumerate(request.inputs):
                dtype = _DTYPES.get(inp.datatype)
                if dtype is None:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"dtype {inp.datatype}")
                raw = request.raw_input_contents[i]
                tensors[inp.name] = np.frombuffer(raw, dtype=dtype).reshape(tuple(inp.shape))

            expected = {i.name for i in model.config.input}
            if set(tensors) != expected:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"expected inputs {sorted(expected)}, got {sorted(tensors)}",
                )
            size = next(iter(tensors.values())).shape[0]
            if size > model.config.max_batch_size:
                await context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT,
                    f"batch {size} exceeds max_batch_size {model.config.max_batch_size}",
                )

            outputs = await model.batcher.submit(tensors)
            wanted = [o.name for o in request.outputs] or list(outputs)
            resp = service_pb2.ModelInferResponse(
                model_name=request.model_name, model_version="1", id=request.id
            )
            for name in wanted:
                arr = np.ascontiguousarray(outputs[name], dtype=np.float32)
                out = resp.outputs.add()
                out.name = name
                out.datatype = "FP32"
                out.shape.extend(arr.shape)
                resp.raw_output_contents.append(arr.tobytes())
            return resp


class LocalTritonServer:
    def __init__(
        self,
        repository: Path = DEFAULT_REPOSITORY,
        use_onnx: bool = True,
        synthetic_base_ms: float = 2.0,
        synthetic_per_item_ms: float = 0.2,
    ):
        self.repository = repository
        self.use_onnx = use_onnx
        self.synthetic_base_ms = synthetic_base_ms
        self.synthetic_per_item_ms = synthetic_per_item_ms
        self.models: dict[str, LoadedModel] = {}
        self.received_traceparents: deque[str] = deque(maxlen=10_000)
        self._grpc: grpc.aio.Server | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.port: int | None = None

    def _load_models(self) -> None:
        dirs = sorted(p.parent for p in self.repository.glob("*/config.pbtxt"))
        configs = [load_config(d) for d in dirs]
        instances = {c.name: max(1, sum(g.count or 1 for g in c.instance_group)) for c in configs}
        self._executor = ThreadPoolExecutor(
            max_workers=sum(instances.values()), thread_name_prefix="triton-model"
        )
        for d, cfg in zip(dirs, configs, strict=True):
            onnx_path = d / "1" / "model.onnx"
            threads = int(cfg.parameters["intra_op_thread_count"].string_value or 2)
            if self.use_onnx and onnx_path.exists():
                runner, backend = OnnxRunner(onnx_path, threads), "onnxruntime"
            else:
                runner = SyntheticRunner(cfg, self.synthetic_base_ms, self.synthetic_per_item_ms)
                backend = "synthetic"
            delay_us = cfg.dynamic_batching.max_queue_delay_microseconds
            batcher = DynamicBatcher(
                runner, cfg.max_batch_size, delay_us / 1e6, instances[cfg.name], self._executor
            )
            batcher.start()
            self.models[cfg.name] = LoadedModel(cfg, batcher, backend)
            log.info(
                "loaded model %s backend=%s max_batch_size=%d max_queue_delay_us=%d instances=%d",
                cfg.name,
                backend,
                cfg.max_batch_size,
                delay_us,
                instances[cfg.name],
            )

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._load_models()
        self._grpc = grpc.aio.server(
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ]
        )
        service_pb2_grpc.add_GRPCInferenceServiceServicer_to_server(_Servicer(self), self._grpc)
        self.port = self._grpc.add_insecure_port(f"{host}:{port}")
        await self._grpc.start()
        log.info("local triton listening on %s:%d", host, self.port)
        return self.port

    def stats(self) -> dict[str, dict]:
        return {name: m.batcher.stats.as_dict() for name, m in self.models.items()}

    async def stop(self) -> None:
        for m in self.models.values():
            await m.batcher.stop()
        if self._grpc:
            await self._grpc.stop(grace=0.5)
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)


async def _serve(args: argparse.Namespace) -> None:
    server = LocalTritonServer(
        Path(args.repository),
        use_onnx=not args.synthetic,
        synthetic_base_ms=args.base_ms,
        synthetic_per_item_ms=args.per_item_ms,
    )
    port = await server.start(args.host, args.port)
    print(f"READY {port}", flush=True)
    try:
        await server._grpc.wait_for_termination()
    finally:
        await server.stop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--repository", default=str(DEFAULT_REPOSITORY))
    p.add_argument("--synthetic", action="store_true", help="ignore model.onnx files")
    p.add_argument("--base-ms", type=float, default=2.0)
    p.add_argument("--per-item-ms", type=float, default=0.2)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(_serve(p.parse_args()))


if __name__ == "__main__":
    main()
