"""Phase 2 load benchmark: Triton gRPC inference + WebSocket streaming gateway.

Both servers run as separate processes and are driven over real sockets:

  [1] gRPC: tritonclient.grpc.aio -> Triton (or the local Triton-protocol server
      with ONNX models), closed-loop load at several concurrency levels. Batching
      is read back via Triton's ModelStatistics RPC.
  [2] WebSocket: N concurrent sessions -> uvicorn gateway -> hybrid retrieval
      -> Triton embed + cross-encoder rerank over gRPC -> streamed tokens.
      Every response is integrity-checked (contiguous token indices + SHA-256).

Usage:
    python -m scripts.bench_phase2                     # local server, ONNX if exported
    python -m scripts.bench_phase2 --synthetic         # local server, synthetic compute
    python -m scripts.bench_phase2 --triton-url host:8001   # a real Triton
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
from websockets.asyncio.client import connect

from model_server.inference_client import TritonInferenceClient
from tests.conftest import load_samples

QUERIES = [
    "EIO-5 controller timeout",
    "machines get hot and become sluggish",
    "pods take forever to resolve hostnames",
    "PSU redundancy lost",
    "TLS handshake failure certificate chain",
    "replica lagging behind during compaction",
]


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_local_triton(synthetic: bool) -> tuple[subprocess.Popen, str]:
    cmd = [sys.executable, "-m", "model_server.local_triton", "--port", "0"]
    if synthetic:
        cmd.append("--synthetic")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        if line.startswith("READY"):
            return proc, f"127.0.0.1:{line.split()[1]}"
    raise RuntimeError(f"local triton failed to start: {proc.stderr.read()}")


async def model_counts(client: TritonInferenceClient, model: str) -> tuple[int, int]:
    stats = await client.client.get_inference_statistics(model, as_json=True)
    ms = stats["model_stats"][0]
    return int(ms.get("inference_count", 0)), int(ms.get("execution_count", 0))


class LoopLag:
    """Measures how late a 5 ms timer fires: a blocked event loop shows up here."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            await asyncio.sleep(0.005)
            self.samples.append((time.perf_counter() - t0 - 0.005) * 1000)

    def stop(self) -> None:
        self._stop.set()


async def closed_loop(conc: int, n: int, call) -> tuple[list[float], float, list[float]]:
    """Run `n` calls with `conc` workers. Returns (latencies ms, wall s, client loop lag ms)."""
    lat: list[float] = []
    counter = iter(range(n))
    lag = LoopLag()
    lag_task = asyncio.create_task(lag.run())

    async def worker() -> None:
        for i in counter:
            t0 = time.perf_counter()
            await call(i)
            lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    wall = time.perf_counter() - t0
    lag.stop()
    await lag_task
    return lat, wall, lag.samples


async def bench_grpc(url: str, requests: int, rerank_passages: int) -> None:
    client = TritonInferenceClient(url)
    await client.embed(["warm-up"] * 8)
    await client.rerank("warm-up", ["warm-up"] * 8)

    print("\n[1] gRPC -> Triton  (embed: 1 sentence/request, closed loop)")
    print(
        f"    {'conc':>5} {'reqs':>6} {'req/s':>8} {'p50 ms':>8} {'p95 ms':>8} {'p99 ms':>8}"
        f" {'batches':>8} {'avg batch':>9} {'client lag max ms':>18}"
    )

    async def embed_one(i: int) -> None:
        await client.embed([QUERIES[i % len(QUERIES)] + f" #{i}"])

    for conc in (1, 8, 32, 64, 128):
        n = requests if conc > 1 else max(50, requests // 8)
        items0, execs0 = await model_counts(client, client.embed_model)
        lat, wall, lag = await closed_loop(conc, n, embed_one)
        items1, execs1 = await model_counts(client, client.embed_model)
        batches = execs1 - execs0
        print(
            f"    {conc:>5} {n:>6} {n / wall:>8.0f} {pct(lat, 50):>8.2f} {pct(lat, 95):>8.2f}"
            f" {pct(lat, 99):>8.2f} {batches:>8} {(items1 - items0) / max(batches, 1):>9.1f}"
            f" {max(lag, default=0):>18.2f}"
        )

    texts = [c.text for c in _passages()][:rerank_passages]

    async def rerank_one(i: int) -> None:
        await client.rerank(QUERIES[i % len(QUERIES)], texts)

    conc, n = 16, max(64, requests // 8)
    items0, execs0 = await model_counts(client, client.rerank_model)
    lat, wall, _ = await closed_loop(conc, n, rerank_one)
    items1, execs1 = await model_counts(client, client.rerank_model)
    print(
        f"\n    rerank: conc={conc} reqs={n} x {rerank_passages} pairs -> {n / wall:.1f} req/s"
        f" ({n * rerank_passages / wall:.0f} pairs/s), p50 {pct(lat, 50):.0f} ms,"
        f" p95 {pct(lat, 95):.0f} ms,"
        f" avg batch {(items1 - items0) / max(execs1 - execs0, 1):.1f} pairs"
    )
    await client.close()


def _passages():
    from backend.ingestion.chunker import chunk_document

    return [c for d in load_samples() for c in chunk_document(d, 800, 120)] * 4


async def bench_websocket(
    triton_url: str, sessions: int, per_session: int, rerank: bool = True
) -> None:
    port = free_port()
    env = {
        **os.environ,
        "EMBEDDING_PROVIDER": "triton",
        "INFERENCE_BACKEND": "triton",
        "TRITON_GRPC_URL": triton_url,
        "RERANK_ENABLED": "true" if rerank else "false",
        "RATE_LIMIT_CAPACITY": "1000",
        "WS_MAX_SESSIONS": str(sessions * 2),
        "LOG_LEVEL": "WARNING",
        "QDRANT_URL": "",
        "ELASTICSEARCH_URL": "",
        "REDIS_URL": "",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--ws-max-size",
            "1048576",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient(base_url=base, timeout=60) as http:
            for _ in range(600):
                with contextlib.suppress(httpx.TransportError):
                    if (await http.get("/health")).status_code == 200:
                        break
                await asyncio.sleep(0.1)
            else:
                raise RuntimeError("gateway did not start")
            docs = [d.model_dump() for d in load_samples()]
            report = (await http.post("/ingest", json={"documents": docs})).json()
            health = (await http.get("/health")).json()

        print(
            f"\n[2] WebSocket gateway  ({sessions} concurrent sessions x {per_session} queries,"
            f" embed{' + rerank' if rerank else ''} via Triton gRPC)"
        )
        print(f"    gateway backends: {health['backends']}")
        print(f"    ingested {report['chunks']} chunks (dim {report['embedding_dim']})")

        ttft: list[float] = []
        total: list[float] = []
        integrity_failures: list[str] = []
        errors: list[str] = []
        reranked = fallbacks = 0
        frames_in = 0

        async def session(n: int) -> None:
            nonlocal frames_in, reranked, fallbacks
            async with connect(f"ws://127.0.0.1:{port}/ws/stream", max_size=2**20) as ws:
                hello = json.loads(await ws.recv())
                assert hello["type"] == "session.created"
                last_seq = hello["seq"]
                for i in range(per_session):
                    rid = f"s{n}q{i}"
                    t0 = time.perf_counter()
                    await ws.send(
                        json.dumps(
                            {
                                "type": "query",
                                "id": rid,
                                "text": QUERIES[(n + i) % len(QUERIES)],
                                "top_k": 3,
                            }
                        )
                    )
                    tokens: list[dict] = []
                    first = None
                    while True:
                        ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
                        frames_in += 1
                        if ev["seq"] != last_seq + 1:
                            integrity_failures.append(f"{rid}: seq gap {last_seq}->{ev['seq']}")
                        last_seq = ev["seq"]
                        if ev["type"] == "ping":
                            await ws.send(json.dumps({"type": "pong", "ts": ev["ts"]}))
                        elif ev["type"] == "response.token":
                            first = first or time.perf_counter()
                            tokens.append(ev)
                        elif ev["type"] == "response.sources":
                            reranked += ev["reranked"]
                            fallbacks += ev["rerank_skipped"] is not None
                        elif ev["type"] in ("response.error", "rate_limited"):
                            errors.append(f"{rid}: {ev}")
                            break
                        elif ev["type"] == "response.end":
                            text = "".join(t["text"] for t in tokens)
                            ok = (
                                [t["index"] for t in tokens] == list(range(len(tokens)))
                                and ev["tokens"] == len(tokens)
                                and hashlib.sha256(text.encode()).hexdigest() == ev["sha256"]
                            )
                            if not ok:
                                integrity_failures.append(f"{rid}: integrity check failed")
                            break
                    done = time.perf_counter()
                    if first:
                        ttft.append((first - t0) * 1000)
                    total.append((done - t0) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(session(n) for n in range(sessions)))
        wall = time.perf_counter() - t0
        n_queries = sessions * per_session
        print(
            f"    queries {n_queries}  completed {n_queries - len(errors)}  errors {len(errors)}"
            f"  integrity/seq failures {len(integrity_failures)}  frames received {frames_in}"
        )
        print(f"    reranked {reranked}  rerank fallback to RRF order {fallbacks}")
        print(f"    throughput {n_queries / wall:.0f} queries/s  ({frames_in / wall:.0f} frames/s)")
        print(
            f"    time-to-first-token  p50 {pct(ttft, 50):.1f} ms  p95 {pct(ttft, 95):.1f} ms"
            f"  p99 {pct(ttft, 99):.1f} ms"
        )
        print(
            f"    full response        p50 {pct(total, 50):.1f} ms  p95 {pct(total, 95):.1f} ms"
            f"  p99 {pct(total, 99):.1f} ms"
        )
        for f in (integrity_failures + errors)[:5]:
            print(f"    FAIL {f}")
        async with httpx.AsyncClient(base_url=base, timeout=10) as http:
            lag = (await http.get("/health")).json()["event_loop_lag"]
        print(
            f"    gateway event-loop lag ({lag['samples']} samples @10ms): p50 {lag['p50_ms']} ms"
            f"  p99 {lag['p99_ms']} ms  max {lag['max_ms']} ms"
        )

        stats_client = TritonInferenceClient(triton_url)
        for model in (stats_client.embed_model, stats_client.rerank_model):
            items, execs = await model_counts(stats_client, model)
            print(
                f"    triton {model:<24} lifetime: {items} items in {execs} batches"
                f" (avg {items / max(execs, 1):.1f})"
            )
        await stats_client.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triton-url", help="use an existing Triton instead of the local server")
    p.add_argument("--synthetic", action="store_true", help="local server: synthetic compute")
    p.add_argument("--requests", type=int, default=1024)
    p.add_argument("--sessions", type=int, default=100)
    p.add_argument("--queries-per-session", type=int, default=5)
    p.add_argument("--rerank-passages", type=int, default=10)
    p.add_argument("--no-rerank", action="store_true", help="gateway: skip cross-encoder")
    p.add_argument("--skip-grpc", action="store_true", help="only run the WebSocket benchmark")
    args = p.parse_args()

    proc = None
    url = args.triton_url
    if not url:
        proc, url = start_local_triton(args.synthetic)
    try:
        print("=" * 100)
        kind = (
            "real Triton"
            if args.triton_url
            else (
                "local Triton-protocol server ("
                + ("synthetic" if args.synthetic else "ONNX if exported")
                + ")"
            )
        )
        print(f"Inference server: {kind} at {url}")
        print("=" * 100)
        if not args.skip_grpc:
            await bench_grpc(url, args.requests, args.rerank_passages)
        await bench_websocket(url, args.sessions, args.queries_per_session, not args.no_rerank)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


if __name__ == "__main__":
    asyncio.run(main())
