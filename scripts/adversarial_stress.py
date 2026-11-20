"""Generate synthetic technical data and run isolated retrieval and shared-thread probes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import cProfile
import gc
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np
import redis.asyncio
from fastapi import Request, WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect

from backend.agent.runner import AgentRunner
from backend.core.config import Settings
from backend.core.models import Document
from backend.core.services import Services
from backend.core.stores import InMemoryBM25Index
from backend.ingestion.chunker import chunk_document
from model_server.local_triton import DEFAULT_REPOSITORY, LocalTritonServer
from search.pipeline import retrieve_with_rerank

NEEDLE = "0xD15EA5E07F2B"
BURIED = "logs-buried"
POLICY = "policy-256"
REPLICA = "policy-100"


def corpus() -> list[Document]:
    docs = []
    templates = {
        "kube": (
            "Kubernetes networking",
            "Service discovery distributes requests across healthy endpoints. "
            "Connection tracking retains transport state during endpoint rollouts. "
            "Network queue timeout reports can also reflect congestion and retries.",
        ),
        "netdev": (
            "Linux netdev",
            "A socket waits in epoll_wait until readiness changes. The sk_buff records "
            "packet metadata while the driver drains the receive ring. A network queue "
            "timeout can reflect stalled completion polling or delayed wakeups.",
        ),
        "logs": (
            "Server incident log",
            "worker WARN network queue timeout, budget expired while awaiting a completion. "
            "Supervisor marked the event transient and scheduled a retry. This fixture "
            "does not establish a connection limit or a root cause.",
        ),
    }
    for kind, (title, text) in templates.items():
        for n in range(180):
            sections = [
                f"## {name}\n{text} Synthetic trace {kind}-{n:03d}/{i}."
                for i, name in enumerate(("Diagnosis", "Operational context", "Recovery"))
            ]
            docs.append(
                Document(
                    doc_id=f"{kind}-{n:03d}", title=f"{title} {n:03d}", text="\n".join(sections)
                )
            )
    lines = [
        f"seq={i:04d} INFO cpu={i % 16} sk_buff rx ring idle pid={1000 + i} "
        f"trace=0x{i:012x} epoll_wait returned zero"
        for i in range(2750)
    ]
    lines[2612] = (
        f"seq=2612 CRIT kernel panic EIO-5 trace={NEEDLE} OR-71 network queue timeout "
        "caused by a stalled netdev transmit completion; see OR-71 connection policy"
    )
    docs.extend(
        [
            Document(
                doc_id=BURIED, title="Unformatted kernel flight recorder", text="\n".join(lines)
            ),
            Document(
                doc_id=POLICY,
                title="OR-71 connection policy A",
                text="The OR-71 network queue timeout policy sets max_connections = 256 "
                "for storage-07. This is the production hard limit.",
            ),
            Document(
                doc_id=REPLICA,
                title="OR-71 connection policy B",
                text="The OR-71 network queue timeout policy sets max_connections = 100 "
                "for storage-07. This is the production hard limit.",
            ),
        ]
    )
    return docs


def emit(section: str, **data) -> None:
    print(json.dumps({"section": section, **data}, default=str), flush=True)


def ids(rows: list) -> list[str]:
    return [chunk.chunk_id for chunk, _ in rows]


def ranking(rank_ids: list[str], targets: set[str]) -> int | None:
    return next((i for i, item in enumerate(rank_ids, start=1) if item in targets), None)


async def verify_indexes(services: Services, expected: dict) -> tuple[int, int]:
    dense_ids = set()
    offset = None
    while True:
        points, offset = await services.dense.client.scroll(
            services.dense.collection, offset=offset, limit=256, with_payload=True
        )
        for point in points:
            dense_ids.add(str(point.id))
            assert point.payload == expected[str(point.id)].payload()
        if offset is None:
            break
    sparse = services.sparse
    if isinstance(sparse, InMemoryBM25Index):
        sparse_ids = set(sparse._chunks)
        assert sparse._chunks == expected
    else:
        sparse_ids = set()
        await sparse.client.indices.refresh(index=sparse.index_name)
        keys = list(expected)
        for start in range(0, len(keys), 200):
            response = await sparse.client.mget(
                index=sparse.index_name, ids=keys[start : start + 200]
            )
            for doc in response["docs"]:
                assert doc["found"] and doc["_source"] == expected[doc["_id"]].payload()
                sparse_ids.add(doc["_id"])
        count = await sparse.client.count(index=sparse.index_name)
        assert count["count"] == len(expected)
    assert dense_ids == sparse_ids == set(expected)
    return len(dense_ids), len(sparse_ids)


async def search_probes(args: argparse.Namespace) -> None:
    for model in ("minilm_embedder", "cross_encoder_reranker"):
        assert (DEFAULT_REPOSITORY / model / "1/model.onnx").is_file(), f"Export {model} first"
    triton = LocalTritonServer()
    port = await triton.start()
    name = f"omni_stress_{uuid.uuid4().hex[:10]}"
    settings = Settings(
        _env_file=None,
        qdrant_url=args.qdrant_url,
        elasticsearch_url=args.elasticsearch_url,
        qdrant_collection=name,
        elasticsearch_index=name,
        embedding_provider="triton",
        inference_backend="triton",
        triton_grpc_url=f"127.0.0.1:{port}",
        rerank_enabled=True,
        llm_provider="openai" if args.model else "heuristic",
        llm_base_url=args.model_url,
        llm_model=args.model,
        llm_api_key=None,
        openai_api_key=None,
        redis_url=None,
        otel_exporter_otlp_endpoint=None,
        log_level="WARNING",
    )
    services = await Services.build(settings)
    try:
        docs = corpus()
        counts: dict[str, list[int]] = defaultdict(list)
        expected = {}
        needle_ids = set()
        target_ids = {key: set() for key in (POLICY, REPLICA)}
        for doc in docs:
            chunks = chunk_document(doc, settings.chunk_size, settings.chunk_overlap)
            counts[doc.doc_id.split("-")[0]].append(len(chunks))
            for chunk in chunks:
                expected[chunk.chunk_id] = chunk
                if NEEDLE in chunk.text:
                    needle_ids.add(chunk.chunk_id)
                if doc.doc_id in target_ids:
                    target_ids[doc.doc_id].add(chunk.chunk_id)
        assert len(needle_ids) == 1
        report = await services.processor.ingest(docs)
        dense, sparse = await verify_indexes(services, expected)
        assert report.chunks == dense == sparse
        emit(
            "ingest",
            **report.model_dump(),
            dense_count=dense,
            sparse_count=sparse,
            backends=services.backends,
            distribution={k: [len(v), sum(v), min(v), max(v)] for k, v in counts.items()},
            characters=sum(len(doc.text) for doc in docs),
            buried_line=2613,
        )
        questions = {
            "lexical": NEEDLE,
            "symbol": "epoll_wait sk_buff",
            "noise": (
                "What caused the network queue timeout and what connection limit should I use?"
            ),
            "conflict": (
                "For incident OR-71 on storage-07, what is the production "
                "max_connections policy limit?"
            ),
            "out_of_scope": "How do I configure the flux capacitor on storage-07?",
        }
        if args.out_of_scope_only:
            questions = {"out_of_scope": questions["out_of_scope"]}
        for name, query in questions.items():
            embed, _ = await services.retriever._dense(query)
            bm25, _ = await services.retriever._sparse(query)
            fused = await services.retriever.search(query, 50)
            start = time.monotonic()
            reranked = await retrieve_with_rerank(
                services.retriever, services.inference, settings, query, 10
            )
            lists = {
                "dense": ids(embed),
                "sparse": ids(bm25),
                "fused": [h.chunk.chunk_id for h in fused.hits],
                "rerank": [h.chunk.chunk_id for h in reranked.hits],
            }
            emit(
                "retrieval",
                name=name,
                query=query,
                candidate_counts={k: len(v) for k, v in lists.items()},
                ranks={
                    label: {k: ranking(v, target) for k, v in lists.items()}
                    for label, target in {"needle": needle_ids, **target_ids}.items()
                },
                overlap=len(set(lists["dense"]) & set(lists["sparse"])),
                rrf_top=[h.chunk.doc_id for h in fused.hits[:10]],
                rerank_top=[h.chunk.doc_id for h in reranked.hits],
                rerank_scores=reranked.rerank_scores,
                rerank_skipped=reranked.rerank_skipped,
                search_ms=fused.timings.model_dump(),
                rerank_pipeline_ms=round((time.monotonic() - start) * 1000, 2),
            )
        agent = AgentRunner.from_services(services)
        try:
            for name in (n for n in questions if n not in {"lexical", "symbol"}):
                events: list[dict] = []

                async def collect(event: dict, events=events) -> None:
                    events.append(event)

                start = time.monotonic()
                state = await asyncio.wait_for(
                    agent.run_turn(uuid.uuid4().hex, questions[name], collect), 150
                )
                terminal = events[-1]
                stream = [e for e in events if e["type"] == "response.token"]
                final = [e for e in stream if e["attempt"] == terminal.get("attempt")]
                final_text = "".join(e["text"] for e in final)
                emit(
                    "agent",
                    name=name,
                    provider=agent.reasoner.name,
                    latency_ms=round((time.monotonic() - start) * 1000, 1),
                    answer=state.get("answer"),
                    tool=terminal.get("tool"),
                    outcome=terminal.get("outcome"),
                    retries=terminal.get("retries"),
                    grade=terminal.get("grounding"),
                    relevance=terminal.get("relevance"),
                    terminal_event=terminal.get("type"),
                    abstained=state.get("abstained"),
                    streamed_attempts=len({e["attempt"] for e in stream}),
                    stream_verified=(
                        terminal.get("tokens") == len(final)
                        and terminal.get("sha256")
                        == hashlib.sha256(final_text.encode()).hexdigest()
                        and final_text == state.get("answer")
                    ),
                    source_documents=[
                        source["document_title"]
                        for event in events
                        if event["type"] == "response.sources"
                        for source in event["sources"]
                    ],
                    steps=[e for e in events if e["type"] == "agent.step"],
                    discarded=sum(e["type"] == "agent.generation.discarded" for e in events),
                )
        finally:
            await agent.close()
    finally:
        await services.close()
        await triton.stop()


@contextlib.asynccontextmanager
async def redis_instance():
    binary = shutil.which("redis-server")
    if not binary:
        raise RuntimeError("redis-server must be installed")
    with tempfile.TemporaryDirectory(prefix="omni-stress-redis-") as folder:
        sock_path = str(Path(folder) / "redis.sock")
        proc = subprocess.Popen(
            [
                binary,
                "--port",
                "0",
                "--unixsocket",
                sock_path,
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                folder,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"unix://{sock_path}"
        try:
            async with redis.asyncio.from_url(url) as client:
                async with asyncio.timeout(10):
                    while True:
                        try:
                            await client.ping()
                            break
                        except redis.exceptions.ConnectionError:
                            if proc.poll() is not None:
                                raise RuntimeError("Redis exited before readiness") from None
                            await asyncio.sleep(0.05)
            yield url
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def benchmark_app():
    from backend.main import app as baseline_app
    from backend.main import server_app

    freeze = os.environ["OMNI_STRESS_FREEZE_STARTUP_HEAP"] == "true"
    app = server_app() if freeze else baseline_app

    gc_max_ms: dict[int, float] = {}
    collection_start = 0.0
    profile_path = os.environ.get("OMNI_STRESS_SETUP_PROFILE")
    profiler = cProfile.Profile() if profile_path else None

    def measure_gc(phase: str, info: dict) -> None:
        nonlocal collection_start
        if phase == "start":
            collection_start = time.monotonic()
        else:
            generation = info["generation"]
            duration_ms = (time.monotonic() - collection_start) * 1000
            gc_max_ms[generation] = max(gc_max_ms.get(generation, 0.0), duration_ms)

    gc.callbacks.append(measure_gc)

    @app.post("/__benchmark/reset-lag")
    async def reset_lag() -> None:
        app.state.loop_lag.samples.clear()
        gc_max_ms.clear()
        if profiler is not None:
            profiler.enable()

    @app.get("/__benchmark/setup")
    async def setup_metrics(request: Request) -> dict:
        if profiler is not None and profile_path is not None:
            profiler.disable()
            profiler.dump_stats(profile_path)
        return {
            "websocket": request.app.state.connections.snapshot(),
            "event_loop_lag": request.app.state.loop_lag.snapshot(),
            "gc_max_ms": {str(gen): round(ms, 2) for gen, ms in gc_max_ms.items()},
        }

    @app.websocket("/ws/setup-control")
    async def setup_control(ws: WebSocket) -> None:
        await ws.accept()
        await ws.send_json({"type": "session.created"})
        with contextlib.suppress(WebSocketDisconnect):
            await ws.receive_text()

    return app


@contextlib.asynccontextmanager
async def gateway(redis_url: str, args: argparse.Namespace, profile_name: str = ""):
    # Keep WebSocket client scheduling and frame parsing off the gateway's loop.
    env = {
        **os.environ,
        "REDIS_URL": redis_url,
        "EMBEDDING_PROVIDER": "hashing",
        "INFERENCE_BACKEND": "mock",
        "LLM_PROVIDER": "heuristic",
        "QDRANT_URL": "",
        "ELASTICSEARCH_URL": "",
        "SEED_PATH": "",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "",
        "OTEL_CONSOLE_EXPORT": "false",
        "LOG_LEVEL": "CRITICAL",
        "OMNI_STRESS_FREEZE_STARTUP_HEAP": str(args.freeze_startup_heap).lower(),
    }
    if args.setup_profile_dir:
        env["OMNI_STRESS_SETUP_PROFILE"] = str(args.setup_profile_dir / f"{profile_name}.prof")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = sock.getsockname()[1]
        fd = sock.fileno()
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "scripts.adversarial_stress:benchmark_app",
            "--factory",
            "--fd",
            str(fd),
            "--loop",
            "asyncio",
            "--ws",
            args.ws_protocol,
            "--log-level",
            "critical",
        ]
        proc = subprocess.Popen(command, env=env, pass_fds=(fd,))
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=0.5) as http:
                async with asyncio.timeout(10):
                    while True:
                        if proc.poll() is not None:
                            raise RuntimeError(f"Gateway exited with status {proc.returncode}")
                        with contextlib.suppress(httpx.RequestError):
                            if (await http.get("/readyz")).is_success:
                                break
                        await asyncio.sleep(0.05)
                yield f"ws://127.0.0.1:{port}/ws/agent", http
        finally:
            proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 5)
            except subprocess.TimeoutExpired:
                proc.kill()
                await asyncio.to_thread(proc.wait)


async def frame(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), 30))


async def reset_setup_lag(http: httpx.AsyncClient) -> None:
    await asyncio.sleep(0.2)
    (await http.post("/__benchmark/reset-lag")).raise_for_status()


async def at_setup_barrier(http: httpx.AsyncClient) -> dict:
    # Allow the 10 ms monitor to publish a sample that overlapped the last upgrade.
    await asyncio.sleep(0.02)
    return (await http.get("/__benchmark/setup", timeout=5)).raise_for_status().json()


async def setup_control(base: str, http: httpx.AsyncClient, clients: int) -> None:
    gate = asyncio.Event()
    ready: asyncio.Queue[float] = asyncio.Queue()

    async def client() -> None:
        start = time.monotonic()
        async with connect(base.replace("/ws/agent", "/ws/setup-control")) as ws:
            assert (await frame(ws))["type"] == "session.created"
            await ready.put((time.monotonic() - start) * 1000)
            await gate.wait()

    await reset_setup_lag(http)
    async with asyncio.TaskGroup() as group:
        for _ in range(clients):
            group.create_task(client())
        async with asyncio.timeout(15):
            setup_ms = [await ready.get() for _ in range(clients)]
        barrier = await at_setup_barrier(http)
        gate.set()
    emit(
        "setup_control",
        clients=clients,
        setup_p50_p95_ms=np.percentile(setup_ms, [50, 95]).tolist(),
        lag_at_barrier=barrier["event_loop_lag"],
        setup_gc_max_ms=barrier["gc_max_ms"],
    )


async def race_case(
    base: str, http: httpx.AsyncClient, args: argparse.Namespace, shared: bool, cap: int
) -> None:
    clients = args.clients
    ready = asyncio.Event()
    gate = asyncio.Event()
    joined = 0
    thread = uuid.uuid4().hex
    number = 4000
    replay_ids = {}
    setup_ms = []

    def prompt(n: int) -> str:
        expression = f"{number + 2 * n}/2"
        if args.standalone_calculations:
            return f"Compute the arithmetic quotient of {expression}"
        return expression

    async def client(n: int, retry: bool = False) -> dict:
        nonlocal joined
        url = f"{base}?thread_id={thread}" if shared else base
        connect_start = time.monotonic()
        try:
            async with connect(url) as ws:
                hello = await frame(ws)
                replay_ids[n] = hello["thread_id"]
                assert hello["type"] == "session.created"
                seq = hello["seq"]
                if hello["history_turns"]:
                    history = await frame(ws)
                    assert history["type"] == "agent.history" and history["seq"] == seq + 1
                    seq = history["seq"]
                if not retry:
                    setup_ms.append((time.monotonic() - connect_start) * 1000)
                    joined += 1
                    if joined == clients:
                        ready.set()
                    await gate.wait()
                start = time.monotonic()
                await ws.send(json.dumps({"type": "query", "id": str(n), "text": prompt(n)}))
                tokens = []
                while True:
                    event = await frame(ws)
                    assert event["seq"] == seq + 1
                    seq = event["seq"]
                    if event["type"] == "ping":
                        await ws.send(json.dumps({"type": "pong", "ts": event["ts"]}))
                        continue
                    assert event.get("id") == str(n)
                    if event["type"] == "response.token":
                        assert event["index"] == len(tokens)
                        tokens.append(event["text"])
                    if event["type"] == "response.error":
                        return {
                            "n": n,
                            "error": event.get("code", event["error"]),
                            "retryable": event.get("retryable", False),
                            "partial_tokens": len(tokens),
                        }
                    if event["type"] == "response.end":
                        answer = "".join(tokens)
                        return {
                            "n": n,
                            "integrity": event["tokens"] == len(tokens)
                            and event["sha256"] == hashlib.sha256(answer.encode()).hexdigest(),
                            "correct": event["tool"] == "calculator"
                            and f"= {number // 2 + n}" in answer,
                            "ms": (time.monotonic() - start) * 1000,
                        }
        except Exception as exc:
            return {"n": n, "error": type(exc).__name__}

    await reset_setup_lag(http)
    tasks = asyncio.gather(*(client(n) for n in range(clients)))
    try:
        await asyncio.wait_for(ready.wait(), 15)
    except TimeoutError:
        pass
    barrier = await at_setup_barrier(http)
    active = barrier["websocket"]["active"]
    start = time.monotonic()
    gate.set()
    results = await tasks
    elapsed = time.monotonic() - start
    lag = (await http.get("/health", timeout=5)).raise_for_status().json()["event_loop_lag"]
    successes = [row for row in results if "error" not in row]
    acknowledged = {prompt(result["n"]) for result in successes}

    async def history() -> list[str]:
        replayed = []
        for replay_thread in set(replay_ids.values()):
            async with connect(f"{base}?thread_id={replay_thread}") as ws:
                hello = await frame(ws)
                if hello["history_turns"]:
                    replay = await frame(ws)
                    assert replay["type"] == "agent.history" and replay["seq"] == hello["seq"] + 1
                    replayed.extend(m["content"] for m in replay["messages"] if m["role"] == "user")
        return replayed

    replayed = await history()
    replay_set = set(replayed)
    emit(
        "race",
        shared_thread=shared,
        pool_cap=cap,
        clients=clients,
        active_at_barrier=active,
        setup_p50_p95_ms=np.percentile(setup_ms, [50, 95]).tolist() if setup_ms else [],
        response_end=len(successes),
        integrity_errors=sum(not r["integrity"] for r in successes),
        incorrect=sum(not r["correct"] for r in successes),
        errors=dict(Counter(row["error"] for row in results if "error" in row)),
        retryable_errors=sum(row.get("retryable", False) for row in results),
        error_token_frames=sum(row.get("partial_tokens", 0) for row in results),
        replayed_turns=len(replayed),
        acknowledged_missing=len(acknowledged - replay_set),
        unacknowledged_replayed=len(replay_set - acknowledged),
        replay_duplicates=len(replayed) - len(replay_set),
        latency_p50_p95_ms=np.percentile([r["ms"] for r in successes], [50, 95]).tolist()
        if successes
        else [],
        wall_s=elapsed,
        lag_at_barrier=barrier["event_loop_lag"],
        setup_gc_max_ms=barrier["gc_max_ms"],
        lag=lag,
    )
    if shared and args.retry_busy:
        deferred = [row["n"] for row in results if row.get("error") == "thread_busy"]
        retries = [await client(n, retry=True) for n in deferred]
        replayed = await history()
        emit(
            "retry",
            pool_cap=cap,
            attempted=len(deferred),
            acknowledged=sum("error" not in row for row in retries),
            verified=sum(row.get("integrity", False) for row in retries),
            correct=sum(row.get("correct", False) for row in retries),
            exact_history=Counter(replayed) == Counter(prompt(n) for n in range(clients)),
            replayed_turns=len(replayed),
        )


async def race_probes(args: argparse.Namespace) -> None:
    async with redis_instance() as url:
        if args.setup_control:
            async with gateway(url, args, "minimal") as (base, http):
                await setup_control(base, http, args.clients)
        for cap in args.pool_caps:
            for shared in (False, True):
                layout = "shared" if shared else "separate"
                async with gateway(f"{url}?max_connections={cap}", args, f"{layout}-{cap}") as (
                    base,
                    http,
                ):
                    await race_case(base, http, args, shared, cap)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("search", "race"))
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--pool-caps", type=int, nargs="+", default=[100, 256, 512])
    parser.add_argument("--retry-busy", action="store_true")
    parser.add_argument("--setup-control", action="store_true")
    parser.add_argument(
        "--setup-profile-dir", type=Path, help="Save per-layout setup call profiles."
    )
    parser.add_argument(
        "--freeze-startup-heap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the worker startup GC optimization. Disable to measure the original gateway.",
    )
    parser.add_argument(
        "--ws-protocol",
        choices=("auto", "websockets", "websockets-sansio", "wsproto"),
        default="auto",
    )
    parser.add_argument("--standalone-calculations", action="store_true")
    parser.add_argument("--out-of-scope-only", action="store_true")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--elasticsearch-url")
    parser.add_argument("--model")
    parser.add_argument("--model-url", default="http://127.0.0.1:11434/v1")
    args = parser.parse_args()
    if args.clients < 1:
        parser.error("--clients must be positive")
    if any(cap < 1 for cap in args.pool_caps):
        parser.error("--pool-caps must be positive")
    if args.setup_profile_dir:
        args.setup_profile_dir.mkdir(parents=True, exist_ok=True)
    for url in (args.qdrant_url, args.elasticsearch_url, args.model_url):
        if url and urlparse(url).hostname not in {"localhost", "127.0.0.1", "::1"}:
            parser.error("audit services must be on loopback")
    asyncio.run(search_probes(args) if args.suite == "search" else race_probes(args))


if __name__ == "__main__":
    main()
