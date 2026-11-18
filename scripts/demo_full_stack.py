"""One-command full-stack demo: boot every service, drive concurrent agent users,
print a live health/performance line and a final summary.

    python -m scripts.demo_full_stack                 # services + simulated users, then exit
    python -m scripts.demo_full_stack --ui --serve    # also build/serve the UI and keep running
    python -m scripts.demo_full_stack --users 24 --rerank

Services (separate processes, real sockets):
  redis-server          conversation memory (RedisCheckpointSaver) + rate limits
  local Triton server   ONNX MiniLM embedder + cross-encoder over Triton gRPC (if exported:
                        `python -m model_server.export_onnx`), else in-process embeddings
  FastAPI backend       /ws/agent, seeded knowledge base, Prometheus /metrics
  UI (--ui)             production build served by `vite preview`, proxied to the backend

Qdrant and Elasticsearch use their in-process fallbacks here; Kubernetes runs the real ones.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from websockets.asyncio.client import connect

from scripts.demo_phase3 import fmt_step, free_port

ROOT = Path(__file__).resolve().parent.parent
ONNX = ROOT / "model_server/model_repository/minilm_embedder/1/model.onnx"
W = 112

SCRIPTS: dict[str, list[str | None]] = {
    # None = disconnect and resume the same thread on a new connection
    "storage-incident": [
        "hey so my storage box keeps spitting out EIO5 over and over, any idea whats going on??",
        "and which firmware version is affected?",
        None,
        "show me recent errors on storage-07",
        "if the queue depth is 64 what's 64/2?",
    ],
    "out-of-scope": [
        "how do I rotate the kafka broker TLS keystore?",
        "ok, what does a PSU redundancy lost alert mean?",
        "thanks!",
    ],
    "network": [
        "edge router dropping packets intermittently, could it be MTU?",
        "what overlay header size should I account for?",
        "show me recent warnings on edge-02",
    ],
}


# ----------------------------------------------------------------------------
# Process management
# ----------------------------------------------------------------------------


class Stack:
    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.procs: list[tuple[str, subprocess.Popen]] = []

    def start(
        self, name: str, cmd: list[str], env: dict | None = None, cwd: Path = ROOT, stdout=None
    ) -> subprocess.Popen:
        log = open(self.workdir / f"{name}.log", "w")  # noqa: SIM115 - closed with process
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=stdout or log,
            stderr=subprocess.STDOUT if not stdout else log,
            start_new_session=True,
            text=True,
        )
        self.procs.append((name, proc))
        return proc

    def stop(self) -> None:
        for _name, proc in reversed(self.procs):
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
        for _name, proc in self.procs:
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=10)


async def wait_http(url: str, timeout_s: float, proc: subprocess.Popen, name: str) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2) as http:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{name} exited early (code {proc.returncode})")
            with contextlib.suppress(httpx.HTTPError):
                if (await http.get(url)).status_code == 200:
                    return
            await asyncio.sleep(0.25)
    raise RuntimeError(f"{name} not ready after {timeout_s}s ({url})")


async def boot(args, stack: Stack) -> dict:
    info: dict = {}
    t0 = time.perf_counter()

    redis_url = ""
    if shutil.which("redis-server"):
        port = free_port()
        stack.start(
            "redis",
            [
                "redis-server",
                "--port",
                str(port),
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(stack.workdir),
            ],
        )
        redis_url = f"redis://127.0.0.1:{port}/0"
        for _ in range(100):
            with contextlib.suppress(OSError):
                import socket

                socket.create_connection(("127.0.0.1", port), 0.1).close()
                break
            await asyncio.sleep(0.05)
    info["redis"] = redis_url or "not installed: in-process memory"

    env = {
        "SEED_PATH": "samples",
        "LOG_FORMAT": "json",
        "LOG_LEVEL": "WARNING",
        "QDRANT_URL": "",
        "ELASTICSEARCH_URL": "",
        "REDIS_URL": redis_url,
        "RATE_LIMIT_CAPACITY": "1000",
        "WS_MAX_SESSIONS": "5000",
        "RERANK_ENABLED": "true" if args.rerank else "false",
        "LLM_PROVIDER": args.llm,
    }
    if ONNX.exists():
        tport = free_port()
        proc = stack.start(
            "triton", [sys.executable, "-m", "model_server.local_triton", "--port", str(tport)]
        )
        for _ in range(600):
            if "READY" in (stack.workdir / "triton.log").read_text():
                break
            if proc.poll() is not None:
                raise RuntimeError("local triton exited early")
            await asyncio.sleep(0.1)
        env |= {
            "INFERENCE_BACKEND": "triton",
            "EMBEDDING_PROVIDER": "triton",
            "TRITON_GRPC_URL": f"127.0.0.1:{tport}",
        }
        info["inference"] = f"local Triton-protocol server (ONNX) :{tport}"
    else:
        env |= {"INFERENCE_BACKEND": "mock", "EMBEDDING_PROVIDER": "sentence-transformers"}
        info["inference"] = "in-process MiniLM (export ONNX models to use Triton)"

    bport = free_port()
    proc = stack.start(
        "backend",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(bport),
        ],
        env,
    )
    await wait_http(f"http://127.0.0.1:{bport}/readyz", 180, proc, "backend")
    info["backend"] = f"http://127.0.0.1:{bport}"

    if args.ui:
        fe = ROOT / "frontend"
        build = subprocess.run(["npm", "run", "build"], cwd=fe, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(f"frontend build failed:\n{build.stdout[-2000:]}")
        uport = free_port()
        proc = stack.start(
            "ui",
            ["npx", "vite", "preview", "--host", "127.0.0.1", "--port", str(uport), "--strictPort"],
            {"OMNI_BACKEND_URL": info["backend"]},
            cwd=fe,
        )
        await wait_http(f"http://127.0.0.1:{uport}/", 60, proc, "ui")
        info["ui"] = f"http://127.0.0.1:{uport}"
    info["boot_s"] = round(time.perf_counter() - t0, 1)
    return info


# ----------------------------------------------------------------------------
# Simulated users
# ----------------------------------------------------------------------------


@dataclass
class TurnResult:
    user: int
    script: str
    question: str
    tool: str = ""
    outcome: str = ""
    retries: int = 0
    discarded: int = 0
    ttft_ms: float | None = None
    latency_ms: float = 0.0
    tokens: int = 0
    integrity_ok: bool = False
    steps: list[dict] = field(default_factory=list)
    answer: str = ""
    error: str | None = None


@dataclass
class Run:
    results: list[TurnResult] = field(default_factory=list)
    seq_gaps: int = 0
    resumes_ok: int = 0
    resumes: int = 0
    users_done: int = 0
    users_total: int = 0
    turns_total: int = 0
    tokens_window: list[tuple[float, int]] = field(default_factory=list)


async def run_user(n: int, base_ws: str, run: Run, think_s: tuple[float, float]) -> None:
    name = list(SCRIPTS)[n % len(SCRIPTS)]
    script = SCRIPTS[name]
    thread: str | None = None
    rng = random.Random(n)
    await asyncio.sleep(n * 0.05)  # staggered arrival
    turns = iter(enumerate(script))
    while True:
        url = f"{base_ws}/ws/agent" + (f"?thread_id={thread}" if thread else "")
        async with connect(url, max_size=2**20) as ws:
            hello = json.loads(await ws.recv())
            last_seq = hello["seq"]
            if thread:
                run.resumes += 1
                hist = json.loads(await ws.recv())
                last_seq = hist["seq"]
                if (
                    hist["type"] == "agent.history"
                    and len(hist["messages"]) == hello["history_turns"] * 2 > 0
                ):
                    run.resumes_ok += 1
            thread = hello["thread_id"]
            for i, question in turns:
                if question is None:
                    break  # disconnect; resume on a new connection
                res = TurnResult(user=n, script=name, question=question)
                rid = f"u{n}t{i}"
                t0 = time.perf_counter()
                await ws.send(json.dumps({"type": "query", "id": rid, "text": question}))
                final: dict[int, list[str]] = {}
                while True:
                    ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
                    if ev["seq"] != last_seq + 1:
                        run.seq_gaps += 1
                    last_seq = ev["seq"]
                    t = ev["type"]
                    if t == "ping":
                        await ws.send(json.dumps({"type": "pong", "ts": ev["ts"]}))
                    elif t == "response.token":
                        res.ttft_ms = res.ttft_ms or (time.perf_counter() - t0) * 1000
                        final.setdefault(ev["attempt"], []).append(ev["text"])
                        run.tokens_window.append((time.monotonic(), 1))
                    elif t == "agent.step":
                        res.steps.append(ev)
                    elif t == "agent.generation.discarded":
                        res.discarded += 1
                    elif t == "response.error":
                        res.error = ev["error"]
                        break
                    elif t == "response.end":
                        parts = final.get(ev["attempt"], [])
                        res.answer = "".join(parts)
                        res.integrity_ok = (
                            len(parts) == ev["tokens"]
                            and hashlib.sha256(res.answer.encode()).hexdigest() == ev["sha256"]
                        )
                        res.tool, res.outcome = ev["tool"], ev["outcome"]
                        res.retries, res.tokens = ev["retries"], ev["tokens"]
                        break
                res.latency_ms = (time.perf_counter() - t0) * 1000
                run.results.append(res)
                await asyncio.sleep(rng.uniform(*think_s))
            else:
                break  # script finished
    run.users_done += 1


def pct(values, p) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


async def monitor(run: Run, backend: str, t0: float, stop: asyncio.Event) -> None:
    async with httpx.AsyncClient(timeout=2) as http:
        while not stop.is_set():
            await asyncio.sleep(1.0)
            now = time.monotonic()
            run.tokens_window = [(t, c) for t, c in run.tokens_window if now - t < 3]
            tps = sum(c for _, c in run.tokens_window) / 3
            done = run.results
            ttft = [r.ttft_ms for r in done if r.ttft_ms]
            lat = [r.latency_ms for r in done]
            outcomes: dict[str, int] = {}
            for r in done:
                outcomes[r.outcome or "error"] = outcomes.get(r.outcome or "error", 0) + 1
            try:
                health = (await http.get(f"{backend}/health")).json()
                ws_active = health["websocket"]["active"]
                lag = health["event_loop_lag"].get("p99_ms", 0)
            except (httpx.HTTPError, KeyError, ValueError):
                ws_active, lag = "?", float("nan")
            ok = sum(r.integrity_ok for r in done)
            print(
                f"  t={now - t0:5.1f}s  users {run.users_done:>3}/{run.users_total} done"
                f"  sessions {ws_active!s:>3}  turns {len(done):>3}/{run.turns_total}"
                f"  | {' '.join(f'{k}={v}' for k, v in sorted(outcomes.items()))}"
                f"  | rewrites {sum(r.retries for r in done)}"
                f"  | TTFT p50 {pct(ttft, 50):4.0f} p95 {pct(ttft, 95):4.0f} ms"
                f"  lat p95 {pct(lat, 95):5.0f} ms  {tps:6.0f} tok/s"
                f"  | verified {ok}/{len(done)}  loop-lag p99 {lag} ms",
                flush=True,
            )


def scrape(metrics_text: str, name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if line.startswith(name):
            m = re.match(rf"{name}(\{{[^}}]*\}})?\s+([0-9.eE+-]+)$", line)
            if m:
                out[m.group(1) or ""] = float(m.group(2))
    return out


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--users", type=int, default=12)
    p.add_argument("--ui", action="store_true", help="build and serve the React UI")
    p.add_argument("--serve", action="store_true", help="keep services running afterwards")
    p.add_argument("--rerank", action="store_true", help="enable cross-encoder rerank")
    p.add_argument("--llm", choices=["heuristic", "anthropic", "openai"], default="heuristic")
    p.add_argument("--think", type=float, nargs=2, default=(0.2, 0.6), metavar=("MIN", "MAX"))
    args = p.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="omniagent-demo-"))
    stack = Stack(workdir)
    try:
        print("=" * W)
        print("OmniAgent full-stack demo")
        print("=" * W)
        info = await boot(args, stack)
        for k in ("redis", "inference", "backend", "ui"):
            if k in info:
                print(f"  {k:<10} {info[k]}")
        print(f"  {'reasoner':<10} {args.llm}")
        print(f"  {'logs':<10} {workdir}")
        print(f"  booted in {info['boot_s']}s\n")

        run = Run(users_total=args.users)
        run.turns_total = sum(
            sum(q is not None for q in SCRIPTS[list(SCRIPTS)[n % len(SCRIPTS)]])
            for n in range(args.users)
        )
        print(
            f"[load] {args.users} concurrent users x multi-turn scripts {list(SCRIPTS)}"
            f" ({run.turns_total} turns, one mid-conversation reconnect per storage user)"
        )
        base_ws = info["backend"].replace("http", "ws")
        t0 = time.monotonic()
        stop = asyncio.Event()
        mon = asyncio.create_task(monitor(run, info["backend"], t0, stop))
        await asyncio.gather(
            *(run_user(n, base_ws, run, tuple(args.think)) for n in range(args.users))
        )
        wall = time.monotonic() - t0
        stop.set()
        await mon
        await summarize(run, info, wall, stack)

        if args.serve:
            print(
                f"\nServices are running. UI: {info.get('ui', '(start with --ui)')}  "
                f"API: {info['backend']}  - Ctrl-C to stop."
            )
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.Event().wait()
    finally:
        stack.stop()


async def summarize(run: Run, info: dict, wall: float, stack: Stack) -> None:
    res = run.results
    ok = [r for r in res if not r.error]
    print("\n" + "=" * W)
    print("SUMMARY")
    print("=" * W)
    print(
        f"  turns {len(res)}  errors {len(res) - len(ok)}  wall {wall:.1f}s  "
        f"throughput {len(res) / wall:.1f} turns/s"
    )
    print(
        f"  stream integrity: {sum(r.integrity_ok for r in res)}/{len(res)} checksums verified, "
        f"{run.seq_gaps} sequence gaps"
    )
    print(f"  memory: {run.resumes_ok}/{run.resumes} reconnects resumed with full history")
    ttft = [r.ttft_ms for r in ok if r.ttft_ms]
    lat = [r.latency_ms for r in ok]
    print(
        f"  TTFT    p50 {pct(ttft, 50):6.1f} ms   p95 {pct(ttft, 95):6.1f} ms   "
        f"p99 {pct(ttft, 99):6.1f} ms"
    )
    print(
        f"  latency p50 {pct(lat, 50):6.1f} ms   p95 {pct(lat, 95):6.1f} ms   "
        f"p99 {pct(lat, 99):6.1f} ms"
    )
    print("\n  by tool / outcome:")
    table: dict[tuple[str, str], list[TurnResult]] = {}
    for r in ok:
        table.setdefault((r.tool, r.outcome), []).append(r)
    for (tool, outcome), rows in sorted(table.items()):
        print(
            f"    {tool:<15} {outcome:<21} {len(rows):>4} turns   "
            f"avg rewrites {sum(r.retries for r in rows) / len(rows):.1f}   "
            f"p50 latency {pct([r.latency_ms for r in rows], 50):6.1f} ms"
        )
    corrected = [r for r in ok if r.retries]
    print(
        f"\n  self-correction: {len(corrected)} turns rewrote their query "
        f"({sum(r.retries for r in corrected)} rewrites, {sum(r.discarded for r in ok)} "
        f"answers discarded by the grounding check)"
    )

    async with httpx.AsyncClient(timeout=5) as http:
        health = (await http.get(f"{info['backend']}/health")).json()
        metrics_text = (await http.get(f"{info['backend']}/metrics")).text
    turns = scrape(metrics_text, "omniagent_agent_turns_total")
    tokens = scrape(metrics_text, "omniagent_tokens_streamed_total")
    triton = scrape(metrics_text, "omniagent_triton_request_seconds_count")
    triton_sum = scrape(metrics_text, "omniagent_triton_request_seconds_sum")
    print("\n  server-side (/health, /metrics):")
    print(
        f"    event-loop lag p50 {health['event_loop_lag'].get('p50_ms')} ms  "
        f"p99 {health['event_loop_lag'].get('p99_ms')} ms  "
        f"max {health['event_loop_lag'].get('max_ms')} ms"
    )
    print(
        f"    sessions accepted {health['websocket']['total_accepted']}  "
        f"rejected {health['websocket']['total_rejected']}  backends {health['backends']}"
    )
    print(
        f"    agent turns counted {int(sum(turns.values()))}  tokens streamed "
        f"{int(sum(tokens.values()))}"
    )
    for labels, count in triton.items():
        mean = triton_sum.get(labels, 0) / count * 1000 if count else 0
        print(f"    triton {labels}: {int(count)} calls, mean {mean:.1f} ms")
    if shutil.which("redis-cli") and info["redis"].startswith("redis://"):
        port = info["redis"].rsplit(":", 1)[1].split("/")[0]
        keys = subprocess.run(
            ["redis-cli", "-p", port, "--scan", "--pattern", "omni:lg:*"],
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        print(
            f"    redis: {len(keys)} checkpoint keys across "
            f"{len({k.split(':')[2] for k in keys})} conversation threads"
        )

    example = next((r for r in ok if r.retries), None)
    if example:
        print(f"\n  example self-correcting turn (user {example.user}, {example.script}):")
        print(f"    USER  > {example.question}")
        for step in example.steps:
            print("      ·  " + fmt_step(step).replace("\n", "\n         "))
        print(f"    AGENT > [{example.outcome}] {example.answer[:140]}")
    print(f"\n  service logs: {stack.workdir}")


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
