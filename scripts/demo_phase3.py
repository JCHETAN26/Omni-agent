"""Phase 3 demo: multi-turn agentic RAG with self-correction and Redis memory.

Part A (real stack, offline reasoner):
  real redis-server -> RedisCheckpointSaver; real app under uvicorn; /ws/agent;
  MiniLM embeddings + BM25 hybrid retrieval; HeuristicReasoner (no LLM).
  The app is restarted between sessions, so memory can only come from Redis.

Part B (grounding self-correction):
  the real LLMReasoner prompts/parsing and graph loop, driven by a SIMULATED LLM
  that hallucinates on its first answer. Use --llm anthropic to run Part B against
  Claude instead (needs ANTHROPIC_API_KEY).

    python -m scripts.demo_phase3 [--llm anthropic]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import re
import shutil
import socket
import subprocess
import tempfile
import textwrap

from websockets.asyncio.client import connect

from backend.agent.llm import LLMError
from backend.agent.reasoning import LLMReasoner
from backend.agent.runner import AgentRunner
from backend.core.services import Services
from tests.conftest import live_server, load_samples, make_settings

W = 100


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def redis_server():
    exe = shutil.which("redis-server")
    if not exe:
        raise SystemExit("redis-server not found (brew install redis / apt install redis)")
    port = free_port()
    tmp = tempfile.mkdtemp(prefix="omni-redis-")
    proc = subprocess.Popen(
        [exe, "--port", str(port), "--dir", tmp, "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.1):
                break
        yield f"redis://127.0.0.1:{port}/0", port
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        shutil.rmtree(tmp, ignore_errors=True)


def fmt_step(ev: dict) -> str:
    n = ev["node"]
    fb = "  [heuristic fallback]" if ev.get("fallback") else ""
    if n == "contextualize":
        return (
            f"contextualize     standalone={ev['standalone']!r}"
            if ev["changed"]
            else "contextualize     (already standalone)"
        ) + fb
    if n == "route":
        arg = f" arg={ev['argument']!r}" if ev["tool"] != "knowledge_base" else ""
        return f"route             -> {ev['tool']}{arg}  ({ev['reason']}){fb}"
    if n == "retrieve":
        hits = ", ".join(h.split(" > ")[-1] for h in ev["hits"][:3])
        return (
            f"retrieve #{ev['attempt']}       query={ev['query']!r}\n" + " " * 22 + f"top: {hits}"
        )
    if n == "grade_documents":
        mark = "relevant" if ev["relevant"] else "NOT relevant"
        return f"grade_documents   {mark} score={ev['score']:.2f}  ({ev['reason']}){fb}"
    if n == "rewrite_query":
        return (
            f"rewrite_query #{ev['retry']}   [{ev['strategy']}] {ev['query']!r}\n"
            + " " * 22
            + f"because: {ev['because'][:70]}{fb}"
        )
    if n == "inspect_logs":
        return f"inspect_logs      host={ev['host']} lines={ev['lines']}"
    if n == "calculate":
        return f"calculate         {ev['result']}"
    if n == "generate":
        flags = " insufficient-context" if ev["insufficient_context"] else ""
        flags += " abstained" if ev["abstained"] else ""
        return f"generate #{ev['attempt']}       {ev['chars']} chars{flags}{fb}"
    if n == "grade_generation":
        mark = "grounded" if ev["grounded"] else "NOT grounded"
        tail = "  -> retry" if ev["will_retry"] else ""
        return f"grade_generation  {mark} score={ev['score']:.2f} ({ev['reason']}){tail}{fb}"
    if n == "finalize":
        return f"finalize          outcome={ev['outcome']} retries={ev['retries']}"
    return f"{n} {ev}"


def show_turn(question: str, events: list[dict]) -> None:
    print(f"\nUSER  > {question}")
    for ev in events:
        if ev["type"] == "agent.step":
            print("   ·  " + fmt_step(ev))
        elif ev["type"] == "agent.generation.discarded":
            print(f"   x  discarded attempt {ev['attempt']}: {ev['reason']}")
        elif ev["type"] == "agent.warning":
            print(f"   !  {ev['message']}")
    end = events[-1]
    final = [e for e in events if e["type"] == "response.token" and e["attempt"] == end["attempt"]]
    text = "".join(e["text"] for e in final)
    ok = hashlib.sha256(text.encode()).hexdigest() == end["sha256"] and end["tokens"] == len(final)
    body = textwrap.indent(textwrap.fill(text, W - 8) if "\n" not in text else text, " " * 8)
    print(
        f"AGENT > ({end['outcome']}, tool={end['tool']}, retries={end['retries']}, "
        f"{end['tokens']} tokens, checksum {'OK' if ok else 'MISMATCH'}, "
        f"{end['latency_ms']:.0f} ms)"
    )
    print(body)


async def ask(ws, rid: str, question: str) -> list[dict]:
    await ws.send(json.dumps({"type": "query", "id": rid, "text": question}))
    events = []
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
        if ev["type"] == "ping":
            await ws.send(json.dumps({"type": "pong", "ts": ev["ts"]}))
            continue
        if ev.get("id") != rid:
            continue
        events.append(ev)
        if ev["type"] in ("response.end", "response.error"):
            return events


async def part_a(redis_url: str, redis_port: int) -> None:
    print("=" * W)
    print("PART A  /ws/agent  |  real Redis checkpointer  |  heuristic reasoner (no LLM)")
    print("=" * W)
    cfg = dict(embedding_provider="sentence-transformers", redis_url=redis_url)

    async with live_server(**cfg) as (ws_url, _, app):
        url = ws_url.replace("/ws/stream", "/ws/agent")
        print(f"app #1 up  memory={type(app.state.agent.checkpointer).__name__}")
        async with connect(url) as ws:
            hello = json.loads(await ws.recv())
            thread = hello["thread_id"]
            print(f"[session 1] thread_id={thread} resumed={hello['resumed']}")
            for i, q in enumerate(
                [
                    "hey so my storage box keeps spitting out EIO5 over and over, any idea whats "
                    "going on??",
                    "and which firmware version is affected?",
                ]
            ):
                show_turn(q, await ask(ws, f"a{i}", q))
        print("\n[session 1 closed]  app #1 shutting down ...")

    keys = subprocess.run(
        ["redis-cli", "-p", str(redis_port), "--scan", "--pattern", "omni:lg:*"],
        capture_output=True,
        text=True,
    ).stdout.split()
    print(f"redis now holds {len(keys)} checkpoint keys for the conversation")

    async with live_server(**cfg) as (ws_url, _, app):
        url = ws_url.replace("/ws/stream", "/ws/agent")
        print("app #2 up (fresh process state)")
        async with connect(f"{url}?thread_id={thread}") as ws:
            hello = json.loads(await ws.recv())
            hist = json.loads(await ws.recv())
            print(
                f"[session 2] thread_id={thread} resumed={hello['resumed']} "
                f"history_turns={hello['history_turns']}"
            )
            for m in hist["messages"]:
                print(f"   history  {m['role']:<9} {m['content'][:80]}")
            for i, q in enumerate(
                [
                    "show me recent errors on storage-07",
                    "how do I rotate the kafka broker TLS keystore?",
                    "if the queue depth is 64 what's 64/2?",
                ]
            ):
                show_turn(q, await ask(ws, f"b{i}", q))


class SimulatedLLM:
    """Stands in for Claude in Part B. Answers through the real LLMReasoner code path:
    structured decisions are returned as schema-valid JSON, answers are streamed. The
    first answer contains an invented firmware version, to trigger the grounding loop."""

    name = "simulated-llm"

    def __init__(self) -> None:
        self.answers = 0

    async def complete_json(self, system, messages, schema, effort="low"):
        content = messages[-1]["content"]
        if schema.__name__ == "RouteDecision":
            return schema(
                tool="knowledge_base",
                argument="EIO-5 controller timeout fix firmware",
                reason="error-code troubleshooting question",
            )
        if schema.__name__ == "RelevanceGrade":
            ok = "EIO-5" in content
            return schema(
                score=0.92 if ok else 0.1,
                relevant=ok,
                reason="passage [1] covers EIO-5 cause and workaround",
            )
        if schema.__name__ == "Rewrite":
            return schema(
                query="EIO-5 timeout firmware 4.2.1 queue_depth workaround",
                strategy="anchor_on_source_terms",
            )
        if schema.__name__ == "GroundingGrade":
            bad = "6.0.3" in content.split("Answer:", 1)[-1]
            return schema(
                score=0.33 if bad else 1.0,
                grounded=not bad,
                unsupported_claims=["Upgrading to firmware 6.0.3 fixes it permanently."]
                if bad
                else [],
                reason="checked each claim against [1]",
            )
        raise LLMError(f"unexpected schema {schema.__name__}")

    async def stream_text(self, system, messages, effort="medium"):
        self.answers += 1
        if self.answers == 1:
            text = (
                "EIO-5 is a controller timeout on block writes [1]. Upgrading to firmware "
                "6.0.3 fixes it permanently."
            )
        else:
            m = re.search(r"Firmware below ([\d.]+)", messages[-1]["content"])
            text = (
                f"EIO-5 means the controller timed out waiting for a block write [1]. Firmware "
                f"below {m.group(1) if m else '?'} has a known bug when queue depth exceeds "
                f"64; set queue_depth=32 as a workaround and check cable seating [1]."
            )
        for word in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0)
            yield word

    async def close(self) -> None:
        pass


async def part_b(llm: str) -> None:
    print("\n" + "=" * W)
    title = "Claude (" + llm + ")" if llm != "simulated" else "SIMULATED LLM (no API key)"
    print(f"PART B  grounding self-correction  |  LLMReasoner + {title}")
    print("=" * W)
    services = await Services.build(make_settings(embedding_provider="sentence-transformers"))
    await services.processor.ingest(load_samples())
    if llm == "simulated":
        model = SimulatedLLM()
    else:
        from backend.agent.llm_anthropic import AnthropicChatModel

        model = AnthropicChatModel()
    from langgraph.checkpoint.memory import InMemorySaver

    agent = AgentRunner(services, LLMReasoner(model), InMemorySaver(), model)
    q = "our storage node is throwing EIO-5, what's the permanent fix?"
    events: list[dict] = []

    async def emit(ev):
        events.append(ev)

    await agent.run_turn("demo-grounding-01", q, emit, request_id="g1")
    for ev in events:
        if ev["type"] == "response.token" and ev["index"] == 0:
            attempt_text = "".join(
                e["text"]
                for e in events
                if e["type"] == "response.token" and e["attempt"] == ev["attempt"]
            )
            print(f"   ~  streamed attempt {ev['attempt']}: {attempt_text.strip()[:110]}")
    show_turn(q, events)
    await agent.close()
    await services.close()


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", choices=["simulated", "anthropic"], default="simulated")
    args = p.parse_args()
    with redis_server() as (url, port):
        await part_a(url, port)
    await part_b(args.llm)


if __name__ == "__main__":
    import logging

    logging.disable(logging.WARNING)
    asyncio.run(main())
