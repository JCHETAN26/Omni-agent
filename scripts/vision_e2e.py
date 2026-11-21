"""Live end-to-end check: frames on /ws/frames reach the agent's next turn.

    python -m scripts.vision_e2e --server http://127.0.0.1:8080 [--output e2e.json]

1. Streams JPEG frames on /ws/frames?thread_id=T and records the labels Triton
   returned.
2. Asks a question on /sse/agent with the same thread_id and checks that the
   turn carries a `visual_context` step whose observation names those labels
   (the text the agent appends to the question it answers).
3. Control: the same question on a fresh thread must have no visual context.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from urllib.parse import urlencode
from urllib.request import urlopen

from websockets.asyncio.client import connect

from benchmarks.vision_preprocess import make_frames


async def send_frames(ws_url: str, frames: list[bytes]) -> list[dict]:
    results = []
    async with connect(ws_url, max_size=2**22, ping_interval=None) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "session.created", hello
        for frame in frames:
            await ws.send(frame)
            results.append(json.loads(await ws.recv()))
    return results


def agent_turn(server: str, question: str, thread: str) -> dict:
    """One /sse/agent turn: its steps and the final answer text."""
    query = urlencode({"q": question, "thread_id": thread})
    steps, tokens, end = [], [], None
    with urlopen(f"{server}/sse/agent?{query}", timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event["type"] == "agent.step":
                steps.append(event)
            elif event["type"] == "response.token":
                tokens.append(event["text"])
            elif event["type"] == "response.end":
                end = event
            elif event["type"] == "response.error":
                raise RuntimeError(event.get("error"))
    visual = [s for s in steps if s.get("node") == "visual_context"]
    return {
        "answer": "".join(tokens),
        "outcome": end and end.get("outcome"),
        "nodes": [s.get("node") for s in steps],
        "visual_context": visual[0]["observation"] if visual else None,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--question", default="What can you see on my camera right now?")
    p.add_argument("--output")
    args = p.parse_args()

    thread = f"e2e-{uuid.uuid4().hex[:12]}"
    ws_url = args.server.replace("http", "ws", 1) + f"/ws/frames?thread_id={thread}"
    results = asyncio.run(send_frames(ws_url, make_frames(720, 1280, args.frames)))
    ok_frames = [r for r in results if r["type"] == "frame.result"]
    top1 = ok_frames[-1]["labels"][0]["label"] if ok_frames else None

    turn = agent_turn(args.server, args.question, thread)
    control = agent_turn(args.server, args.question, f"e2e-{uuid.uuid4().hex[:12]}")

    checks = {
        "all_frames_classified": len(ok_frames) == len(results) > 0,
        "turn_has_visual_context": bool(turn["visual_context"]),
        "visual_context_names_top1_label": bool(
            top1 and turn["visual_context"] and top1 in turn["visual_context"]
        ),
        "visual_context_counts_frames": bool(
            turn["visual_context"] and f"frame #{len(results)}" in turn["visual_context"]
        ),
        "fresh_thread_has_no_visual_context": control["visual_context"] is None,
    }
    report = {
        "server": args.server,
        "thread_id": thread,
        "frames": [{k: r.get(k) for k in ("type", "seq", "labels", "timings")} for r in results],
        "turn": turn,
        "control_turn": control,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
