"""Closed-loop live-frame load test against /ws/frames.

    python -m benchmarks.multimodal --url ws://127.0.0.1:8080/ws/frames \
        --sessions 1 4 8 16 32 --duration 30 --label batching-on --output on.json

Each session is one WebSocket that sends a JPEG frame, waits for its result,
then sends the next (closed loop). Frame-to-response latency is measured at
the client, from just before the frame is sent to the parsed reply. With
--triton-metrics (Triton's :8002/metrics), it also reports how many requests
Triton merged per model execution (average batch size) during each level.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import re
import time

import httpx
import numpy as np
from websockets.asyncio.client import connect

from benchmarks.vision_preprocess import make_frames


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))], 2)


async def triton_counts(url: str | None, model: str) -> tuple[float, float] | None:
    """(requests, executions) for `model` from Triton's Prometheus metrics."""
    if not url:
        return None
    async with httpx.AsyncClient(timeout=5) as client:
        text = (await client.get(url)).text

    def value(metric: str) -> float:
        pattern = rf'^{metric}{{[^}}]*model="{re.escape(model)}"[^}}]*}} ([0-9.e+]+)$'
        return sum(float(m) for m in re.findall(pattern, text, re.M))

    return value("nv_inference_request_success"), value("nv_inference_exec_count")


async def session(url: str, frames: list[bytes], stop: float, out: dict, offset: int) -> None:
    async with connect(url, max_size=2**22, ping_interval=None) as ws:
        hello = json.loads(await ws.recv())
        out.setdefault("server", hello)
        i = offset
        while time.perf_counter() < stop:
            frame = frames[i % len(frames)]
            i += 1
            t = time.perf_counter()
            await ws.send(frame)
            reply = json.loads(await ws.recv())
            elapsed = (time.perf_counter() - t) * 1000
            if reply["type"] == "frame.result":
                out["latency"].append(elapsed)
                out["preprocess"].append(reply["timings"]["preprocess_ms"])
                out["infer"].append(reply["timings"]["infer_ms"])
                out["server"].append(reply["timings"]["server_ms"])
                out["top1"][reply["labels"][0]["label"]] = (
                    out["top1"].get(reply["labels"][0]["label"], 0) + 1
                )
            else:
                out["errors"][reply["type"]] = out["errors"].get(reply["type"], 0) + 1


def _buckets() -> dict:
    return {"latency": [], "preprocess": [], "infer": [], "server": [], "errors": {}, "top1": {}}


async def level(args, frames: list[bytes], sessions: int) -> dict:
    # Warm-up round (discarded) so connections, worker threads and Triton are hot.
    warm_stop = time.perf_counter() + args.warmup
    await asyncio.gather(
        *(session(args.url, frames, warm_stop, _buckets(), i) for i in range(sessions))
    )
    out = _buckets()
    before = await triton_counts(args.triton_metrics, args.model)
    t0 = time.perf_counter()
    stop = t0 + args.duration
    await asyncio.gather(*(session(args.url, frames, stop, out, i * 3) for i in range(sessions)))
    wall = time.perf_counter() - t0
    after = await triton_counts(args.triton_metrics, args.model)
    done = len(out["latency"])
    row = {
        "label": args.label,
        "sessions": sessions,
        "frames": done,
        "errors": out["errors"],
        "wall_s": round(wall, 2),
        "frames_per_s": round(done / wall, 2),
        "latency_ms": {
            q: pct(out["latency"], v) for q, v in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        },
        "server_ms_p50": {
            "preprocess": pct(out["preprocess"], 0.5),
            "infer": pct(out["infer"], 0.5),
            "total": pct(out["server"], 0.5),
        },
        "top1_labels": dict(sorted(out["top1"].items(), key=lambda kv: -kv[1])[:3]),
    }
    if before and after:
        requests, execs = after[0] - before[0], after[1] - before[1]
        row["triton"] = {
            "model": args.model,
            "requests": int(requests),
            "executions": int(execs),
            "avg_batch_size": round(requests / execs, 2) if execs else None,
        }
    return row


async def run(args) -> dict:
    frames = make_frames(*map(int, args.frame_size.split("x")), n=args.distinct_frames)
    report = {
        "label": args.label,
        "url": args.url,
        "host": platform.platform(),
        "frame_size": args.frame_size,
        "jpeg_kb_mean": round(float(np.mean([len(f) for f in frames])) / 1024, 1),
        "duration_s": args.duration,
        "levels": [],
    }
    for n in args.sessions:
        row = await level(args, frames, n)
        report["levels"].append(row)
        print(json.dumps(row), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="ws://127.0.0.1:8080/ws/frames")
    p.add_argument("--sessions", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--frame-size", default="720x1280")
    p.add_argument("--distinct-frames", type=int, default=32)
    p.add_argument("--triton-metrics", help="e.g. http://127.0.0.1:8002/metrics")
    p.add_argument("--model", default="vit_b16", help="Triton model the gateway is using")
    p.add_argument("--label", default="run")
    p.add_argument("--output")
    args = p.parse_args()
    report = asyncio.run(run(args))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
