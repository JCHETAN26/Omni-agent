"""Live frames: /ws/frames.

    ws://host/ws/frames?thread_id=<id>

Each binary message is one encoded frame (JPEG or PNG) from the user's camera
or screen. The gateway preprocesses it (C++ OpenCV extension when built), runs
the ViT classifier on Triton, replies with the top labels and per-stage
timings, and records the observation for the thread. The next agent turn on
that thread (/ws/agent or /sse/agent with the same thread_id) receives it as
visual context.

Replies (JSON text messages, one per frame, in order):
    {"type": "frame.result", "seq": n, "labels": [...], "timings": {...}}
    {"type": "frame.unavailable", "seq": n, "reason": "..."}   Triton down: no context
    {"type": "frame.error", "seq": n, "error": "..."}          undecodable or too large

Frames on one connection are handled in order, one at a time; a client that
wants more throughput opens more sessions (each is a separate Triton request,
which Triton's dynamic batcher can merge).
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.agent_ws import _THREAD_RE as THREAD_RE
from backend.core import metrics
from backend.core.services import Services
from backend.vision import context
from model_server.vision_client import VisionUnavailable

router = APIRouter()

CLOSE_POLICY = 1008


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


@router.websocket("/ws/frames")
async def ws_frames(ws: WebSocket, thread_id: str | None = None) -> None:
    services: Services = ws.app.state.services
    await ws.accept()
    if thread_id is not None and not THREAD_RE.match(thread_id):
        await ws.close(code=CLOSE_POLICY, reason="invalid thread_id")
        return
    if services.vision is None or services.preprocessor is None:
        await ws.close(code=CLOSE_POLICY, reason="vision is disabled (VISION_BACKEND=off)")
        return
    s = services.settings
    thread = thread_id or uuid.uuid4().hex
    await ws.send_json(
        {
            "type": "session.created",
            "thread_id": thread,
            "preprocessor": services.preprocessor.name,
            "classifier": services.vision.name,
        }
    )
    metrics.WS_SESSIONS.labels("/ws/frames").inc()
    seq = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if data is None:
                continue  # text messages (keepalives) carry no frame
            seq += 1
            t0 = time.perf_counter()
            if len(data) > s.vision_max_frame_bytes:
                metrics.FRAMES.labels("invalid").inc()
                await ws.send_json({"type": "frame.error", "seq": seq, "error": "frame too large"})
                continue
            try:
                # C++ releases the GIL: preprocessing runs in parallel worker threads.
                pixels = await asyncio.to_thread(services.preprocessor.encoded, data)
            except ValueError as exc:
                metrics.FRAMES.labels("invalid").inc()
                await ws.send_json({"type": "frame.error", "seq": seq, "error": str(exc)})
                continue
            t1 = time.perf_counter()
            try:
                labels = await services.vision.classify(pixels)
            except VisionUnavailable as exc:
                metrics.FRAMES.labels("unavailable").inc()
                await ws.send_json({"type": "frame.unavailable", "seq": seq, "reason": str(exc)})
                continue
            t2 = time.perf_counter()
            await context.record(services.cache, thread, seq, labels, s.vision_context_ttl_s)
            metrics.FRAMES.labels("classified").inc()
            metrics.FRAME_SECONDS.labels("preprocess").observe(t1 - t0)
            metrics.FRAME_SECONDS.labels("infer").observe(t2 - t1)
            metrics.FRAME_SECONDS.labels("total").observe(time.perf_counter() - t0)
            await ws.send_json(
                {
                    "type": "frame.result",
                    "seq": seq,
                    "labels": [{"label": lb.label, "score": lb.score} for lb in labels],
                    "timings": {
                        "preprocess_ms": _ms(t1 - t0),
                        "infer_ms": _ms(t2 - t1),
                        "server_ms": _ms(time.perf_counter() - t0),
                    },
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        metrics.WS_SESSIONS.labels("/ws/frames").dec()
