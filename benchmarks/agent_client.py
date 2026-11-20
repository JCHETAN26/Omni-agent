"""Streaming SSE client for one agent turn, with sequence and checksum checks.

Used by the throughput benchmark; the `omni` Go CLI implements the same protocol.
"""

from __future__ import annotations

import hashlib
import json
import time
from urllib.parse import urlencode
from urllib.request import urlopen


class GatewayError(Exception):
    """The gateway returned an error or an incomplete response."""


def ask(server: str, question: str, thread_id: str | None = None) -> dict:
    params = {"q": question}
    if thread_id:
        params["thread_id"] = thread_id
    start = time.perf_counter()
    first_token = None
    parts: list[str] = []
    sources = []
    event_id = 0
    with urlopen(f"{server}/sse/agent?{urlencode(params)}", timeout=90) as response:
        thread = response.headers.get("x-thread-id")
        for line in response:
            if line.startswith(b"id: "):
                event_id += 1
                if int(line[4:]) != event_id:
                    raise GatewayError("unexpected SSE event sequence")
            elif line.startswith(b"data: "):
                event = json.loads(line[6:])
                match event["type"]:
                    case "response.error":
                        raise GatewayError(event.get("error", "agent request failed"))
                    case "response.sources":
                        sources = event["sources"]
                    case "response.token":
                        if event["index"] != len(parts):
                            raise GatewayError("unexpected answer token sequence")
                        first_token = first_token or time.perf_counter()
                        parts.append(event["text"])
                    case "response.end":
                        text = "".join(parts)
                        checksum = hashlib.sha256(text.encode()).hexdigest()
                        if len(parts) != event["tokens"] or checksum != event["sha256"]:
                            raise GatewayError("answer integrity check failed")
                        return {
                            "text": text,
                            "sources": sources,
                            "outcome": event["outcome"],
                            "tool": event["tool"],
                            "retries": event.get("retries", 0),
                            "thread_id": thread,
                            "ttft_ms": None
                            if first_token is None
                            else (first_token - start) * 1000,
                            "total_ms": (time.perf_counter() - start) * 1000,
                        }
    raise GatewayError("stream ended without a complete answer")
