"""The latest frame observation per conversation thread, kept in the shared
cache (Redis when configured) so any gateway replica can inject it into an
agent turn."""

from __future__ import annotations

import time

from backend.core.cache import Cache
from model_server.vision_client import Label

_KEY = "vision:thread:{}"


async def record(cache: Cache, thread_id: str, seq: int, labels: list[Label], ttl_s: int) -> None:
    value = {
        "seq": seq,
        "at": time.time(),
        "labels": [{"label": lb.label, "score": lb.score} for lb in labels],
    }
    await cache.set(_KEY.format(thread_id), value, ttl_s)


async def describe(cache: Cache, thread_id: str, ttl_s: int, now: float | None = None) -> str:
    """One line for the agent, or "" when no fresh frame exists for the thread."""
    obs = await cache.get(_KEY.format(thread_id))
    if not obs:
        return ""
    age = (now or time.time()) - obs["at"]
    if age > ttl_s or not obs["labels"]:
        return ""
    top = ", ".join(f"{x['label']} ({x['score']:.2f})" for x in obs["labels"][:3])
    return f"The user's live camera/screen frame #{obs['seq']} ({age:.0f} s ago) shows: {top}."
