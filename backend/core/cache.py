"""JSON cache: Redis when REDIS_URL is set, otherwise an in-process TTL dict."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from backend.core.config import Settings


class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: int) -> None: ...
    async def close(self) -> None: ...


class RedisCache:
    def __init__(self, url: str):
        import redis.asyncio as redis

        self.client = redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> Any | None:
        raw = await self.client.get(key)
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        await self.client.set(key, json.dumps(value), ex=ttl)

    async def close(self) -> None:
        await self.client.aclose()


class InMemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, raw = item
        if time.monotonic() >= expires_at:
            del self._data[key]
            return None
        return json.loads(raw)

    async def set(self, key: str, value: Any, ttl: int) -> None:
        self._data[key] = (time.monotonic() + ttl, json.dumps(value))

    async def close(self) -> None:
        self._data.clear()


def cache_from_settings(settings: Settings) -> Cache:
    return RedisCache(settings.redis_url) if settings.redis_url else InMemoryCache()
