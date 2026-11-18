"""Token-bucket rate limiting: Redis (shared across workers) or in-memory.

A bucket holds up to `capacity` tokens and refills continuously at
`refill_per_s`. Each message spends `cost` tokens. When there aren't enough,
the message is rejected and `retry_after_s` says when it would succeed.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from backend.core.config import Settings


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: float
    retry_after_s: float


class RateLimiter(Protocol):
    async def acquire(self, key: str, cost: float = 1.0) -> RateDecision: ...
    async def close(self) -> None: ...


class InMemoryTokenBucket:
    def __init__(
        self,
        capacity: float,
        refill_per_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 100_000,
    ):
        if capacity <= 0 or refill_per_s <= 0:
            raise ValueError("capacity and refill_per_s must be positive")
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.clock = clock
        self.max_keys = max_keys
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_ts)

    async def acquire(self, key: str, cost: float = 1.0) -> RateDecision:
        now = self.clock()
        tokens, last = self._buckets.get(key, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_s)
        if tokens >= cost:
            tokens -= cost
            decision = RateDecision(True, tokens, 0.0)
        else:
            decision = RateDecision(False, tokens, (cost - tokens) / self.refill_per_s)
        if key not in self._buckets and len(self._buckets) >= self.max_keys:
            self._evict_full(now)
        self._buckets[key] = (tokens, now)
        return decision

    def _evict_full(self, now: float) -> None:
        """Drop buckets that have refilled completely: they carry no state."""
        for k, (tokens, last) in list(self._buckets.items()):
            if tokens + (now - last) * self.refill_per_s >= self.capacity:
                del self._buckets[k]

    async def close(self) -> None:
        self._buckets.clear()


# Atomic refill + spend. Stored as a hash {tokens, ts}. Idle keys expire once
# they would be full again, so Redis memory tracks only active clients.
_LUA_TOKEN_BUCKET = """
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
if now > ts then
  tokens = math.min(capacity, tokens + (now - ts) * rate)
else
  now = ts
end
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  retry = (cost - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity / rate * 1000) + 1000)
return {allowed, tostring(tokens), tostring(retry)}
"""


class RedisTokenBucket:
    def __init__(
        self,
        client,
        capacity: float,
        refill_per_s: float,
        prefix: str = "ratelimit:",
        clock: Callable[[], float] = time.time,
    ):
        self.client = client
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.prefix = prefix
        self.clock = clock
        self._script = client.register_script(_LUA_TOKEN_BUCKET)

    @classmethod
    def from_url(cls, url: str, capacity: float, refill_per_s: float) -> RedisTokenBucket:
        import redis.asyncio as redis

        return cls(redis.from_url(url, decode_responses=True), capacity, refill_per_s)

    async def acquire(self, key: str, cost: float = 1.0) -> RateDecision:
        allowed, remaining, retry = await self._script(
            keys=[self.prefix + key],
            args=[self.capacity, self.refill_per_s, self.clock(), cost],
        )
        return RateDecision(bool(int(allowed)), float(remaining), float(retry))

    async def close(self) -> None:
        await self.client.aclose()


def rate_limiter_from_settings(settings: Settings) -> RateLimiter:
    if settings.redis_url:
        return RedisTokenBucket.from_url(
            settings.redis_url, settings.rate_limit_capacity, settings.rate_limit_refill_per_s
        )
    return InMemoryTokenBucket(settings.rate_limit_capacity, settings.rate_limit_refill_per_s)
