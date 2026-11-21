"""Shared incident-room state: Redis for multi-replica deployments, memory otherwise.

Per room (all keys expire after `ttl_s` without activity):

    room:{id}:meta       hash   title, created_at, created_by
    room:{id}:seq        int    last assigned sequence number (INCR)
    room:{id}:events     list   the last `max_events` sequenced events, oldest first
    room:{id}:presence   hash   user -> {"name", "focus", "last_seen"}
    room:{id}:pins       hash   pin_id -> pinned evidence
    channel room:{id}           every event, as published JSON

`append` is one Lua script: INCR the sequence, add it to the event, push,
trim, refresh TTLs and PUBLISH, atomically. So the order on the channel is
the sequence order on every replica, and a late joiner can fetch
`events_after(seq)` from the list without gaps or duplicates.

The sequence number is spliced into the JSON as text (`{"seq":N,` + rest)
rather than re-encoded, so client timestamps and other numbers keep their
exact precision, and subscribers can read `seq` without parsing the event.
Ephemeral events (typing, focus, agent tokens) are published without a
sequence number and not stored.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

APPEND_LUA = """
local seq = redis.call('INCR', KEYS[1])
local ev = '{"seq":' .. seq .. ',' .. string.sub(ARGV[1], 2)
redis.call('RPUSH', KEYS[2], ev)
redis.call('LTRIM', KEYS[2], -tonumber(ARGV[3]), -1)
redis.call('EXPIRE', KEYS[1], ARGV[4])
redis.call('EXPIRE', KEYS[2], ARGV[4])
redis.call('EXPIRE', KEYS[3], ARGV[4])
redis.call('PUBLISH', ARGV[2], ev)
return seq
"""


def channel(room: str) -> str:
    return f"room:{room}"


def with_seq(seq: int, body: str) -> str:
    return '{"seq":' + str(seq) + "," + body[1:]


def seq_of(raw: str) -> int | None:
    """The sequence number of a published event, without parsing it."""
    if not raw.startswith('{"seq":'):
        return None
    end = raw.find(",", 7)
    return int(raw[7:end])


def dumps(event: dict) -> str:
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False)


class RoomStore(Protocol):
    async def create(self, room: str, meta: dict) -> None: ...
    async def meta(self, room: str) -> dict | None: ...
    async def append(self, room: str, event: dict) -> tuple[int, str]: ...
    async def publish_ephemeral(self, room: str, event: dict) -> None: ...
    async def last_seq(self, room: str) -> int: ...
    async def events_after(self, room: str, after: int, limit: int) -> tuple[list[str], bool]: ...
    async def claim_once(self, room: str, key: str, ttl_s: int = 600) -> bool: ...
    async def set_presence(self, room: str, user: str, value: dict) -> None: ...
    async def remove_presence(self, room: str, user: str) -> None: ...
    async def presence(self, room: str) -> dict[str, dict]: ...
    async def add_pin(self, room: str, pin_id: str, pin: dict) -> None: ...
    async def remove_pin(self, room: str, pin_id: str) -> bool: ...
    async def pins(self, room: str) -> dict[str, dict]: ...
    def subscribe(self) -> Subscription: ...
    async def close(self) -> None: ...


class Subscription(Protocol):
    async def add(self, room: str) -> None: ...
    async def remove(self, room: str) -> None: ...
    def messages(self) -> AsyncIterator[tuple[str, str]]: ...  # (room, raw event)
    async def close(self) -> None: ...


# --- Redis -------------------------------------------------------------------


class RedisRoomStore:
    def __init__(
        self, url: str, max_events: int = 1000, ttl_s: int = 86_400, max_connections: int = 64
    ):
        import redis.asyncio as redis

        # A bounded *blocking* pool: a join burst (hundreds of sessions at once)
        # waits briefly for a free connection instead of failing with
        # MaxConnectionsError, and never opens hundreds of Redis connections.
        pool = redis.BlockingConnectionPool.from_url(
            url, max_connections=max_connections, timeout=10, decode_responses=True
        )
        self.r = redis.Redis(connection_pool=pool)
        self.url = url
        self.max_events = max_events
        self.ttl_s = ttl_s
        self._append = self.r.register_script(APPEND_LUA)

    def _k(self, room: str, part: str) -> str:
        return f"room:{room}:{part}"

    async def create(self, room: str, meta: dict) -> None:
        key = self._k(room, "meta")
        await self.r.hset(key, mapping={k: str(v) for k, v in meta.items()})
        await self.r.expire(key, self.ttl_s)

    async def meta(self, room: str) -> dict | None:
        m = await self.r.hgetall(self._k(room, "meta"))
        return m or None

    async def append(self, room: str, event: dict) -> tuple[int, str]:
        body = dumps(event)
        seq = await self._append(
            keys=[self._k(room, "seq"), self._k(room, "events"), self._k(room, "meta")],
            args=[body, channel(room), self.max_events, self.ttl_s],
        )
        return int(seq), with_seq(int(seq), body)

    async def publish_ephemeral(self, room: str, event: dict) -> None:
        await self.r.publish(channel(room), dumps(event))

    async def last_seq(self, room: str) -> int:
        return int(await self.r.get(self._k(room, "seq")) or 0)

    async def events_after(self, room: str, after: int, limit: int) -> tuple[list[str], bool]:
        """Events with seq > after (at most `limit`, newest ones), and whether
        older ones the caller asked for were already trimmed (a gap)."""
        key = self._k(room, "events")
        first = await self.r.lrange(key, 0, 0)
        if not first:
            return [], after < await self.last_seq(room)
        first_seq = seq_of(first[0]) or 1
        start = max(after + 1 - first_seq, 0)
        raw = await self.r.lrange(key, start, -1)
        trimmed = after + 1 < first_seq
        if len(raw) > limit:
            raw, trimmed = raw[-limit:], True
        return raw, trimmed

    async def claim_once(self, room: str, key: str, ttl_s: int = 600) -> bool:
        """True the first time `key` is claimed in this room (idempotent sends)."""
        return bool(await self.r.set(self._k(room, f"once:{key}"), 1, nx=True, ex=ttl_s))

    async def set_presence(self, room: str, user: str, value: dict) -> None:
        key = self._k(room, "presence")
        await self.r.hset(key, user, dumps(value))
        await self.r.expire(key, self.ttl_s)

    async def remove_presence(self, room: str, user: str) -> None:
        await self.r.hdel(self._k(room, "presence"), user)

    async def presence(self, room: str) -> dict[str, dict]:
        return {
            u: json.loads(v) for u, v in (await self.r.hgetall(self._k(room, "presence"))).items()
        }

    async def add_pin(self, room: str, pin_id: str, pin: dict) -> None:
        key = self._k(room, "pins")
        await self.r.hset(key, pin_id, dumps(pin))
        await self.r.expire(key, self.ttl_s)

    async def remove_pin(self, room: str, pin_id: str) -> bool:
        return bool(await self.r.hdel(self._k(room, "pins"), pin_id))

    async def pins(self, room: str) -> dict[str, dict]:
        return {k: json.loads(v) for k, v in (await self.r.hgetall(self._k(room, "pins"))).items()}

    def subscribe(self) -> RedisSubscription:
        return RedisSubscription(self.r)

    async def close(self) -> None:
        await self.r.aclose()


class RedisSubscription:
    """One Pub/Sub connection per replica, subscribed to the rooms that have
    local members."""

    def __init__(self, r):
        self.ps = r.pubsub(ignore_subscribe_messages=True)
        self._ready = asyncio.Event()

    async def add(self, room: str) -> None:
        await self.ps.subscribe(channel(room))
        self._ready.set()

    async def remove(self, room: str) -> None:
        await self.ps.unsubscribe(channel(room))

    async def messages(self) -> AsyncIterator[tuple[str, str]]:
        while True:
            await self._ready.wait()  # get_message needs a connection (first subscribe)
            msg = await self.ps.get_message(timeout=1.0)
            if msg and msg["type"] == "message":
                yield msg["channel"][len("room:") :], msg["data"]

    async def close(self) -> None:
        await self.ps.aclose()


# --- In-memory (single process: dev, tests) -----------------------------------


class MemoryRoomStore:
    def __init__(self, max_events: int = 1000, ttl_s: int = 86_400):
        self.max_events = max_events
        self.ttl_s = ttl_s
        self._meta: dict[str, dict] = {}
        self._seq: dict[str, int] = defaultdict(int)
        self._events: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=self.max_events))
        self._presence: dict[str, dict[str, dict]] = defaultdict(dict)
        self._pins: dict[str, dict[str, dict]] = defaultdict(dict)
        self._subs: list[MemorySubscription] = []
        self._claims: dict[str, float] = {}

    async def create(self, room: str, meta: dict) -> None:
        self._meta[room] = {k: str(v) for k, v in meta.items()}

    async def meta(self, room: str) -> dict | None:
        return self._meta.get(room)

    def _publish(self, room: str, raw: str) -> None:
        for s in self._subs:
            s.deliver(room, raw)

    async def append(self, room: str, event: dict) -> tuple[int, str]:
        self._seq[room] += 1
        seq = self._seq[room]
        raw = with_seq(seq, dumps(event))
        self._events[room].append(raw)
        self._publish(room, raw)
        return seq, raw

    async def publish_ephemeral(self, room: str, event: dict) -> None:
        self._publish(room, dumps(event))

    async def last_seq(self, room: str) -> int:
        return self._seq.get(room, 0)

    async def events_after(self, room: str, after: int, limit: int) -> tuple[list[str], bool]:
        events = self._events.get(room)
        if not events:
            return [], after < self._seq.get(room, 0)
        first_seq = seq_of(events[0]) or 1
        raw = [e for e in events if (seq_of(e) or 0) > after]
        trimmed = after + 1 < first_seq
        if len(raw) > limit:
            raw, trimmed = raw[-limit:], True
        return raw, trimmed

    async def claim_once(self, room: str, key: str, ttl_s: int = 600) -> bool:
        now, k = time.time(), f"{room}:{key}"
        if self._claims.get(k, 0) > now:
            return False
        self._claims[k] = now + ttl_s
        return True

    async def set_presence(self, room: str, user: str, value: dict) -> None:
        self._presence[room][user] = value

    async def remove_presence(self, room: str, user: str) -> None:
        self._presence[room].pop(user, None)

    async def presence(self, room: str) -> dict[str, dict]:
        return dict(self._presence.get(room, {}))

    async def add_pin(self, room: str, pin_id: str, pin: dict) -> None:
        self._pins[room][pin_id] = pin

    async def remove_pin(self, room: str, pin_id: str) -> bool:
        return self._pins[room].pop(pin_id, None) is not None

    async def pins(self, room: str) -> dict[str, dict]:
        return dict(self._pins.get(room, {}))

    def subscribe(self) -> MemorySubscription:
        s = MemorySubscription(lambda: self._subs.remove(s))
        self._subs.append(s)
        return s

    async def close(self) -> None:
        return None


class MemorySubscription:
    def __init__(self, on_close: Callable[[], None]):
        self.rooms: set[str] = set()
        self.queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._on_close = on_close

    def deliver(self, room: str, raw: str) -> None:
        if room in self.rooms:
            self.queue.put_nowait((room, raw))

    async def add(self, room: str) -> None:
        self.rooms.add(room)

    async def remove(self, room: str) -> None:
        self.rooms.discard(room)

    async def messages(self) -> AsyncIterator[tuple[str, str]]:
        while True:
            yield await self.queue.get()

    async def close(self) -> None:
        self._on_close()


def now_ms() -> float:
    return round(time.time() * 1000, 3)


Event = dict[str, Any]
