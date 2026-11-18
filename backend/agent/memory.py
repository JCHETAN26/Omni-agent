"""Conversation memory: LangGraph checkpointers.

`RedisCheckpointSaver` persists LangGraph checkpoints using only core Redis data
types (hashes and sorted sets). It therefore runs on any managed Redis
(ElastiCache, Memorystore, Upstash, ...). LangGraph's official Redis saver
requires the RedisJSON and RediSearch modules (Redis Stack / Redis 8).

Layout per thread (prefix default "omni:lg:"):
    {p}{thread}             HASH  cp|ns|id -> checkpoint   md|ns|id -> metadata
                                  pa|ns|id -> parent id    bl|ns|ch|ver -> channel value
    {p}{thread}:idx         ZSET  "ns|id" (score 0, lexicographic = chronological ids)
    {p}{thread}:w:ns|id     HASH  task|idx -> pending write
("|" stands for the 0x1f unit separator.) Every write refreshes the TTL, so a
conversation idle longer than `ttl_s` expires as a whole. A checkpoint's pending
writes expire `ttl_s` after they were written. That only affects time-travel to
checkpoints older than the TTL, never the latest state.

Without REDIS_URL the agent uses LangGraph's InMemorySaver (per process).
"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
    writes_sort_key,
)
from langgraph.checkpoint.memory import InMemorySaver

from backend.core.config import Settings

SEP = b"\x1f"


def _b(*parts: str | bytes) -> bytes:
    return SEP.join(p if isinstance(p, bytes) else str(p).encode() for p in parts)


def _pack(typed: tuple[str, bytes]) -> bytes:
    return typed[0].encode() + b"\x00" + typed[1]


def _unpack(raw: bytes) -> tuple[str, bytes]:
    typ, _, data = raw.partition(b"\x00")
    return typ.decode(), data


def _escape_glob(s: str) -> str:
    return "".join("\\" + c if c in "*?[]\\" else c for c in s)


class RedisCheckpointSaver(BaseCheckpointSaver[str]):
    def __init__(self, client, prefix: str = "omni:lg:", ttl_s: int | None = None, serde=None):
        """`client` must be a redis.asyncio client with decode_responses=False."""
        super().__init__(serde=serde)
        self.r = client
        self.prefix = prefix
        self.ttl_s = ttl_s

    @classmethod
    def from_url(cls, url: str, ttl_s: int | None = None) -> RedisCheckpointSaver:
        import redis.asyncio as redis

        return cls(redis.from_url(url, decode_responses=False), ttl_s=ttl_s)

    # -- keys ---------------------------------------------------------------

    def _main(self, thread: str) -> str:
        return f"{self.prefix}{thread}"

    def _idx(self, thread: str) -> str:
        return f"{self.prefix}{thread}:idx"

    def _wkey(self, thread: str, ns: str, cid: str) -> bytes:
        return f"{self.prefix}{thread}:w:".encode() + _b(ns, cid)

    # -- reads --------------------------------------------------------------

    async def _latest_id(self, thread: str, ns: str) -> str | None:
        members = await self.r.zrevrangebylex(
            self._idx(thread), b"[" + _b(ns, b"\xff"), b"[" + _b(ns, b""), start=0, num=1
        )
        return members[0].split(SEP, 1)[1].decode() if members else None

    async def _load(self, thread: str, ns: str, cid: str) -> CheckpointTuple | None:
        main = self._main(thread)
        cp_raw, md_raw, parent = await self.r.hmget(
            main, _b("cp", ns, cid), _b("md", ns, cid), _b("pa", ns, cid)
        )
        if cp_raw is None:
            return None
        checkpoint: Checkpoint = self.serde.loads_typed(_unpack(cp_raw))
        versions = checkpoint.get("channel_versions", {})
        channels = list(versions)
        values: dict[str, Any] = {}
        if channels:
            blobs = await self.r.hmget(main, [_b("bl", ns, ch, versions[ch]) for ch in channels])
            for ch, raw in zip(channels, blobs, strict=True):
                if raw is not None:
                    typed = _unpack(raw)
                    if typed[0] != "empty":
                        values[ch] = self.serde.loads_typed(typed)
        stored = await self.r.hgetall(self._wkey(thread, ns, cid))
        writes = []
        for field_, raw in stored.items():
            task_id, idx = field_.split(SEP)
            hlen = int.from_bytes(raw[:4], "big")
            _, channel, task_path, typ = json.loads(raw[4 : 4 + hlen])
            writes.append((task_id.decode(), int(idx), channel, task_path, (typ, raw[4 + hlen :])))
        writes.sort(key=lambda w: writes_sort_key(w[3], w[0], w[1]))
        parent_id = parent.decode() if parent else None
        return CheckpointTuple(
            config={
                "configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": cid}
            },
            checkpoint={**checkpoint, "channel_values": values},
            metadata=self.serde.loads_typed(_unpack(md_raw)),
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_id,
                    }
                }
                if parent_id
                else None
            ),
            pending_writes=[(w[0], w[2], self.serde.loads_typed(w[4])) for w in writes],
        )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = get_checkpoint_id(config) or await self._latest_id(thread, ns)
        return await self._load(thread, ns, cid) if cid else None

    async def _threads(self) -> list[str]:
        out = []
        suffix = ":idx"
        async for key in self.r.scan_iter(match=f"{_escape_glob(self.prefix)}*{suffix}"):
            k = key.decode()
            out.append(k[len(self.prefix) : -len(suffix)])
        return out

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        threads = [config["configurable"]["thread_id"]] if config else await self._threads()
        want_ns = config["configurable"].get("checkpoint_ns") if config else None
        want_id = get_checkpoint_id(config) if config else None
        before_id = get_checkpoint_id(before) if before else None
        for thread in threads:
            members = await self.r.zrevrange(self._idx(thread), 0, -1)
            for member in members:
                ns_b, cid_b = member.split(SEP, 1)
                ns, cid = ns_b.decode(), cid_b.decode()
                if want_ns is not None and ns != want_ns:
                    continue
                if want_id and cid != want_id:
                    continue
                if before_id and cid >= before_id:
                    continue
                tup = await self._load(thread, ns, cid)
                if tup is None:
                    continue
                if filter and not all(tup.metadata.get(k) == v for k, v in filter.items()):
                    continue
                if limit is not None:
                    if limit <= 0:
                        return
                    limit -= 1
                yield tup

    # -- writes -------------------------------------------------------------

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        parent = config["configurable"].get("checkpoint_id")
        c = checkpoint.copy()
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        cid = checkpoint["id"]
        fields: dict[bytes, bytes] = {
            _b("cp", ns, cid): _pack(self.serde.dumps_typed(c)),
            _b("md", ns, cid): _pack(
                self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
            ),
        }
        if parent:
            fields[_b("pa", ns, cid)] = parent.encode()
        for ch, ver in new_versions.items():
            typed = self.serde.dumps_typed(values[ch]) if ch in values else ("empty", b"")
            fields[_b("bl", ns, ch, ver)] = _pack(typed)
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(self._main(thread), mapping=fields)
        pipe.zadd(self._idx(thread), {_b(ns, cid): 0})
        if self.ttl_s:
            pipe.expire(self._main(thread), self.ttl_s)
            pipe.expire(self._idx(thread), self.ttl_s)
        await pipe.execute()
        return {"configurable": {"thread_id": thread, "checkpoint_ns": ns, "checkpoint_id": cid}}

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        cid = config["configurable"]["checkpoint_id"]
        key = self._wkey(thread, ns, cid)
        pipe = self.r.pipeline(transaction=True)
        for i, (channel, value) in enumerate(writes):
            idx = WRITES_IDX_MAP.get(channel, i)
            typ, data = self.serde.dumps_typed(value)
            header = json.dumps([task_id, channel, task_path, typ]).encode()
            raw = len(header).to_bytes(4, "big") + header + data
            field_ = _b(task_id, str(idx))
            # Same semantics as InMemorySaver: regular writes are first-wins,
            # special channels (negative idx, e.g. errors/interrupts) overwrite.
            (pipe.hsetnx if idx >= 0 else pipe.hset)(key, field_, raw)
        if self.ttl_s:
            pipe.expire(key, self.ttl_s)
        await pipe.execute()

    async def adelete_thread(self, thread_id: str) -> None:
        keys = [self._main(thread_id), self._idx(thread_id)]
        pattern = f"{_escape_glob(self.prefix + thread_id)}:w:*"
        keys += [k async for k in self.r.scan_iter(match=pattern)]
        await self.r.delete(*keys)

    def get_next_version(self, current: str | None, channel: None = None) -> str:
        # Same scheme as InMemorySaver: zero-padded counter + random tiebreak (sortable).
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    async def aclose(self) -> None:
        await self.r.aclose()

    # Sync API is not supported: the agent graph runs fully async.
    def get_tuple(self, config):  # pragma: no cover
        raise NotImplementedError("RedisCheckpointSaver is async-only; use aget_tuple")

    def list(self, config, **kwargs):  # pragma: no cover
        raise NotImplementedError("RedisCheckpointSaver is async-only; use alist")

    def put(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("RedisCheckpointSaver is async-only; use aput")

    def put_writes(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("RedisCheckpointSaver is async-only; use aput_writes")


def checkpointer_from_settings(settings: Settings) -> BaseCheckpointSaver:
    if settings.redis_url:
        return RedisCheckpointSaver.from_url(settings.redis_url, ttl_s=settings.agent_memory_ttl_s)
    return InMemorySaver()
