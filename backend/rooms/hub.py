"""Per-replica fan-out of room events to local WebSocket sessions.

One Pub/Sub subscription per replica covers every room that has a member
connected here. Its reader hands each published event (raw JSON, never
re-encoded) to every local session of that room through a bounded queue:
`offer` never blocks, so one slow client cannot stall a room. A session whose
queue overflows is marked slow; its sender closes it with 4409 and the client
reconnects with `last_seq`, catching up from the room's event list.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import WebSocket

from backend.core import metrics
from backend.rooms.store import RoomStore, seq_of

log = logging.getLogger(__name__)


@dataclass(eq=False)
class RoomSession:
    room: str
    name: str
    ws: WebSocket
    queue_size: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sent_seq: int = 0  # highest sequenced event delivered (or replayed) to this client
    slow: bool = False
    queue: asyncio.Queue[str] = field(init=False)
    joined_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.queue_size)

    def offer(self, raw: str) -> None:
        if self.slow:
            return
        try:
            self.queue.put_nowait(raw)
        except asyncio.QueueFull:
            self.slow = True
            metrics.ROOM_SLOW_CONSUMERS.inc()
            # Wake the sender so it can close the session for a clean resync.
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.queue.put_nowait('{"type":"__slow__"}')

    async def run_sender(self) -> None:
        """Deliver queued events in order; skip sequenced events already sent
        (history replay and live delivery overlap during a join)."""
        while True:
            raw = await self.queue.get()
            if raw == '{"type":"__slow__"}':
                return
            seq = seq_of(raw)
            if seq is not None:
                if seq <= self.sent_seq:
                    continue
                self.sent_seq = seq
            await self.ws.send_text(raw)


class RoomHub:
    def __init__(self, store: RoomStore):
        self.store = store
        self.sub = store.subscribe()
        self.sessions: dict[str, set[RoomSession]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._reader: asyncio.Task | None = None

    def start(self) -> None:
        self._reader = asyncio.create_task(self._read(), name="room-hub-reader")

    async def stop(self) -> None:
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
        await self.sub.close()

    @property
    def local_sessions(self) -> int:
        return sum(len(s) for s in self.sessions.values())

    async def join(self, s: RoomSession) -> None:
        async with self._lock:
            if not self.sessions[s.room]:
                await self.sub.add(s.room)
            self.sessions[s.room].add(s)
        metrics.ROOM_SESSIONS.inc()

    async def leave(self, s: RoomSession) -> None:
        async with self._lock:
            members = self.sessions.get(s.room)
            if members is None or s not in members:
                return
            members.discard(s)
            if not members:
                del self.sessions[s.room]
                await self.sub.remove(s.room)
        metrics.ROOM_SESSIONS.dec()

    async def _read(self) -> None:
        async for room, raw in self.sub.messages():
            for s in tuple(self.sessions.get(room, ())):
                s.offer(raw)

    async def sweep(self, presence_ttl_s: float, interval_s: float) -> None:
        """Drop presence entries whose owner stopped heartbeating (e.g. a
        replica crashed), announcing them as having left."""
        while True:
            await asyncio.sleep(interval_s)
            cutoff = time.time() - presence_ttl_s
            for room in tuple(self.sessions):
                try:
                    for sid, p in (await self.store.presence(room)).items():
                        if p.get("last_seen", 0) < cutoff:
                            await self.store.remove_presence(room, sid)
                            await self.store.append(
                                room,
                                {
                                    "type": "member.left",
                                    "session": sid,
                                    "name": p.get("name"),
                                    "reason": "heartbeat timeout",
                                },
                            )
                except Exception:  # noqa: BLE001 - keep sweeping other rooms
                    log.exception("presence sweep failed", extra={"room": room})
