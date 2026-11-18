"""Bounded async FIFO that drops the OLDEST item when full (ring-buffer semantics).

For live streams (audio frames especially), stale data is worth less than fresh
data, so a slow consumer should lose old frames instead of blocking the reader
or growing memory without bound.
"""

from __future__ import annotations

import asyncio


class RingBuffer[T]:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=capacity)
        self.dropped = 0

    def push(self, item: T) -> T | None:
        """Enqueue without blocking. Returns the evicted item, if any."""
        evicted = None
        if self._queue.full():
            evicted = self._queue.get_nowait()
            self.dropped += 1
        self._queue.put_nowait(item)
        return evicted

    async def get(self) -> T:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()
