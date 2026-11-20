"""Exact-match LLM response cache with single-flight request coalescing.

Wraps any ChatModel. Two mechanisms, both keyed on everything that determines
the output (model, system prompt, full message list including retrieved
passages, response schema and effort):

* Cache: a completed structured result or streamed answer is stored in the
  shared Cache (Redis when configured, so all gateway replicas share it) for
  ``ttl_s``. Retrieved passages are part of the key, so re-indexing that
  changes the context changes the key and can never serve a stale answer.
* Single flight: concurrent identical calls share one upstream request. A
  streamed answer is fanned out token by token, so followers do not wait for
  the leader to finish before their first token.

Errors and refusals are never cached; they propagate to every coalesced caller.
Caching assumes deterministic decoding (temperature 0) for identical prompts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

from backend.agent.llm import ChatModel, Effort, LLMError, Message, T
from backend.core import metrics
from backend.core.cache import Cache

_REPLAY_CHUNK_WORDS = 4  # cached answers still stream, in small word groups


class _Broadcast:
    """Fan a leader's streamed deltas out to any number of followers."""

    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.done = False
        self.error: BaseException | None = None
        self._cond = asyncio.Condition()

    async def publish(self, delta: str) -> None:
        async with self._cond:
            self.chunks.append(delta)
            self._cond.notify_all()

    async def finish(self, error: BaseException | None = None) -> None:
        async with self._cond:
            self.done, self.error = True, error
            self._cond.notify_all()

    async def follow(self) -> AsyncIterator[str]:
        sent = 0
        while True:
            async with self._cond:
                await self._cond.wait_for(lambda n=sent: len(self.chunks) > n or self.done)
                pending, finished, error = self.chunks[sent:], self.done, self.error
            for delta in pending:
                yield delta
            sent += len(pending)
            if finished and sent == len(self.chunks):
                if error is not None:
                    raise error
                return


class CachedChatModel:
    def __init__(self, inner: ChatModel, cache: Cache, ttl_s: int = 3600):
        self.inner = inner
        self.cache = cache
        self.ttl_s = ttl_s
        self.name = inner.name
        self._json_flights: dict[str, asyncio.Future] = {}
        self._stream_flights: dict[str, _Broadcast] = {}

    def _key(self, kind: str, system: str, messages: list[Message], extra: dict[str, Any]) -> str:
        material = json.dumps(
            {
                "kind": kind,
                "model": self.inner.name,
                "system": system,
                "messages": messages,
                **extra,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"llm:{kind}:{hashlib.sha256(material.encode()).hexdigest()}"

    # -- structured steps --------------------------------------------------------

    async def complete_json(
        self, system: str, messages: list[Message], schema: type[T], effort: Effort = "low"
    ) -> T:
        key = self._key(
            "json",
            system,
            messages,
            {"schema": schema.__name__, "fields": schema.model_json_schema(), "effort": effort},
        )
        if (cached := await self.cache.get(key)) is not None:
            metrics.LLM_CACHE.labels("json", "hit").inc()
            return schema.model_validate(cached)
        if (flight := self._json_flights.get(key)) is not None:
            metrics.LLM_CACHE.labels("json", "coalesced").inc()
            return schema.model_validate(await asyncio.shield(flight))
        metrics.LLM_CACHE.labels("json", "miss").inc()
        flight = asyncio.get_running_loop().create_future()
        self._json_flights[key] = flight
        try:
            result = await self.inner.complete_json(system, messages, schema, effort)
            payload = result.model_dump()
            await self.cache.set(key, payload, self.ttl_s)
            flight.set_result(payload)
            return result
        except BaseException as exc:
            flight.set_exception(exc)
            flight.exception()  # mark retrieved: no "never retrieved" warning without followers
            raise
        finally:
            self._json_flights.pop(key, None)

    # -- streamed answers --------------------------------------------------------

    async def stream_text(
        self, system: str, messages: list[Message], effort: Effort = "medium"
    ) -> AsyncIterator[str]:
        key = self._key("stream", system, messages, {"effort": effort})
        if (cached := await self.cache.get(key)) is not None:
            metrics.LLM_CACHE.labels("stream", "hit").inc()
            words = cached.split(" ")
            for i in range(0, len(words), _REPLAY_CHUNK_WORDS):
                piece = " ".join(words[i : i + _REPLAY_CHUNK_WORDS])
                yield piece if i + _REPLAY_CHUNK_WORDS >= len(words) else piece + " "
            return
        if (flight := self._stream_flights.get(key)) is not None:
            metrics.LLM_CACHE.labels("stream", "coalesced").inc()
            async for delta in flight.follow():
                yield delta
            return
        metrics.LLM_CACHE.labels("stream", "miss").inc()
        flight = _Broadcast()
        self._stream_flights[key] = flight
        error: BaseException | None = None
        try:
            async for delta in self.inner.stream_text(system, messages, effort):
                await flight.publish(delta)
                yield delta
            text = "".join(flight.chunks)
            if text.strip():
                await self.cache.set(key, text, self.ttl_s)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._stream_flights.pop(key, None)
            if isinstance(error, (GeneratorExit, asyncio.CancelledError)):
                # The leader's caller went away; followers see an ordinary failure.
                error = LLMError("coalesced stream ended before completion")
            await flight.finish(error)

    async def close(self) -> None:
        await self.inner.close()
