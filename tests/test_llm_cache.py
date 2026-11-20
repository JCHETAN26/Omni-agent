import asyncio

import pytest
from pydantic import BaseModel

from backend.agent.llm import LLMError
from backend.agent.llm_cache import CachedChatModel
from backend.core.cache import InMemoryCache


class Verdict(BaseModel):
    score: float
    reason: str


class FakeModel:
    name = "fake-model"

    def __init__(self, text="Rotate the TLS keystore before restarting the broker [1]."):
        self.json_calls = 0
        self.stream_calls = 0
        self.text = text
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail: Exception | None = None

    async def complete_json(self, system, messages, schema, effort="low"):
        self.json_calls += 1
        await self.gate.wait()
        if self.fail:
            raise self.fail
        return schema(score=0.75, reason=messages[-1]["content"])

    async def stream_text(self, system, messages, effort="medium"):
        self.stream_calls += 1
        words = self.text.split(" ")
        for i, word in enumerate(words):
            if i == 1:
                await self.gate.wait()  # hold the stream open after the first token
            if self.fail:
                raise self.fail
            yield word if i == len(words) - 1 else word + " "

    async def close(self):
        pass


def _msgs(text="what broke?"):
    return [{"role": "user", "content": text}]


async def _collect(stream):
    return "".join([delta async for delta in stream])


async def test_structured_result_is_cached_by_full_prompt():
    inner = FakeModel()
    model = CachedChatModel(inner, InMemoryCache())
    first = await model.complete_json("sys", _msgs(), Verdict)
    again = await model.complete_json("sys", _msgs(), Verdict)
    assert again == first and inner.json_calls == 1
    await model.complete_json("sys", _msgs("different"), Verdict)
    await model.complete_json("other system", _msgs(), Verdict)
    await model.complete_json("sys", _msgs(), Verdict, effort="high")
    assert inner.json_calls == 4


async def test_concurrent_identical_calls_share_one_upstream_request():
    inner = FakeModel()
    inner.gate.clear()
    model = CachedChatModel(inner, InMemoryCache())
    tasks = [asyncio.create_task(model.complete_json("sys", _msgs(), Verdict)) for _ in range(20)]
    await asyncio.sleep(0.01)
    inner.gate.set()
    results = await asyncio.gather(*tasks)
    assert inner.json_calls == 1 and len({r.reason for r in results}) == 1


async def test_errors_reach_every_coalesced_caller_and_are_not_cached():
    inner = FakeModel()
    inner.gate.clear()
    inner.fail = LLMError("upstream 500")
    model = CachedChatModel(inner, InMemoryCache())
    tasks = [asyncio.create_task(model.complete_json("sys", _msgs(), Verdict)) for _ in range(5)]
    await asyncio.sleep(0.01)
    inner.gate.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, LLMError) for r in results) and inner.json_calls == 1
    inner.fail = None
    assert (await model.complete_json("sys", _msgs(), Verdict)).score == 0.75
    assert inner.json_calls == 2  # the failure was not cached


async def test_streamed_answer_is_cached_and_replayed_exactly():
    inner = FakeModel(text="line one\n  spaced   words [1] end.")
    model = CachedChatModel(inner, InMemoryCache())
    first = await _collect(model.stream_text("sys", _msgs()))
    replay_deltas = [d async for d in model.stream_text("sys", _msgs())]
    assert first == inner.text == "".join(replay_deltas)
    assert len(replay_deltas) > 1  # a cached answer still streams
    assert inner.stream_calls == 1


async def test_followers_receive_tokens_before_the_leader_finishes():
    inner = FakeModel()
    inner.gate.clear()
    model = CachedChatModel(inner, InMemoryCache())
    leader = model.stream_text("sys", _msgs())
    follower = model.stream_text("sys", _msgs())
    lead_first = await leader.__anext__()
    follow_first = await asyncio.wait_for(follower.__anext__(), 1)
    assert lead_first == follow_first == "Rotate "
    inner.gate.set()
    lead_rest = await _collect(leader)
    follow_rest = await _collect(follower)
    assert lead_first + lead_rest == follow_first + follow_rest == inner.text
    assert inner.stream_calls == 1


async def test_abandoned_leader_turns_into_an_error_for_followers():
    inner = FakeModel()
    inner.gate.clear()
    model = CachedChatModel(inner, InMemoryCache())
    leader = model.stream_text("sys", _msgs())
    follower = model.stream_text("sys", _msgs())
    await leader.__anext__()
    await follower.__anext__()
    await leader.aclose()  # e.g. the leader's WebSocket client disconnected
    with pytest.raises(LLMError, match="before completion"):
        await _collect(follower)
    inner.gate.set()
    assert await _collect(model.stream_text("sys", _msgs())) == inner.text  # nothing was cached


async def test_runner_wraps_the_model_only_when_enabled(monkeypatch):
    from backend.agent import runner as runner_module
    from backend.core.services import Services
    from tests.conftest import make_settings

    monkeypatch.setattr(runner_module, "chat_model_from_settings", lambda s: FakeModel())
    for enabled, expected in ((True, CachedChatModel), (False, FakeModel)):
        services = await Services.build(make_settings(llm_cache_enabled=enabled))
        try:
            agent = runner_module.AgentRunner.from_services(services)
            assert type(agent.deps.reasoner.model) is expected
        finally:
            await services.close()
