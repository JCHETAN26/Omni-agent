"""RedisCheckpointSaver: LangGraph's official conformance suite + TTL/isolation."""

import fakeredis
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate

from backend.agent.memory import RedisCheckpointSaver


@checkpointer_test(name="RedisCheckpointSaver")
async def redis_saver():
    yield RedisCheckpointSaver(fakeredis.FakeAsyncRedis(decode_responses=False))


async def test_passes_langgraph_checkpointer_conformance_suite():
    report = await validate(redis_saver)
    failures = {
        name: [f for f in r.failures] for name, r in report.results.items() if r.passed is False
    }
    assert report.passed_all_base, failures
    base = ("put", "put_writes", "get_tuple", "list", "delete_thread")
    assert sum(report.results[c].tests_passed for c in base) >= 58


async def _write_one(saver, thread="thread-ttl-1", ns=""):
    from langgraph.checkpoint.base import empty_checkpoint

    cp = empty_checkpoint()
    cfg = {"configurable": {"thread_id": thread, "checkpoint_ns": ns}}
    out = await saver.aput(cfg, cp, {"source": "input", "step": -1}, {})
    await saver.aput_writes(out, [("messages", "hi")], task_id="task-1")
    return out


async def test_every_key_of_a_thread_gets_the_ttl():
    r = fakeredis.FakeAsyncRedis(decode_responses=False)
    saver = RedisCheckpointSaver(r, ttl_s=3600)
    await _write_one(saver)
    keys = [k async for k in r.scan_iter(match=b"omni:lg:thread-ttl-1*")]
    assert len(keys) == 3  # main hash, index, writes
    for k in keys:
        assert 0 < await r.ttl(k) <= 3600


async def test_threads_and_prefixes_are_isolated():
    r = fakeredis.FakeAsyncRedis(decode_responses=False)
    a = RedisCheckpointSaver(r, prefix="tenant-a:")
    b = RedisCheckpointSaver(r, prefix="tenant-b:")
    await _write_one(a, "shared-thread")
    assert await b.aget_tuple({"configurable": {"thread_id": "shared-thread"}}) is None
    assert await a.aget_tuple({"configurable": {"thread_id": "shared-thread"}}) is not None
    await a.adelete_thread("shared-thread")
    assert [k async for k in r.scan_iter(match=b"tenant-a:*")] == []


@pytest.mark.parametrize("thread", ["glob*chars?[x]", "with:colons:in:it"])
async def test_odd_thread_ids_round_trip_and_delete(thread):
    r = fakeredis.FakeAsyncRedis(decode_responses=False)
    saver = RedisCheckpointSaver(r)
    await _write_one(saver, thread)
    tup = await saver.aget_tuple({"configurable": {"thread_id": thread}})
    assert tup.pending_writes == [("task-1", "messages", "hi")]
    await saver.adelete_thread(thread)
    assert await r.dbsize() == 0
