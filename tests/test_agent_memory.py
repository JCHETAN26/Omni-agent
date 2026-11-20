"""RedisCheckpointSaver: LangGraph's official conformance suite + TTL/isolation."""

import asyncio

import fakeredis
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from redis.exceptions import ConnectionError

from backend.agent.memory import RedisCheckpointSaver, ThreadLeaseLostError


@checkpointer_test(name="RedisCheckpointSaver")
async def redis_saver():
    yield RedisCheckpointSaver(fakeredis.FakeAsyncRedis(decode_responses=False))


async def test_passes_langgraph_checkpointer_conformance_suite():
    report = await validate(redis_saver)
    failures = {
        name: [f for f in r.failures] for name, r in report.results.items() if r.passed is False
    }
    assert report.passed_all_base(), failures
    base = ("put", "put_writes", "get_tuple", "list", "delete_thread")
    assert sum(report.results[c].tests_passed for c in base) >= 58


@pytest.mark.filterwarnings("ignore:Call to .*get_connection.*:DeprecationWarning")
async def test_checkpoint_pool_waits_for_a_slot_with_a_deadline(monkeypatch):
    saver = RedisCheckpointSaver.from_url("redis://127.0.0.1:6379?max_connections=1")
    pool = saver.r.connection_pool

    async def skip_connect(_connection):
        pass

    async def no_data(_connection):
        return False

    monkeypatch.setattr(pool.connection_class, "connect", skip_connect)
    monkeypatch.setattr(pool.connection_class, "can_read", no_data, raising=False)
    monkeypatch.setattr(pool.connection_class, "can_read_destructive", no_data)
    try:
        assert pool.max_connections == 1
        held = await pool.get_connection("GET")
        try:
            waiter = asyncio.create_task(pool.get_connection("GET"))
            await asyncio.sleep(0)
            failed_immediately = waiter.done()
        finally:
            await pool.release(held)
        assert await asyncio.wait_for(waiter, 1) is held
        assert not failed_immediately
        monkeypatch.setattr(pool, "timeout", 0.01)
        with pytest.raises(ConnectionError, match="No connection available"):
            await pool.get_connection("GET")
        await pool.release(held)
    finally:
        await saver.aclose()


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


async def test_expired_turn_cannot_write_or_release_the_new_owners_lease():
    async with fakeredis.FakeAsyncRedis() as r:
        saver = RedisCheckpointSaver(r)
        thread = "thread-ttl-1"
        cfg = await _write_one(saver, thread)
        key = saver._turn_key(thread)
        async with saver.turn_lock(thread):
            await r.set(key, b"new-owner", px=30_000)
            with pytest.raises(ThreadLeaseLostError):
                await _write_one(saver, thread)
            with pytest.raises(ThreadLeaseLostError):
                await saver.aput_writes(cfg, [("messages", "stale")], task_id="stale")
            with pytest.raises(ThreadLeaseLostError):
                await saver.verify_turn_owner(thread)

        assert await r.get(key) == b"new-owner"
        assert (await saver.aget_tuple(cfg)).pending_writes == [("task-1", "messages", "hi")]


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
