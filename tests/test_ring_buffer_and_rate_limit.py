import fakeredis
import pytest

from backend.api.ring_buffer import RingBuffer
from backend.core.rate_limit import InMemoryTokenBucket, RedisTokenBucket


async def test_ring_buffer_drops_oldest_when_full():
    buf: RingBuffer[int] = RingBuffer(3)
    assert [buf.push(i) for i in range(3)] == [None, None, None]
    assert buf.push(3) == 0  # evicts the oldest
    assert buf.push(4) == 1
    assert buf.dropped == 2
    assert [await buf.get() for _ in range(3)] == [2, 3, 4]


def test_ring_buffer_rejects_zero_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture(params=["memory", "redis"])
def bucket_factory(request):
    clock = FakeClock()

    def make(capacity: float, rate: float):
        if request.param == "memory":
            return InMemoryTokenBucket(capacity, rate, clock=clock)
        client = fakeredis.FakeAsyncRedis(decode_responses=True)
        return RedisTokenBucket(client, capacity, rate, clock=clock)

    return make, clock


async def test_bucket_allows_burst_then_limits(bucket_factory):
    make, clock = bucket_factory
    bucket = make(capacity=3, rate=1.0)
    results = [(await bucket.acquire("k")).allowed for _ in range(5)]
    assert results == [True, True, True, False, False]
    denied = await bucket.acquire("k")
    assert denied.retry_after_s == pytest.approx(1.0)


async def test_bucket_refills_over_time(bucket_factory):
    make, clock = bucket_factory
    bucket = make(capacity=2, rate=4.0)  # one token every 250 ms
    assert (await bucket.acquire("k")).allowed
    assert (await bucket.acquire("k")).allowed
    assert not (await bucket.acquire("k")).allowed
    clock.t += 0.25
    assert (await bucket.acquire("k")).allowed
    clock.t += 10  # never exceeds capacity
    assert (await bucket.acquire("k", cost=2)).allowed
    assert not (await bucket.acquire("k")).allowed


async def test_bucket_keys_are_independent_and_costs_apply(bucket_factory):
    make, _ = bucket_factory
    bucket = make(capacity=1, rate=1.0)
    assert (await bucket.acquire("a")).allowed
    assert (await bucket.acquire("b")).allowed
    assert not (await bucket.acquire("a")).allowed
    cheap = make(capacity=1, rate=1.0)
    assert all([(await cheap.acquire("c", cost=0.25)).allowed for _ in range(4)])
    assert not (await cheap.acquire("c", cost=0.25)).allowed
