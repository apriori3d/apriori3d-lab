import asyncio
import random
import time

import pytest

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.dsl.parallel_stream import ParallelStream


def test_parallel_stream_basic() -> None:
    """Ensure all items are processed by ParallelStream."""
    ops = [IcoOperator[int, int](lambda x: x * 2) for _ in range(3)]
    stream = ParallelStream[int, int](ops)

    data = [1, 2, 3, 4, 5]
    result = list(stream(iter(data)))
    assert result == [x * 2 for x in data]


def test_parallel_stream_parallel_execution() -> None:
    """Simulate random processing delays to ensure concurrency."""

    async def delayed_op(x):
        await asyncio.sleep(random.uniform(0.05, 0.2))
        return x * 10

    ops = [IcoOperator(delayed_op) for _ in range(3)]
    stream = ParallelStream(ops)

    data = list(range(10))
    t0 = time.perf_counter()
    result = list(stream(iter(data)))
    t1 = time.perf_counter()

    # All items processed
    assert len(result) == len(data)
    # Should complete faster than sequential 10 * 0.1s = 1s
    assert (t1 - t0) < 0.8


def test_parallel_stream_ordered() -> None:
    """Test ordered=True preserves order of results."""

    def delayed_double(x):
        time.sleep(0.05 if x % 2 == 0 else 0.01)
        return x * 2

    ops = [IcoOperator(delayed_double) for _ in range(3)]
    stream = ParallelStream(ops, ordered=True)

    data = [1, 2, 3, 4, 5, 6]
    result = list(stream(iter(data)))

    assert result == [x * 2 for x in data]


def test_parallel_stream_unordered() -> None:
    """Test ordered=False change order of results."""

    async def delayed_double(x):
        await asyncio.sleep(x * 0.01)
        return x * 2

    data = [1, 2, 3, 4, 5, 6]
    ops = [IcoOperator(delayed_double) for _ in range(len(data))]
    stream = ParallelStream(ops, ordered=False)
    result = list(stream(reversed(data)))

    assert result == [x * 2 for x in data]


def test_parallel_stream_exception() -> None:
    """Test that exceptions in operators are propagated."""

    def faulty_op(x):
        if x == 3:
            raise ValueError("boom")
        return x

    ops = [IcoOperator(faulty_op) for _ in range(2)]
    stream = ParallelStream(ops)

    data = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        list(stream(iter(data)))


def test_parallel_stream_async_operator() -> None:
    """Ensure async operators work transparently."""

    async def async_double(x):
        await asyncio.sleep(0.01)
        return x * 2

    ops = [IcoOperator(async_double) for _ in range(2)]
    stream = ParallelStream(ops)

    data = [1, 2, 3, 4]
    result = list(stream(iter(data)))
    assert sorted(result) == sorted([x * 2 for x in data])


if __name__ == "__main__":
    test_parallel_stream_unordered()

    # import sys

    # import pytest

    # sys.exit(pytest.main([__file__]))
