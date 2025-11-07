import multiprocessing
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import TYPE_CHECKING, Any

from apriori.ico.core import (
    IcoLifecycleEvent,
)
from apriori.ico.core.agent.process.messages import (
    ShutdownPayload,
    WorkerMessage,
)
from apriori.ico.core.runtime.lifecycle import IcoLifecycleMixin
from apriori.ico.core.types import IcoOperatorProtocol, NodeType

if TYPE_CHECKING:
    WorkerQueue = Queue[WorkerMessage[Any]]
else:
    WorkerQueue = Queue  # noqa: F401

# ──── Test Operators ────


# ──── Simple Echo Operator ────


@dataclass
class EchoOperator(IcoOperatorProtocol[int, int]):
    name: str = "Echo"
    node_type: NodeType = NodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
    fn: Callable[[int], int] = lambda x: x

    def __call__(self, item: int) -> int:
        return item

    async def run_async(self, item: int) -> int:
        return item

    @staticmethod
    def create() -> IcoOperatorProtocol[int, int]:
        return EchoOperator()


# ──── Failing with Exception Operator ────


@dataclass
class FailingOperator(IcoOperatorProtocol[int, int]):
    name: str = "Fail"
    node_type: NodeType = NodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
    fn: Callable[[int], int] = lambda x: x

    def __call__(self, item: int) -> int:
        raise ValueError("Intentional failure")

    async def run_async(self, item: int) -> int:
        raise ValueError("Intentional async failure")

    @staticmethod
    def create() -> IcoOperatorProtocol[int, int]:
        return FailingOperator()


# ──── Nested Recording Event Operator ────


@dataclass
class NestedRecordingOperator(
    IcoOperatorProtocol[int, list[IcoLifecycleEvent]],
    IcoLifecycleMixin,  # Added lifecycle support to allow event recording
):
    name: str = "NestedRecorder"
    node_type: NodeType = NodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)

    # Mock function
    fn: Callable[[int], list[IcoLifecycleEvent]] = lambda x: [IcoLifecycleEvent.prepare]

    received_events: list[IcoLifecycleEvent] = field(default_factory=list)

    def __call__(self, item: int) -> list[IcoLifecycleEvent]:
        return self.received_events

    async def run_async(self, _: int) -> list[IcoLifecycleEvent]:
        raise NotImplementedError()

    def on_event(self, event: IcoLifecycleEvent) -> None:
        self.received_events.append(event)

    @staticmethod
    def create() -> IcoOperatorProtocol[int, list[IcoLifecycleEvent]]:
        return NestedRecordingOperator()


# ──── Recording Event Operator ────


@dataclass
class RecordingOperator(
    IcoOperatorProtocol[int, list[IcoLifecycleEvent]],
    IcoLifecycleMixin,  # Added lifecycle support to allow event recording
):
    name: str = "Recorder"
    node_type: NodeType = NodeType.operator
    children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)

    # Create nested recorder
    fn: Callable[[int], list[IcoLifecycleEvent]] = field(
        default_factory=NestedRecordingOperator.create
    )

    def __call__(self, item: int) -> list[IcoLifecycleEvent]:
        return self.fn(item)

    async def run_async(self, _: int) -> list[IcoLifecycleEvent]:
        raise NotImplementedError()

    @staticmethod
    def create() -> IcoOperatorProtocol[int, list[IcoLifecycleEvent]]:
        return RecordingOperator()


# ──── Worker Shutdown Helper ────


def shutdown_worker(
    in_q: WorkerQueue, out_q: WorkerQueue, proc: multiprocessing.Process
) -> None:
    in_q.put(WorkerMessage.create(ShutdownPayload()))
    proc.join(timeout=2)

    assert not proc.is_alive()
    assert proc.exitcode == 0, f"ProcessWorker exited with code {proc.exitcode}"
