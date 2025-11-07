# import multiprocessing
# from collections.abc import Callable
# from dataclasses import dataclass, field
# from multiprocessing import Queue
# from typing import TYPE_CHECKING, Any

# import pytest

# from apriori.ico.core import (
#     IcoLifecycleEvent,
#     IcoProcessAgent,
# )
# from apriori.ico.core.agent.process.messages import (
#     AcknowledgePayload,
#     ExecutionStatePayload,
#     InputPayload,
#     LifecycleEventPayload,
#     MessageType,
#     OutputPayload,
#     ShutdownPayload,
#     WorkerMessage,
# )
# from apriori.ico.core.agent.process.process_worker import ProcessWorker
# from apriori.ico.core.runtime.execution import IcoExecutionState
# from apriori.ico.core.runtime.lifecycle import IcoLifecycleMixin
# from apriori.ico.core.types import IcoOperatorProtocol, NodeType

# if TYPE_CHECKING:
#     WorkerQueue = Queue[WorkerMessage[Any]]
# else:
#     WorkerQueue = Queue  # noqa: F401

# # ──── Test Operators ────


# @dataclass
# class EchoOperator(IcoOperatorProtocol[int, int]):
#     name: str = "Echo"
#     node_type: NodeType = NodeType.operator
#     children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
#     fn: Callable[[int], int] = lambda x: x

#     def __call__(self, item: int) -> int:
#         return item

#     async def run_async(self, item: int) -> int:
#         return item

#     @staticmethod
#     def create() -> IcoOperatorProtocol[int, int]:
#         return EchoOperator()


# @dataclass
# class FailingOperator(IcoOperatorProtocol[int, int]):
#     name: str = "Fail"
#     node_type: NodeType = NodeType.operator
#     children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)
#     fn: Callable[[int], int] = lambda x: x

#     def __call__(self, item: int) -> int:
#         raise ValueError("Intentional failure")

#     async def run_async(self, item: int) -> int:
#         raise ValueError("Intentional async failure")

#     @staticmethod
#     def create() -> IcoOperatorProtocol[int, int]:
#         return FailingOperator()


# @dataclass
# class NestedRecordingOperator(
#     IcoOperatorProtocol[int, list[IcoLifecycleEvent]],
#     IcoLifecycleMixin,  # Added lifecycle support to allow event recording
# ):
#     name: str = "NestedRecorder"
#     node_type: NodeType = NodeType.operator
#     children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)

#     # Mock function
#     fn: Callable[[int], list[IcoLifecycleEvent]] = lambda x: [IcoLifecycleEvent.prepare]

#     received_events: list[IcoLifecycleEvent] = field(default_factory=list)

#     def __call__(self, item: int) -> list[IcoLifecycleEvent]:
#         return self.received_events

#     async def run_async(self, _: int) -> list[IcoLifecycleEvent]:
#         raise NotImplementedError()

#     def on_event(self, event: IcoLifecycleEvent) -> None:
#         self.received_events.append(event)

#     @staticmethod
#     def create() -> IcoOperatorProtocol[int, list[IcoLifecycleEvent]]:
#         return NestedRecordingOperator()


# @dataclass
# class RecordingOperator(
#     IcoOperatorProtocol[int, list[IcoLifecycleEvent]],
#     IcoLifecycleMixin,  # Added lifecycle support to allow event recording
# ):
#     name: str = "Recorder"
#     node_type: NodeType = NodeType.operator
#     children: list[IcoOperatorProtocol[Any, Any]] = field(default_factory=list)

#     # Create nested recorder
#     fn: Callable[[int], list[IcoLifecycleEvent]] = field(
#         default_factory=NestedRecordingOperator.create
#     )

#     def __call__(self, item: int) -> list[IcoLifecycleEvent]:
#         return self.fn(item)

#     async def run_async(self, _: int) -> list[IcoLifecycleEvent]:
#         raise NotImplementedError()

#     @staticmethod
#     def create() -> IcoOperatorProtocol[int, list[IcoLifecycleEvent]]:
#         return RecordingOperator()


# def test_process_worker_lifecycle_events() -> None:
#     in_q: WorkerQueue = WorkerQueue()
#     out_q: WorkerQueue = WorkerQueue()

#     proc = ProcessWorker.spawn(
#         in_queue=in_q,
#         out_queue=out_q,
#         operator_factory=RecordingOperator.create,
#         name="TestWorker",
#     )

#     # Events to test
#     all_events = list(IcoLifecycleEvent)

#     # Send events and verify acknowledgments
#     for event in all_events:
#         in_q.put(WorkerMessage.create(LifecycleEventPayload(event)))
#         msg = out_q.get(timeout=5)

#         assert isinstance(msg.payload, AcknowledgePayload)
#         assert msg.payload.message_type == MessageType.lifecycle_event

#     # Shut down the worker
#     in_q.put(WorkerMessage.create(ShutdownPayload()))
#     msg = out_q.get(timeout=5)

#     assert isinstance(msg.payload, AcknowledgePayload)
#     assert msg.payload.message_type == MessageType.shutdown
#     proc.join(timeout=2)

#     assert not proc.is_alive(), "ProcessWorker did not exit on cleanup"
#     assert proc.exitcode == 0, f"ProcessWorker exited with code {proc.exitcode}"


# # ──── ProcessWorker Lifecycle Event Forwarding Test ────


# def test_process_worker_lifecycle_event_forwarding() -> None:
#     in_q: WorkerQueue = WorkerQueue()
#     out_q: WorkerQueue = WorkerQueue()

#     proc = ProcessWorker.spawn(
#         in_queue=in_q,
#         out_queue=out_q,
#         operator_factory=NestedRecordingOperator.create,
#         name="LifecycleRecorder",
#     )

#     all_events = list(IcoLifecycleEvent)

#     # Events to send (excluding cleanup for now)
#     for event in all_events:
#         in_q.put(WorkerMessage.create(LifecycleEventPayload(event)))
#         msg = out_q.get(timeout=5)
#         assert isinstance(msg.payload, AcknowledgePayload)

#     # Collect events received by the operator via output
#     in_q.put(WorkerMessage.create(InputPayload(0)))
#     while True:
#         msg = out_q.get(timeout=5)
#         if isinstance(msg.payload, OutputPayload):
#             break

#     # Verify that all events were forwarded to the operator
#     assert msg.payload.output == all_events

#     # Shut down the worker
#     in_q.put(WorkerMessage.create(ShutdownPayload()))
#     msg = out_q.get(timeout=5)
#     assert isinstance(msg.payload, AcknowledgePayload)

#     proc.join(timeout=2)
#     assert not proc.is_alive()


# # ──── ProcessWorker Tests ────


# def test_process_worker_execution() -> None:
#     in_queue = WorkerQueue()
#     out_queue = WorkerQueue()

#     proc = ProcessWorker.spawn(
#         in_queue=in_queue,
#         out_queue=out_queue,
#         operator_factory=EchoOperator.create,
#         name="TestWorker",
#     )

#     in_queue.put(WorkerMessage.create(InputPayload(42)))

#     outputs = []
#     received = 0

#     # Receive two messages: input ack and output
#     while received < 4:
#         msg = out_queue.get(timeout=10)
#         outputs.append(msg.payload)
#         received += 1

#     assert isinstance(outputs[0], AcknowledgePayload)
#     assert isinstance(outputs[1], ExecutionStatePayload)
#     assert isinstance(outputs[2], ExecutionStatePayload)
#     assert isinstance(outputs[3], OutputPayload)
#     assert outputs[0].message_type == MessageType.input
#     assert outputs[1].state == IcoExecutionState.running
#     assert outputs[2].state == IcoExecutionState.done
#     assert outputs[3].output == 42

#     # Shut down the worker
#     in_queue.put(WorkerMessage.create(ShutdownPayload()))
#     msg = out_queue.get(timeout=5)

#     assert isinstance(msg.payload, AcknowledgePayload)
#     proc.join(timeout=2)
#     assert not proc.is_alive()


# # ─── Agent Tests ───


# def test_process_agent_basic() -> None:
#     agent = IcoProcessAgent[int, int](
#         name="TestAgent", operator_factory=EchoOperator.create
#     )

#     agent.on_event(IcoLifecycleEvent.prepare)
#     result = agent(123)
#     assert result == 123
#     agent.on_event(IcoLifecycleEvent.cleanup)


# def test_process_agent_fault() -> None:
#     agent = IcoProcessAgent(name="TestAgent", operator_factory=FailingOperator.create)

#     agent.on_event(IcoLifecycleEvent.prepare)

#     with pytest.raises(RuntimeError, match="Worker fault:"):
#         agent(999)

#     agent.on_event(IcoLifecycleEvent.cleanup)


# if __name__ == "__main__":
#     multiprocessing.set_start_method("spawn", force=True)

#     test_process_worker_lifecycle_event_forwarding()

#     # import sys

#     # import pytest

#     # sys.exit(pytest.main([__file__]))
