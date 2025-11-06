from collections.abc import Callable
from multiprocessing import Process
from typing import Any, Generic, cast

from torch.multiprocessing import Queue

from apriori.ico.core.agent.process.messages import (
    AcknowledgePayload,
    ErrorPayload,
    ExecutionStatePayload,
    InputPayload,
    LifecycleEventPayload,
    MessageType,
    OutputPayload,
    PayloadT,
    WorkerMessage,
)
from apriori.ico.core.agent.process.process_worker import ProcessWorker
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.lifecycle import IcoLifecycleEvent
from apriori.ico.core.runtime.progress import ProgressMixin
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O


class IcoProcessAgent(
    Generic[I, O],
    IcoOperator[I, O],
    ProgressMixin,
):
    """
    Executes an IcoOperator in a separate process via ProcessWorker.

    Acts as a bridge between ICO runtime and a remote worker process using
    multiprocessing.Queue-based communication with WorkerMessage protocol.

    ICO form:
        I → O  (executed via subprocess)

    Worker lifecycle:
        • Process is spawned on IcoLifecycleEvent.prepare
        • Process is terminated on IcoLifecycleEvent.cleanup

    Supported communication:
        • InputPayload[I]         → OutputPayload[O]
        • LifecycleEventPayload   → AcknowledgePayload
        • ExecutionStatePayload   → emitted during execution
        • ErrorPayload            ← on fault

    This agent enables execution isolation and scalability for any operator
    by offloading computation into an external process.
    """

    operator_factory: Callable[[], IcoOperatorProtocol[I, O]]
    operator_mirror: IcoOperatorProtocol[I, O]
    in_queue: Queue[WorkerMessage[Any]]
    out_queue: Queue[WorkerMessage[Any]]
    worker_process: Process

    def __init__(
        self,
        name: str,
        operator_factory: Callable[[], IcoOperatorProtocol[I, O]],
    ) -> None:
        # Initialize local operator and queues
        operator_mirror = operator_factory()
        in_queue = Queue[WorkerMessage[Any]]()
        out_queue = Queue[WorkerMessage[Any]]()

        # Initialize base IcoOperator with process-bound fn
        super().__init__(
            fn=self._fn_on_worker,
            name=name,
            node_type=NodeType.agent,
            children=[operator_mirror],
        )

        # Store references and spawn worker process
        self.operator_factory = operator_factory
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.worker_process = ProcessWorker[I, O].spawn(
            operator_factory=operator_factory,
            in_queue=self.in_queue,
            out_queue=self.out_queue,
        )

    # ─── Main call interface ───

    def _fn_on_worker(self, item: I) -> O:
        """
        Sends input to worker and waits for output.
        Used as __call__ delegate in process mode.
        """
        self._send_payload(InputPayload[I](item))
        return self._wait_for_payload(OutputPayload[O]).output

    # ─── Message send/receive helpers ───

    def _send_payload(self, payload: PayloadT) -> None:
        """
        Sends a payload to the worker and waits for acknowledgment.
        """
        message = WorkerMessage(payload.message_type, payload)
        self.in_queue.put(message)
        self._acknowledge(message.type)

    def _acknowledge(self, message_type: MessageType) -> None:
        """
        Waits for AcknowledgePayload of a given message type.
        """
        ack = self._wait_for_payload(AcknowledgePayload)
        if ack.message_type is not message_type:
            raise RuntimeError(
                f"Unexpected acknowledgment: expected {message_type}, got {ack.message_type}"
            )

    def _wait_for_payload(
        self,
        payload_type: type[PayloadT],
        timeout: float | None = 10,
    ) -> PayloadT:
        """
        Waits for a specific response payload from the worker.
        Handles execution events and fault messages.
        """
        message_type = payload_type.__worker_message_type__

        while True:
            message = self.out_queue.get(timeout=timeout)

            match message.type:
                case MessageType.fault:
                    raise RuntimeError(
                        f"Worker fault: {cast(ErrorPayload, message.payload).error}"
                    )

                case MessageType.execution_event:
                    self._set_exec_state(
                        cast(ExecutionStatePayload, message.payload).state
                    )
                    continue  # wait for target response

            if message.payload.message_type is not message_type:
                raise RuntimeError(
                    f"Unexpected message type: expected {message_type}, got {message.payload.message_type}"
                )
            if not isinstance(message.payload, payload_type):
                raise TypeError(
                    f"Invalid payload type: expected {payload_type.__name__}, got {type(message.payload).__name__}"
                )

            return message.payload

    # ─── Lifecycle coordination ───

    def on_event(self, event: IcoLifecycleEvent) -> None:
        """
        Forwards lifecycle events to the worker process.
        Manages worker lifecycle on prepare/cleanup.
        """
        super().on_event(event)

        match event:
            case IcoLifecycleEvent.prepare:
                # Start fresh worker and forward prepare
                self.worker_process = ProcessWorker[I, O].spawn(
                    operator_factory=self.operator_factory,
                    in_queue=self.in_queue,
                    out_queue=self.out_queue,
                )
                self._send_payload(LifecycleEventPayload(event))

            case IcoLifecycleEvent.cleanup:
                # Forward cleanup, then terminate worker
                self._send_payload(LifecycleEventPayload(event))

                try:
                    if self.worker_process and self.worker_process.is_alive():
                        self.worker_process.terminate()
                        self.worker_process.join(timeout=1)

                        if self.worker_process.exitcode is None:
                            self.progress.print(
                                f"⚠️ Process Agent {self.name} did not terminate worker cleanly."
                            )
                except Exception as e:
                    self.progress.print(
                        f"❌ Error while stopping agent {self.name}: {e}"
                    )

            case _:
                # Forward all other lifecycle events
                self._send_payload(LifecycleEventPayload(event))
