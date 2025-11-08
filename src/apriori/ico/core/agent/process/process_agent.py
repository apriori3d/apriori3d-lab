import multiprocessing as mp
from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import TYPE_CHECKING, Any, Generic, cast

from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.ico.core.agent.process.messages import (
    AcknowledgePayload,
    ErrorPayload,
    ExecutionStatePayload,
    InputPayload,
    LifecycleEventPayload,
    MessageType,
    OutputPayload,
    PayloadT,
    ShutdownPayload,
    WorkerMessage,
)
from apriori.ico.core.agent.process.process_worker import ProcessWorker
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.progress import ProgressMixin
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

if TYPE_CHECKING:
    WorkerQueue = Queue[WorkerMessage[Any]]
else:
    WorkerQueue = Queue  # noqa: F401


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

    _operator_factory: Callable[[], IcoOperatorProtocol[I, O]]
    _operator_mirror: IcoOperatorProtocol[I, O]
    _mp_context: SpawnContext
    _in_queue: WorkerQueue
    _out_queue: WorkerQueue
    worker_process: SpawnProcess | None

    def __init__(
        self,
        name: str,
        operator_factory: Callable[[], IcoOperatorProtocol[I, O]],
    ) -> None:
        operator_mirror = operator_factory()

        # Initialize base IcoOperator with process-bound fn
        super().__init__(
            fn=self._fn_on_worker,
            name=name,
            node_type=NodeType.agent,
            children=[operator_mirror],
        )
        # Initialize mp queues
        self._mp_context = mp.get_context("spawn")
        self._in_queue = self._mp_context.Queue()
        self._out_queue = self._mp_context.Queue()
        self._operator_factory = operator_factory
        self.worker_process = None

    # ─── Main call interface ───

    def _fn_on_worker(self, item: I) -> O:
        """
        Sends input to worker and waits for output.
        Used as __call__ delegate in process mode.
        """
        self._send_payload(InputPayload[I](item))
        return self._wait_for_payload(OutputPayload[O]).output

    # ─── Message send/receive helpers ───

    def _send_payload(self, payload: PayloadT, ack: bool = True) -> None:
        """
        Sends a payload to the worker and waits for acknowledgment.
        """
        if self.worker_process is None or not self.worker_process.is_alive():
            raise RuntimeError("Worker process is not running.")

        message = WorkerMessage(payload.message_type, payload)
        self._in_queue.put(message)
        if ack:
            self._acknowledge(message.message_type)

    def _acknowledge(self, message_type: MessageType) -> None:
        """
        Waits for AcknowledgePayload of a given message type.
        """
        if self.worker_process is None or not self.worker_process.is_alive():
            raise RuntimeError("Worker process is not running.")

        ack = self._wait_for_payload(AcknowledgePayload)
        if ack.ack_message_type is not message_type:
            raise RuntimeError(
                f"Unexpected acknowledgment: expected {message_type}, got {ack.ack_message_type}"
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
        if self.worker_process is None or not self.worker_process.is_alive():
            raise RuntimeError("Worker process is not running.")

        message_type = payload_type.get_message_type()

        while True:
            message = self._out_queue.get(timeout=timeout)

            if ProgressRelay.handle_message(self.progress, message):
                continue  # progress message handled

            match message.message_type:
                case MessageType.fault:
                    raise RuntimeError(
                        f"Worker fault: {cast(ErrorPayload, message.payload).error}"
                    )

                case MessageType.execution_event:
                    self._set_exec_state(
                        cast(ExecutionStatePayload, message.payload).state
                    )
                    continue  # wait for target response

            if message.message_type is not message_type:
                raise RuntimeError(
                    f"Unexpected message type: expected {message_type}, got {message.payload.message_type}"
                )

            return cast(PayloadT, message.payload)

    # ─── Lifecycle coordination ───

    def on_event(self, event: IcoRuntimeCommand) -> None:
        """
        Forwards lifecycle events to the worker process.
        Manages worker lifecycle on prepare/cleanup.
        """
        super().on_event(event)

        match event:
            case IcoRuntimeCommand.activate:
                # Start fresh worker and forward prepare
                self.worker_process = ProcessWorker[I, O].spawn(
                    mp_context=self._mp_context,
                    in_queue=self._in_queue,
                    out_queue=self._out_queue,
                    operator_factory=self._operator_factory,
                )
                self._send_payload(LifecycleEventPayload(event))

            case IcoRuntimeCommand.deavtivate:
                self._send_payload(LifecycleEventPayload(event))
                self._send_payload(ShutdownPayload())
                self._shutdown_worker()

            case _:
                # Forward all other lifecycle events
                self._send_payload(LifecycleEventPayload(event))

    def _shutdown_worker(self) -> None:
        if self.worker_process is None:
            return
        try:
            # Gracefully join the worker process
            if self.worker_process and self.worker_process.is_alive():
                self.worker_process.join(timeout=5)

                # Close queues
                self._in_queue.close()
                self._in_queue.join_thread()
                self._out_queue.close()
                self._out_queue.join_thread()
                print(f"Process Agent {self.name} worker joined.")

                # Check if worker exited properly
                if self.worker_process.exitcode is None:
                    self.progress.print("⚠️ Worker did not exit (possibly stuck)")

                elif self.worker_process.exitcode != 0:
                    self.progress.print(
                        f"⚠️ Process Agent {self.name} worker exited with code {self.worker_process.exitcode}."
                    )
        except Exception as e:
            self.progress.print(f"❌ Error while stopping agent {self.name}: {e}")

        finally:
            if self.worker_process.is_alive():
                self.progress.print(
                    f"⚠️ Process Agent {self.name} did not terminate worker gracefully."
                )
                self.worker_process.terminate()
