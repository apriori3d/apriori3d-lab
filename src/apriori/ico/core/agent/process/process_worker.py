from collections.abc import Callable
from multiprocessing import Queue
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import TYPE_CHECKING, Any, Generic, final

from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.ico.core.agent.process.messages import (
    AcknowledgePayload,
    ErrorPayload,
    ExecutionStatePayload,
    InputPayload,
    MessageType,
    OutputPayload,
    WorkerMessage,
)
from apriori.ico.core.agent.process.worker_protocol import WorkerProtocol
from apriori.ico.core.runtime.execution import IcoExecutionMixin, IcoExecutionState
from apriori.ico.core.runtime.lifecycle import (
    IcoLifecycleMixin,
)
from apriori.ico.core.runtime.progress import ProgressMixin
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

if TYPE_CHECKING:
    WorkerQueue = Queue[WorkerMessage[Any]]
else:
    WorkerQueue = Queue  # noqa: F401


@final
class ProcessWorker(
    Generic[I, O],
    #  Implements operator protocol to comply with ICO runtime
    IcoOperatorProtocol[I, O],
    IcoLifecycleMixin,  # Added lifecycle management
    IcoExecutionMixin[I, O],  # Added execution state tracking
    ProgressMixin,  # Added progress tracking
):
    """
    Worker that executes an IcoOperator in a separate process.

    Used exclusively by IcoProcessAgent to isolate operator execution.

    The worker receives messages via in_queue and responds via out_queue,
    following a strict message protocol and acknowledging all commands.

    ICO form:
        I → O (executed in subprocess)

    Lifecycle:
        - Created via ProcessWorker.spawn()
        - run_loop() starts event-driven execution
        - Exits gracefully on cleanup event or failure

    Supported message protocol:
        • InputPayload[I]            → OutputPayload[O]
        • LifecycleEventPayload      → AcknowledgePayload
        • ExecutionStatePayload      → emitted during execution
        • ErrorPayload               → on fault

    Note:
        This class is never instantiated directly in main process;
        it is constructed and run inside a separate process.

    Example:
        Agent → spawn(ProcessWorker)
             → run_loop() waits for Input/Lifecycle messages
             ← responds with Output, Ack, Fault, or State
    """

    __slots__ = (
        "fn",
        "name",
        "in_queue",
        "out_queue",
        "children",
        "node_type",
    )
    # IcoOperatorProtocol attributes
    fn: Callable[[I], O]
    name: str
    node_type: NodeType
    children: list[IcoOperatorProtocol[Any, Any]]

    # Communication queues
    in_queue: WorkerQueue
    out_queue: WorkerQueue

    def __init__(
        self,
        operator_factory: Callable[[], IcoOperatorProtocol[I, O]],
        in_queue: WorkerQueue,
        out_queue: WorkerQueue,
        name: str | None = None,
    ):
        IcoLifecycleMixin.__init__(self)
        IcoExecutionMixin.__init__(self)
        super().__init__()

        operator = operator_factory()
        self.fn = operator
        self.name = name or f"ProcessWorker-{id(self)}"
        self.children = [operator]
        self.in_queue = in_queue
        self.out_queue = out_queue

    # ──── Main loop ────

    def run_loop(self) -> None:
        protocol = WorkerProtocol[I, O](self.fn, name=f"WorkerProtocol({self.name})")

        while True:
            msg = self.in_queue.get()
            response = protocol.handle(msg)
            self.out_queue.put(response)
            if msg.message_type is MessageType.shutdown:
                break

        # Process requests until a cleanup event is received
        while True:
            try:
                message = self.in_queue.get()

                if not isinstance(message, WorkerMessage):
                    raise TypeError(f"Invalid message type: {type(message)}")

                match message.message_type:
                    case MessageType.lifecycle_event:
                        # Broadcast lifecycle event to hosted operator
                        self.broadcast_event(message.payload.event)

                        # Acknowledge lifecycle event
                        self.out_queue.put(
                            WorkerMessage.create(
                                AcknowledgePayload(message.message_type)
                            )
                        )

                    case MessageType.input:
                        # Acknowledge input message
                        self.out_queue.put(
                            WorkerMessage.create(
                                AcknowledgePayload(message.message_type)
                            )
                        )
                        # Execute operator function
                        self._call_fn(message.payload)

                    case MessageType.shutdown:
                        self.progress.print(
                            f"❎ Worker {self.name} received shutdown event. Exiting loop."
                        )
                        self.out_queue.put(
                            WorkerMessage.create(
                                AcknowledgePayload(message.message_type)
                            )
                        )
                        print(f"Worker {self.name} shutting down.")
                        break

            except Exception as e:
                self.progress.print(f"Worker {self.name} encountered an error: {e}")
                self.out_queue.put(WorkerMessage.create(ErrorPayload(error=repr(e))))
                break  # Exit on error

    # ──── Execute operator ────

    def _call_fn(self, payload: InputPayload[I]) -> None:
        self.progress.print(f"🚀 Worker {self.name} processing item: {payload.input}")

        output = self(payload.input)

        self.progress.print(f"✔️ Worker {self.name} completed item: {payload.input}")
        print(f"Worker {self.name} sending output: {output}")
        self.out_queue.put(WorkerMessage.create(OutputPayload(output)))

    def __call__(self, item: I) -> O:
        # Call for standard operator with input
        return self.track(self.fn, item)

    # ─── Asynchronous execution  ───

    async def run_async(self, item: I) -> O:
        raise RuntimeError("ProcessWorker does not support async execution.")

    # ─── Execution event hook ───

    def on_exec_event(self, state: IcoExecutionState) -> None:
        # Notify host about execution state changes
        self.out_queue.put(WorkerMessage.create(ExecutionStatePayload(state)))

    # ──── Spawn Worker ────

    @staticmethod
    def spawn(
        *,
        mp_context: SpawnContext,
        in_queue: WorkerQueue,
        out_queue: WorkerQueue,
        operator_factory: Callable[[], Any],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> SpawnProcess:
        # Create and start agent process
        process = mp_context.Process(
            target=ProcessWorker._process_fn,
            args=(in_queue, out_queue, operator_factory, name, relay_progress),
        )
        process.start()
        return process

    @staticmethod
    def _process_fn(
        in_queue: WorkerQueue,
        out_queue: WorkerQueue,
        operator_factory: Callable[[], Any],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> None:
        worker = ProcessWorker[I, O](
            operator_factory=operator_factory,
            in_queue=in_queue,
            out_queue=out_queue,
            name=name,
        )
        if relay_progress:
            worker.progress = ProgressRelay(out_queue)

        worker.run_loop()
