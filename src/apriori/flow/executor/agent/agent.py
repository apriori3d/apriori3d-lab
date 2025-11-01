import builtins
from enum import Enum, auto
from typing import Any, Generic, TypeAlias, final

from torch.multiprocessing import Queue

from apriori.flow.executor.agent.messages import (
    AgentFaultPayload,
    AgentMessage,
    AgentMessageType,
    LifecyclePayload,
    LifecyclePhasesType,
    LifecycleResponsePayload,
    RunPayload,
    RunResponsePayload,
    create_message,
)
from apriori.flow.executor.types import PipelineExecutorType
from apriori.flow.lifecycle import SupportsLifecycle
from apriori.flow.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
)
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.progress.types import (
    HasProgress,
    SupportsProgressTask,
)
from apriori.flow.runner.types import (
    RunnerInputType,
    RunnerProtocol,
)

AgentRunnerType: TypeAlias = RunnerProtocol[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


class ExecutorAgentState(Enum):
    inactive = auto()
    waiting = auto()
    running = auto()
    failed = auto()


@final
class ExecutorAgent(
    Generic[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    ProgressMixin,  # Add progress support
    SupportsLifecycle,  # The class implements lifecycle support
):
    __slots__ = (
        "agent_id",
        "pipeline_executor",
        "request_queue",
        "response_queue",
    )
    agent_id: int
    pipeline_executor: PipelineExecutorType
    request_queue: Queue
    response_queue: Queue

    def __init__(
        self,
        agent_id: int,
        pipeline_executor: PipelineExecutorType,
        request_queue: Queue,
        response_queue: Queue,
    ):
        self.agent_id = agent_id
        self.pipeline_executor = pipeline_executor
        self.request_queue = request_queue
        self.response_queue = response_queue

    def run_loop(self) -> None:
        # Process requests until a None request is received
        while True:
            try:
                message = self.request_queue.get()
                self._handle_message(message)

                # Exit loop on cleanup message
                if message.type == AgentMessageType.cleanup:
                    break

            except Exception as e:
                self.progress.print(f"Agent {self.agent_id} encountered an error: {e}")
                self.response_queue.put(
                    create_message(self.agent_id, AgentFaultPayload(error=repr(e)))
                )
                break  # Exit on error

    # Main run method
    def run(self, item: PipelineInputType) -> None:
        pipeline_result = self.pipeline_executor.run(item)

        # TODO: Move output to cpu and serialize
        # Send the result back to the main process, skipping large context and pipeline data

        self.progress.print(f"✔️ Agent {self.agent_id} completed item: {item}")
        self.response_queue.put(
            create_message(
                self.agent_id,
                RunResponsePayload(pipeline_result),
            )
        )

    def _handle_message(self, message: AgentMessage[Any]) -> None:
        self.progress.print(f"Agent {self.agent_id} received message: {message.type}")

        if message.agent_id != self.agent_id:
            raise ValueError(
                f"Message agent_id {message.agent_id} does not match this agent_id {self.agent_id}"
            )
        match message.type:
            case AgentMessageType.lifecycle:
                self._handle_lifecycle_message(message.payload)
            case AgentMessageType.run:
                self._handle_run_message(message.payload)
            case _:
                raise TypeError(f"Unknown message type: {message.type}")

        self.progress.print(f"Agent {self.agent_id} processed message: {message.type}")

    def _handle_lifecycle_message(self, lifecycle: LifecyclePayload) -> None:
        self.progress.print(
            f"Agent {self.agent_id} received lifecycle message: {lifecycle.phase}"
        )
        match lifecycle.phase:
            case LifecyclePhasesType.prepare:
                self.prepare()
            case LifecyclePhasesType.on_cycle_start:
                self.on_cycle_start()
            case LifecyclePhasesType.on_cycle_end:
                self.on_cycle_end()
            case LifecyclePhasesType.cleanup:
                self.cleanup()
            case _:
                raise ValueError(f"Unknown lifecycle phase: {lifecycle.phase}")

        self.response_queue.put(
            create_message(
                self.agent_id,
                LifecycleResponsePayload(phase=lifecycle.phase),
            )
        )

    def _handle_run_message(self, payload: RunPayload[RunnerInputType]) -> None:
        self.progress.print(
            f"Agent {self.agent_id} received run message with item {payload.item}"
        )
        self.run(payload.item)

    # Lifecycle method: redirect lifecycle methods to pipeline executor

    def prepare(self) -> None:
        # Redirect print to progress routine, because print interferes with rich.Progress
        builtins.print = self.progress.print

        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.prepare()

    def on_cycle_start(self) -> None:
        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.on_cycle_start()

    def on_cycle_end(self) -> None:
        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.on_cycle_end()

    def cleanup(self) -> None:
        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.cleanup()

    # Progress task support

    def setup_process_task(self):
        super().setup_process_task()

        # Setup executor progress
        if isinstance(self.pipeline_executor, HasProgress):
            self.pipeline_executor.progress = self.progress

        # Enable inline task structure for executor, as agent link already shows overall progress
        # in the main process
        if isinstance(self.pipeline_executor, SupportsProgressTask):
            self.pipeline_executor.enable_task_inline_structure(self.task)
            self.pipeline_executor.setup_process_task()


ExecutorAgentType: TypeAlias = ExecutorAgent[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]
