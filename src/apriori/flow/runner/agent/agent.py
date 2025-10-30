import builtins
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Generic, TypeAlias, final

from torch.multiprocessing import Process, Queue

from apriori.flow.progress.types import (
    HasProgress,
    ProgressMixin,
)
from apriori.flow.runner.agent.messages import (
    AgentFaultPayload,
    AgentMessage,
    LifecyclePayload,
    LifecycleResponsePayload,
    RestoreStatePayload,
    RunPayload,
    RunResponsePayload,
    create_message,
)
from apriori.flow.runner.types import (
    LifecycleProtocol,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    RunnerInputType,
    RunnerProtocol,
    RunnerResult,
)

AgentRunnerType: TypeAlias = RunnerProtocol[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


class RunnerAgentState(Enum):
    waiting = auto()
    running = auto()
    failed = auto()


@final
@dataclass(slots=True)
class RunnerAgentLink(
    Generic[RunnerInputType],
    LifecycleProtocol,
):
    agent_id: int
    process: Process
    queue: Queue
    task: int
    state: RunnerAgentState = RunnerAgentState.waiting

    _input_indices: list[int] = None

    # Lifecycle methods

    def prepare(self) -> None:
        self.queue.put(self.agent_id, create_message(LifecyclePayload("prepare")))
        self.running()

    def on_cycle_start(self) -> None:
        self.queue.put(
            self.agent_id, create_message(LifecyclePayload("on_cycle_start"))
        )
        self.running()

    def on_cycle_end(self) -> None:
        self.queue.put(self.agent_id, create_message(LifecyclePayload("on_cycle_end")))
        self.running()

    def cleanup(self) -> None:
        self.queue.put(self.agent_id, create_message(LifecyclePayload("cleanup")))
        self.running()

    def load_state(self, state: dict) -> None:
        self.queue.put(self.agent_id, create_message(RestoreStatePayload(state)))
        self.running()

    # Execution method

    def run(self, input: RunnerInputType, input_indices: list[int]) -> None:
        self.queue.put(create_message(RunPayload(input)))
        self._input_indices = input_indices
        self.running()

    @property
    def input_indices(self) -> list[int]:
        return self._input_indices

    # State management methods

    def waiting(self) -> None:
        self.state = RunnerAgentState.waiting

    def running(self) -> None:
        self.state = RunnerAgentState.running

    def failed(self) -> None:
        self.state = RunnerAgentState.failed

    @property
    def is_waiting(self) -> bool:
        return self.state == RunnerAgentState.waiting

    @property
    def is_running(self) -> bool:
        return self.state == RunnerAgentState.running

    @property
    def is_failed(self) -> bool:
        return self.state == RunnerAgentState.failed


# Main Agent class


@final
class RunnerAgent(
    ProgressMixin,
    # Implement runner protocol for consistency
    RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
):
    __slots__ = ("agent_id", "runner", "input_queue", "output_queue", "_results")
    agent_id: int
    runner: AgentRunnerType
    request_queue: Queue
    response_queue: Queue
    _results: list[RunnerResult]
    _input_size: int

    def __init__(
        self,
        agent_id: int,
        runner: AgentRunnerType,
        request_queue: Queue,
        response_queue: Queue,
    ):
        self.agent_id = agent_id
        self.runner = runner
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.runner.on_result = self._on_result
        self._results = None
        self._input_size = 0

    def run_loop(self) -> None:
        # Process requests until a None request is received
        while True:
            try:
                message = self.request_queue.get()
                self.progress.print(
                    f"Agent {self.agent_id} received message: {message.type}"
                )

                self._handle_message(message)

                self.progress.print(
                    f"Agent {self.agent_id} processed message: {message.type}"
                )

                if message.type == "cleanup":
                    break

            except Exception as e:
                self.progress.print(f"Agent {self.agent_id} encountered an error: {e}")
                self.response_queue.put(
                    create_message(self.agent_id, AgentFaultPayload(error=repr(e)))
                )
                break  # Exit on error

    # Message handlers

    def _handle_message(self, message: AgentMessage[Any]) -> None:
        if message.agent_id != self.agent_id:
            raise ValueError(
                f"Message agent_id {message.agent_id} does not match this agent_id {self.agent_id}"
            )
        match message.type:
            case "lifecycle":
                self._handle_lifecycle_message(message.payload)
            case "run":
                self._handle_run_message(message.payload)
            case _:
                raise TypeError(f"Unknown message type: {message.type}")

    def _handle_lifecycle_message(self, lifecycle: LifecyclePayload) -> None:
        self.progress.print(
            f"Agent {self.agent_id} received lifecycle message: {lifecycle.phase}"
        )
        match lifecycle.phase:
            case "prepare":
                self.prepare()
            case "on_cycle_start":
                self.on_cycle_start()
            case "on_cycle_end":
                self.on_cycle_end()
            case "cleanup":
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
            f"Agent {self.agent_id} received run message: {payload.items}"
        )
        self.runner.input = payload.items
        # Store input size for progress tracking
        self._input_size = len(payload.items)
        # Storage for results
        self._results = []
        # Execute the runner
        self.run()

    # Lifecycle hooks

    def prepare(self) -> None:
        if self.progress is None:
            raise RuntimeError("Worker progress relay is not set before prepare.")
        # Redirect print to progress routine, because print interferes with rich.Progress
        builtins.print = self.progress.print

        if isinstance(self.runner, HasProgress):
            self.runner.progress = self.progress
            self.runner.shared_task = self.shared_task
        else:
            self.progress.print(
                f"⚠️ Warning: Runner {self.runner} does not support progress relay."
            )
        self.runner.prepare()

    def on_cycle_start(self) -> None:
        self.runner.on_cycle_start()

    def on_cycle_end(self) -> None:
        self.runner.on_cycle_end()

    def cleanup(self) -> None:
        self.runner.cleanup()

    # Main run method

    def run(self) -> None:
        self.runner.run()

    # Result handler

    def _on_result(self, runner_result: RunnerResult) -> None:
        # TODO: Move output to cpu and serialize
        # Send the result back to the main process, skipping large context and pipeline data
        self._results.append(runner_result)

        if len(self._results) == self._input_size:
            self.progress.print(
                f"✔️ Agent {self.agent_id} completed all {self._input_size} items."
            )
            self.response_queue.put(
                create_message(
                    self.agent_id,
                    RunResponsePayload(self._results),
                )
            )


RunnerAgentType: TypeAlias = RunnerAgent[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]
