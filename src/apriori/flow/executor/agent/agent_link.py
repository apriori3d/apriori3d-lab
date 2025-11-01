from collections.abc import Callable
from enum import Enum, auto
from typing import Generic, TypeAlias, final

from torch.multiprocessing import Process, Queue

from apriori.flow.executor.agent.agent import ExecutorAgentType
from apriori.flow.executor.agent.messages import (
    LifecyclePayload,
    LifecyclePhasesType,
    RunPayload,
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
from apriori.flow.progress.progress_relay import ProgressRelay
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
class ExecutorAgentLink(
    Generic[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    SupportsLifecycle,  # The class implements lifecycle support
    ProgressMixin,  # Add progress support
):
    agent_id: int
    state: ExecutorAgentState
    response_queue: Queue
    executor_factory_method: Callable[[], PipelineExecutorType]
    process: Process | None
    _request_queue: Queue
    # Index of the currently processing item
    _input_item_index: int

    def __init__(
        self,
        agent_id: int,
        response_queue: Queue,
        executor_factory_method: Callable[[], PipelineExecutorType],
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.response_queue = response_queue
        self.executor_factory_method = executor_factory_method
        self.state = ExecutorAgentState.inactive
        self.process = None
        self._input_item_index = 0
        self._request_queue = Queue()

    # ──── Execution method ────

    def run(self, input: PipelineInputType, item_index: int) -> None:
        self._input_item_index = item_index
        self._request_queue.put(create_message(RunPayload(input)))
        self.running()

    # ──── Lifecycle methods ────

    def prepare(self) -> None:
        if self.state != ExecutorAgentState.inactive:
            return

        # Spawn process for the agent
        self._spawn_process()

        # Send prepare message to agent
        self._request_queue.put(
            create_message(self.agent_id, LifecyclePayload(LifecyclePhasesType.prepare))
        )
        self.running()

    def on_cycle_start(self) -> None:
        self._request_queue.put(
            create_message(
                self.agent_id, LifecyclePayload(LifecyclePhasesType.on_cycle_start)
            ),
        )
        self.running()

    def on_cycle_end(self) -> None:
        self._request_queue.put(
            create_message(
                self.agent_id, LifecyclePayload(LifecyclePhasesType.on_cycle_end)
            ),
        )
        self.running()

    def cleanup(self) -> None:
        if self.state == ExecutorAgentState.inactive:
            return
        self._request_queue.put(
            create_message(self.agent_id, LifecyclePayload(LifecyclePhasesType.cleanup))
        )
        self.running()

    # ──── Progress task support ────

    def setup_process_task(self):
        super().setup_process_task()

        # Agent will be setting up its own progress on startup
        self.prepare_task(f"Process {self.agent_id} [{self.state.name}]")

    # ──── Activation method

    def _spawn_process(self) -> None:
        # Create and start agent process
        self.process = Process(
            target=self._agent_fn,
            args=(
                self.agent_id,
                self._request_queue,
                self.response_queue,
                self.executor_factory_method,
                self.task,
            ),
        )
        self.process.start()

    @staticmethod
    def _agent_fn(
        agent_id: int,
        agent_queue: Queue,
        response_queue: Queue,
        executor_factory_method: Callable[[], PipelineExecutorType],
        task: int,
    ):
        executor = executor_factory_method()
        agent = ExecutorAgentType(
            agent_id=agent_id,
            pipeline_executor=executor,
            request_queue=agent_queue,
            response_queue=response_queue,
        )
        # Progress relay will forward progress back to the main process via worker's response queue.
        agent.progress = ProgressRelay(response_queue)
        # Use Agent link task for displaying executor progress in line with agent progress
        agent.enable_task_inline_structure(task)
        agent.setup_process_task()

        # Start worker loop
        agent.run_loop()

    # ──── State management methods ────

    def inactive(self) -> None:
        self.state = ExecutorAgentState.inactive

    def waiting(self) -> None:
        self.state = ExecutorAgentState.waiting

    def running(self) -> None:
        self.state = ExecutorAgentState.running

    def failed(self) -> None:
        self.state = ExecutorAgentState.failed

    # Public properties

    @property
    def is_inactive(self) -> bool:
        return self.state == ExecutorAgentState.inactive

    @property
    def is_waiting(self) -> bool:
        return self.state == ExecutorAgentState.waiting

    @property
    def is_running(self) -> bool:
        return self.state == ExecutorAgentState.running

    @property
    def is_failed(self) -> bool:
        return self.state == ExecutorAgentState.failed

    @property
    def input_item_index(self) -> int:
        return self._input_item_index
