from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import TypeAlias, final

import torch

from apriori.flow.progress.console import ConsoleProgress
from apriori.flow.progress.types import (
    ProgressMixin,
)
from apriori.flow.runner.parallel.runner_agent import (
    RunnerAgent,
)
from apriori.flow.runner.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineExecutorType,
    PipelineInputType,
    PipelineOutputType,
    RunnerInputType,
    RunnerProtocol,
)


@dataclass(slots=True)
class ParallelRunnerConfig:
    title: str = ""

    def __str__(self) -> str:
        return f"ParallelRunner({self.title})"


@final
class ParallelRunner2(
    RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    ProgressMixin,  # Add progress support
    # HasState,  # Add state management support
):
    __slots__ = (
        "config",
        "input",
        "executor_factory_method",
        "_prepared",
        "_task",
        "_workers",
        "_response_queue",
    )

    config: ParallelRunnerConfig
    input: RunnerInputType
    executor_factory_method: Callable[[], PipelineExecutorType]
    num_workers: int

    # Internal state
    _workers_pool: "WorkersPoolType"
    _task: int | None
    _prepared: bool

    def __init__(
        self,
        config: ParallelRunnerConfig,
        input: RunnerInputType,
        executor_factory_method: Callable[[], PipelineExecutorType],
        num_workers: int,
    ) -> None:
        self.config = config
        self.input = input
        self.executor_factory_method = executor_factory_method

        # Create workers pool to manage workers
        self._workers_pool = WorkersPoolType(
            num_workers, executor_factory_method=executor_factory_method
        )

        self.progress = ConsoleProgress()  # Default progress
        self._task = None
        self._completed = 0
        self._prepared = False

    @property
    def input_len(self) -> int | None:
        if isinstance(self.input, Sized):
            return len(self.input)
        return None

    # Main execution

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        # Process input items using workers pool
        self._workers_pool.run(self.input)

        # Finalize execution
        self.on_cycle_end()
        self.progress.print("✅ ParallelRunner execution completed!")

    # RunnerLifecycleProtocol methods

    def prepare(self) -> None:
        if self._prepared:
            return

        # Create task for the runner to track progress
        self._task = self.progress.add_task(
            str(self.input), total=self.input_len, completed=0
        )
        # Propagate progress to workers pool
        self._workers_pool.progress = self.progress
        # Assign parent task to workers pool to nest their tasks under runner task
        self._workers_pool.parent_task = self._task

        # Spawn workers
        self._workers_pool.prepare()
        self._prepared = True

    def on_cycle_start(self) -> None:
        # Reset progress for the runner task
        self.progress.update(self._task, completed=0)
        self._completed = 0
        self._workers_pool.on_cycle_start()

    def on_cycle_end(self) -> None:
        self._workers_pool.on_cycle_end()

    def cleanup(self) -> None:
        if not self._prepared:
            return

        self._workers_pool.cleanup()

        # Remove tasks from progress tracker
        self.progress.remove_task(self._task)
        self._task = None
        self._prepared = False
        self._completed = 0

    # State management

    def get_state(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("ParallelRunner does not implement get_state yet.")

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        raise NotImplementedError("ParallelRunner does not implement load_state yet.")

    def print_plan(self) -> None:
        raise NotImplementedError("ParallelRunner does not implement print_plan yet.")
        self._prepare(self)
        self._print_runner_plan(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"


WorkerType: TypeAlias = RunnerAgent[
    PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
]
