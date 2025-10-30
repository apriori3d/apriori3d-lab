from collections.abc import Sized
from dataclasses import dataclass
from typing import Generic, final

import torch

from apriori.flow.pipeline.types import PipelineControlMessage
from apriori.flow.progress.console import ConsoleProgress
from apriori.flow.progress.types import (
    HasProgress,
    ProgressMixin,
)
from apriori.flow.runner.types import (
    HasRunner,
    OnResultCallbackType,
    PipelineConfigType,
    PipelineContextType,
    PipelineExecutorType,
    PipelineInputType,
    PipelineOutputType,
    RunnerInputType,
    RunnerResultType,
)


@dataclass(slots=True)
class SequentialRunnerConfig:
    title: str = ""
    # Whether to show progress for each step in the pipeline
    show_pipeline_progress: bool = True

    def __str__(self) -> str:
        return f"SequentialRunner({self.title})"


@final
class SequentialRunner(
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    ProgressMixin,  # Add progress support
):
    __slots__ = (
        "_on_pipeline_result",
        "_pipeline_task",
        "_prepared",
        "_task",
        "config",
        "input",
        "executor",
    )

    config: SequentialRunnerConfig
    input: RunnerInputType
    executor: PipelineExecutorType

    # Internal state
    _on_pipeline_result: OnResultCallbackType | None
    _prepared: bool
    _task: int | None
    _executor_task: int | None

    def __init__(
        self,
        config: SequentialRunnerConfig,
        input: RunnerInputType,
        executor: PipelineExecutorType,
        *,
        on_pipeline_result: OnResultCallbackType | None = None,
    ) -> None:
        self.config = config
        self.input = input
        self.executor = executor
        self._prepared = False
        self._task = None
        self._executor_task = None
        self._on_pipeline_result = on_pipeline_result
        self.progress = ConsoleProgress()  # Default progress

    @property
    def input_len(self) -> int | None:
        if isinstance(self.input, Sized):
            return len(self.input)
        return None

    # Main execution

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        for i, item in enumerate(self.input):
            # Send cycle start to nested runners
            for step in self.executor.pipeline.steps:
                if isinstance(step, HasRunner):
                    self._on_cycle_start(step.runner)

            # Run the pipeline
            result = self.executor.run(item)

            # Callback with the result if provided
            if self._on_pipeline_result is not None:
                self._on_pipeline_result(
                    RunnerResultType(
                        input_index=i,
                        pipeline_result=result,
                    ),
                )

            self.progress.advance(self._task)

            # Stop execution if the pipeline requests to stop cycle
            if result.control is PipelineControlMessage.StopCycle:
                if self.input_len is not None:
                    self.progress.update(self._task, completed=self.input_len)
                break

            # Send cycle end to nested runners
            for step in self.executor.pipeline.steps:
                if isinstance(step, HasRunner):
                    self._on_cycle_end(step.runner)

        self.on_cycle_end()
        self.progress.print(f"✅ {self.executor.pipeline} completed!")

    # lifecycle hooks

    def prepare(self) -> None:
        SequentialRunner._prepare(self)

    def on_cycle_start(self) -> None:
        SequentialRunner._on_cycle_start(self, nested=False)

    def on_cycle_end(self) -> None:
        SequentialRunner._on_cycle_end(self)

    def cleanup(self) -> None:
        SequentialRunner._cleanup(self)

    @staticmethod
    def _prepare(
        runner: "SequentialRunner",
        parent_task: int | None = None,
        is_last_subtask: bool = True,
    ) -> None:
        if runner._prepared:
            return

        if runner._shared_task is None:
            # Add task for runner to progress tracker
            runner._task = runner.progress.add_task(
                str(runner.input),
                total=runner.input_len,
                completed=0,
                parent_task=parent_task,
                is_last_subtask=is_last_subtask,
            )
            # Add task for pipeline executor to progress tracker
            runner._executor_task = runner.progress.add_task(
                str(runner.executor.pipeline),
                total=runner.executor.pipeline.num_steps,
                completed=0,
                parent_task=runner._task,  # Enable hierarchy
                is_last_subtask=True,  # Pipeline is the only node under runner
            )
        else:
            # Use single shared task
            runner._task = runner._shared_task
            runner._executor_task = runner._shared_task

        # Set progress to the pipeline executor
        if isinstance(runner.executor, HasProgress):
            runner.executor.progress = runner.progress
            # To track progress for a pipeline as a single task,
            # shared task being used for all steps in the pipeline.
            runner.executor.shared_task = runner._executor_task

        # Prepare the pipeline executor
        runner.executor.prepare()

        # Prepare nested runners in pipeline steps
        nested_runners = [
            step.runner
            for step in runner.executor.pipeline.steps
            if isinstance(step, HasRunner)
        ]
        for i, nested_runner in enumerate(nested_runners):
            # Set progress to nested runner
            nested_runner.progress = runner.progress
            # Set flag to correctly format tree
            is_last_subtask = ((i == len(nested_runners) - 1),)
            # Prepare nested runner
            SequentialRunner._prepare(
                nested_runner,
                parent_task=runner._executor_task,  # Set executor task as a parent
                is_last_subtask=is_last_subtask,
            )

        runner._prepared = True

    @staticmethod
    def _on_cycle_start(runner: "SequentialRunner", nested: bool = False) -> None:
        # Reset progress for the runner task
        runner.progress.update(runner._task, completed=0)

        # Send event to the pipeline executor
        runner.executor.on_cycle_start()

        # Send event to nested runners in pipeline steps
        if nested:
            for step in runner.executor.pipeline.steps:
                if isinstance(step, HasRunner):
                    SequentialRunner._on_cycle_start(step.runner, nested=True)

    @staticmethod
    def _on_cycle_end(runner: "SequentialRunner", nested: bool = False) -> None:
        # Send event to the pipeline executor
        runner.executor.on_cycle_end()

        # Send event to nested runners in pipeline steps
        if nested:
            for step in runner.executor.pipeline.steps:
                if isinstance(step, HasRunner):
                    SequentialRunner._on_cycle_end(step.runner, nested=True)

    @staticmethod
    def _cleanup(runner: "SequentialRunner") -> None:
        if not runner._prepared:
            return
        runner._prepared = False

        # Remove tasks from progress tracker
        runner.progress.remove_task(runner._task)
        runner.progress.remove_task(runner._executor_task)
        runner._task = None
        runner._executor_task = None

        # Cleanup the pipeline executor
        runner.executor.cleanup()

        # Cleanup nested runners in pipeline steps
        for step in runner.executor.pipeline.steps:
            if isinstance(step, HasRunner):
                SequentialRunner._cleanup(step.runner)

    # Unitility methods

    def print_plan(self) -> None:
        self._prepare(self)
        self._print_runner_plan(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"

    @staticmethod
    def _print_runner_plan(runner: "SequentialRunner", prefix="") -> None:
        # Print runner with input and pipeline info.

        if prefix == "":
            # Top-level runner
            runner_prefix = prefix
        else:
            # Nested runner
            runner_prefix = f"{prefix}└──"
            prefix = f"{prefix}   "
        runner.progress.print(f"{runner_prefix}{repr(runner)}")

        # Print runner input and pipeline as nodes.
        runner.progress.print(f"{prefix}├──{repr(runner.input)}")
        runner.progress.print(f"{prefix}└──{repr(runner.executor.pipeline)}")

        # Add indentation for pipeline steps.
        prefix = f"{prefix}   "
        # Print pipeline steps.
        runner.progress.print(f"{prefix}├──{repr(runner.executor.pipeline.input_step)}")

        for step in runner.executor.pipeline.steps:
            if isinstance(step, HasRunner) and isinstance(
                step.runner, SequentialRunner
            ):
                runner.progress.print(f"{prefix}├──{repr(step)}")
                # Print runner plan recursively with increased indentation
                SequentialRunner._print_runner_plan(step.runner, prefix=f"{prefix}│  ")
            else:
                runner.progress.print(f"{prefix}├──{repr(step)}")

        runner.progress.print(
            f"{prefix}└──{repr(runner.executor.pipeline.output_step)}"
        )

    # State management methods

    def get_state(self) -> dict[str, torch.Tensor]:
        """Get the current state of the executor as a dictionary of tensors."""
        raise NotImplementedError("get_state method is not implemented yet.")

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load the executor state from a dictionary of tensors."""
        raise NotImplementedError("load_state_dict method is not implemented yet.")
