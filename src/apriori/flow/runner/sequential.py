from dataclasses import dataclass
from typing import final

import torch

from apriori.flow.lifecycle import HasLifecycle
from apriori.flow.pipeline.types import PipelineControlMessage
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.runner.types import (
    HasRunner,
    OnResultCallbackType,
    PipelineConfigType,
    PipelineContextType,
    PipelineExecutorType,
    PipelineInputType,
    PipelineOutputType,
    RunnerInputType,
    RunnerProtocol,
    RunnerResultType,
)
from apriori.flow.runner.utils import get_input_len_or_zero
from apriori.flow.structure import HasFlowStructure


@dataclass(slots=True)
class SequentialRunnerConfig:
    title: str = ""
    # Whether to show progress for each step in the pipeline
    show_pipeline_progress: bool = True

    def __str__(self) -> str:
        return f"SequentialRunner({self.title})"


@final
class SequentialRunner(
    RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    HasLifecycle,  # Implement lifecycle methods
    ProgressMixin,  # Add progress support
    HasFlowStructure,  #  Implement flow structure support
):
    __slots__ = (
        "config",
        "input",
        "pipeline_executor",
        "on_result",
        "_prepared",
    )

    config: SequentialRunnerConfig
    input: RunnerInputType
    pipeline_executor: PipelineExecutorType
    on_result: OnResultCallbackType | None
    _prepared: bool

    def __init__(
        self,
        config: SequentialRunnerConfig,
        input: RunnerInputType,
        executor: PipelineExecutorType,
        *,
        on_result: OnResultCallbackType | None = None,
    ) -> None:
        self.config = config
        self.input = input
        self.pipeline_executor = executor
        self.on_result = on_result
        self._prepared = False

    # Main execution

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        for i, item in enumerate(self.input):
            # Send cycle start event to executor
            if isinstance(self.pipeline_executor, HasLifecycle):
                self.pipeline_executor.on_cycle_start()

            # Run the pipeline
            result = self.pipeline_executor.run(item)

            # Callback with the result if provided
            if self.on_result is not None:
                self.on_result(
                    RunnerResultType(
                        input_index=i,
                        pipeline_result=result,
                    ),
                )
            self.progress.advance(self._task)

            if isinstance(self.pipeline_executor, HasLifecycle):
                self.pipeline_executor.on_cycle_end()

            # Stop execution if the pipeline requests to stop cycle
            if result.control is PipelineControlMessage.StopCycle:
                # Complete the progress for the runner
                if input_len := get_input_len_or_zero(self.input) > 0:
                    self.progress.update(self._task, completed=input_len)
                break

        self.on_cycle_end()
        self.progress.print(f"✅ {self.pipeline_executor.pipeline} completed!")

    # lifecycle methods

    def prepare(self) -> None:
        if self._prepared:
            return

        # Set default structure
        if self.task_structure == "undefined":
            self.enable_task_tree_structure()

        # Prepare runner task for progress tracking
        self.prepare_task(
            description=str(self.input), total=get_input_len_or_zero(self.input)
        )

        # Enable inline task structure for pipeline executor
        if isinstance(self.pipeline_executor, ProgressMixin):
            self.pipeline_executor.enable_task_inline_structure(self.task)

        # Prepare the pipeline executor
        self.pipeline_executor.prepare()

    def on_cycle_start(self) -> None:
        # Reset progress for the runner task
        self.progress.update(self.task, completed=0)

    def on_cycle_end(self) -> None:
        pass

    def cleanup(self) -> None:
        if not self._prepared:
            return
        self._prepared = False

        # Remove tasks from progress tracker
        self.progress.remove_task(self._task)
        self._task = None

        # Cleanup the pipeline executor
        self.pipeline_executor.cleanup()

    # Flow structure methods

    def describe_structure(self) -> HasFlowStructure.FlowStructure:
        return HasFlowStructure.FlowStructure(
            name="SequentialRunner",
            type="runner",
            children=[self.pipeline_executor.describe_structure()],
        )

    # Utility methods

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
        runner.progress.print(f"{prefix}└──{repr(runner.pipeline_executor.pipeline)}")

        # Add indentation for pipeline steps.
        prefix = f"{prefix}   "
        # Print pipeline steps.
        runner.progress.print(
            f"{prefix}├──{repr(runner.pipeline_executor.pipeline.input_step)}"
        )

        for step in runner.pipeline_executor.pipeline.steps:
            if isinstance(step, HasRunner) and isinstance(
                step.runner, SequentialRunner
            ):
                runner.progress.print(f"{prefix}├──{repr(step)}")
                # Print runner plan recursively with increased indentation
                SequentialRunner._print_runner_plan(step.runner, prefix=f"{prefix}│  ")
            else:
                runner.progress.print(f"{prefix}├──{repr(step)}")

        runner.progress.print(
            f"{prefix}└──{repr(runner.pipeline_executor.pipeline.output_step)}"
        )

    # State management methods

    def get_state(self) -> dict[str, torch.Tensor]:
        """Get the current state of the executor as a dictionary of tensors."""
        raise NotImplementedError("get_state method is not implemented yet.")

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load the executor state from a dictionary of tensors."""
        raise NotImplementedError("load_state_dict method is not implemented yet.")
