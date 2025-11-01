from dataclasses import dataclass
from typing import cast, final

import torch

from apriori.flow.core.executor.types import PipelineExecutorType
from apriori.flow.core.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineControlMessage,
    PipelineInputType,
    PipelineOutputType,
)
from apriori.flow.core.runner.types import (
    OnResultCallbackType,
    RunnerInputType,
    RunnerProtocol,
    RunnerResultType,
)
from apriori.flow.core.runner.utils import get_input_len_or_zero
from apriori.flow.lifecycle import LifecycleMixin, SupportsLifecycle
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.progress.types import SupportsProgressTask
from apriori.flow.structure import HasFlowStructure


@dataclass(slots=True)
class RunnerConfig:
    title: str = ""
    # Whether to show progress for each step in the pipeline
    show_pipeline_progress: bool = True

    def __str__(self) -> str:
        return f"Runner({self.title})"


@final
class Runner(
    RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    LifecycleMixin,  # Add NoOp lifecycle methods
    ProgressMixin,  # Add progress support
    HasFlowStructure,  #  The class implements flow structure support
):
    __slots__ = (
        "config",
        "input",
        "pipeline_executor",
        "on_result",
        "_prepared",
    )

    config: RunnerConfig
    input: RunnerInputType
    pipeline_executor: PipelineExecutorType
    on_result: OnResultCallbackType | None
    _prepared: bool

    def __init__(
        self,
        config: RunnerConfig,
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

    # ──── Main execution ────

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        for i, item in enumerate(self.input):
            # Send cycle start event to executor
            if isinstance(self.pipeline_executor, SupportsLifecycle):
                self.pipeline_executor.on_cycle_start()

            # Run the pipeline
            result = self.pipeline_executor.run(item)

            # Callback with the result if provided
            self._callback_result(
                RunnerResultType(
                    input_item_index=i,
                    pipeline_result=result,
                )
            )
            self.progress.advance(self._task)

            if isinstance(self.pipeline_executor, SupportsLifecycle):
                self.pipeline_executor.on_cycle_end()

            # Stop execution if the pipeline requests to stop cycle
            if result.control is PipelineControlMessage.StopCycle:
                # Complete the progress for the runner
                input_len = get_input_len_or_zero(self.input)
                if input_len > 0:
                    self.progress.update(self._task, completed=input_len)
                break

        self.on_cycle_end()
        self.progress.print(f"✅ {str(self.pipeline_executor)} completed!")

    def _callback_result(self, result: RunnerResultType) -> None:
        if self.on_result is not None:
            self.on_result(result)

    # ──── Lifecycle Support ────

    def prepare(self) -> None:
        if self._prepared:
            return
        # Prepare the pipeline executor
        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.prepare()

    def on_cycle_start(self) -> None:
        # Reset progress for the runner task
        self.progress.update(self.task, completed=0)

    def cleanup(self) -> None:
        if not self._prepared:
            return

        if isinstance(self.pipeline_executor, SupportsLifecycle):
            self.pipeline_executor.cleanup()

        # Remove tasks from progress tracker
        self.progress.remove_task(self._task)
        self._task = None

        # Cleanup the pipeline executor
        self.pipeline_executor.cleanup()

        self._prepared = False

    # ──── Progress task support ────

    def setup_process_task(self):
        super().setup_process_task()

        # Prepare runner task for progress tracking
        self.prepare_task(
            description=str(self.input), total=get_input_len_or_zero(self.input)
        )

        # Setup progress for pipeline executor
        if isinstance(self.pipeline_executor, SupportsProgressTask):
            pipeline_executor = cast(ProgressMixin, self.pipeline_executor)
            pipeline_executor.progress = self.progress
            pipeline_executor.enable_task_tree_structure(self.task)
            pipeline_executor.setup_process_task(self.task, is_last_subtask=True)

    # ──── Flow structure supports ────

    def describe_structure(self) -> HasFlowStructure.FlowStructure:
        if isinstance(self.pipeline_executor, HasFlowStructure):
            children = [self.pipeline_executor.describe_structure()]
        else:
            self.progress.print(
                f"⚠️ Pipeline executor {repr(self.pipeline_executor)} does not implement HasFlowStructure."
            )
            children = []

        return HasFlowStructure.FlowStructure(
            name="Runner",
            type="runner",
            children=children,
        )

    # ──── Properties ────

    def __str__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.config.title}')"

    # ──── State management methods ────

    def get_state(self) -> dict[str, torch.Tensor]:
        """Get the current state of the executor as a dictionary of tensors."""
        raise NotImplementedError("get_state method is not implemented yet.")

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Load the executor state from a dictionary of tensors."""
        raise NotImplementedError("load_state_dict method is not implemented yet.")
