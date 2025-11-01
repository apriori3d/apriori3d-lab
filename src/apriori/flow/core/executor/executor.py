from typing import final

from apriori.flow.core.executor.types import PipelineExecutorProtocol
from apriori.flow.core.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineControlMessage,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
    PipelineType,
)
from apriori.flow.core.runner.types import HasRunner
from apriori.flow.lifecycle import SupportsLifecycle
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.progress.types import (
    HasProgress,
    SupportsProgressTask,
)
from apriori.flow.structure import FlowStructure, HasFlowStructure


@final
class PipelineExecutor(
    PipelineExecutorProtocol[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
    SupportsLifecycle,  # The class implements lifecycle support
    ProgressMixin,  # Add progress support
    HasFlowStructure,  # The class implements flow structure support
):
    __slots__ = ("pipeline", "context", "_prepared")

    pipeline: PipelineType
    context: PipelineContextType

    def __init__(
        self,
        pipeline: PipelineType,
        context: PipelineContextType,
    ) -> None:
        self.pipeline = pipeline
        self.context = context
        self._prepared = False

    # ──── Main execution ────

    def run(self, input_data: PipelineInputType) -> PipelineResult[PipelineOutputType]:
        # Progress should be reset before each run, because run() may be called multiple times,
        # e.g., by a runner that executes multiple inputs through the same pipeline.
        self.progress.update(self._task, completed=0, context=str(input_data))

        # Convert input data to context
        self.pipeline.input_step(self.context, input_data)
        self.progress.advance(self._task, 1)
        control_message = PipelineControlMessage.Continue

        # Execute all steps
        for step in self.pipeline.steps:
            step_control_message = step(self.context)
            if (
                step_control_message is not None
                and step_control_message is PipelineControlMessage.StopCycle
            ):
                control_message = step_control_message
            self.progress.advance(self._task, 1)

        # Produce output from context
        output = self.pipeline.output_step(self.context)
        self.progress.update(self._task, completed=self.pipeline.num_steps)

        return PipelineResult(output, control_message)

    # ──── Lifecycle methods ────

    def prepare(self) -> None:
        if self._prepared:
            return

        for step in self.pipeline.all_steps:
            if isinstance(step, SupportsLifecycle):
                step.prepare()

            if isinstance(step, HasRunner) and isinstance(
                step.runner, SupportsLifecycle
            ):
                step.runner.prepare()

        self._prepared = True

    def on_cycle_start(self) -> None:
        # Note: this event may be triggered before the actual execution,
        # e.g., by the runner that owns this executor - in order to reset progress before each iteration.
        self.progress.update(self._task, completed=0)

        for step in self.pipeline.all_steps:
            # Call cycle start on steps that support life cycle
            if isinstance(step, SupportsLifecycle):
                step.on_cycle_start()

            # Call cycle start on nested runners that support life cycle
            if isinstance(step, HasRunner) and isinstance(
                step.runner, SupportsLifecycle
            ):
                step.runner.on_cycle_start()

    def on_cycle_end(self) -> None:
        # Call cycle end on steps that support life cycle
        for step in self.pipeline.all_steps:
            if isinstance(step, SupportsLifecycle):
                step.on_cycle_end()

            if isinstance(step, HasRunner) and isinstance(
                step.runner, SupportsLifecycle
            ):
                step.runner.on_cycle_end()

    def cleanup(self):
        if not self._prepared:
            return

        self.progress.remove_task(self._task)
        self._task = None

        for step in self.pipeline.all_steps:
            if isinstance(step, SupportsLifecycle):
                step.cleanup()

            if isinstance(step, HasRunner) and isinstance(
                step.runner, SupportsLifecycle
            ):
                step.runner.cleanup()

        self._prepared = False

    # ──── Properties ────

    def __str__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    def __repr__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    # ──── Progress task support ────

    def setup_process_task(self):
        super().setup_process_task()

        # Create or update task
        self.prepare_task(
            description=str(self.pipeline),
            total=self.pipeline.num_steps,
        )

        # Setup progress for all steps and nested runners
        steps = self.pipeline.all_steps()
        for step in steps:
            if isinstance(step, HasProgress):
                step.progress = self.progress

            # Setup nested runners' progress as tree structure
            if isinstance(step, HasRunner) and isinstance(
                step.runner, SupportsProgressTask
            ):
                step.runner.enable_task_tree_structure(self.task)
                step.runner.setup_process_task(
                    self.task, is_last_subtask=(step is steps[-1])
                )

    # ──── Flow structure support ────

    def describe_structure(self) -> FlowStructure:
        return FlowStructure(
            name="PipelineExecutor",
            type="executor",
            children=[
                FlowStructure(name=str(step), type="step")
                for step in self.pipeline.all_steps()
            ],
        )
