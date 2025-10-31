from typing import final

from apriori.flow.executor.types import PipelineExecutorProtocol
from apriori.flow.lifecycle import HasLifecycle
from apriori.flow.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineControlMessage,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
    PipelineType,
    StepType,
)
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.progress.types import (
    HasProgress,
)
from apriori.flow.structure import FlowStructure, HasFlowStructure


@final
class PipelineExecutor(
    PipelineExecutorProtocol[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
    ProgressMixin,  # Add progress support
    HasLifecycle,  # Implements lifecycle support
    HasFlowStructure,  # Implements flow structure support
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

    # Properties

    def __str__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    def __repr__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    # Flow structure

    def describe_structure(self) -> FlowStructure:
        return FlowStructure(
            name="PipelineExecutor",
            type="executor",
            children=[self.pipeline.describe_structure()],
        )

    # Lifecycle methods

    def prepare(self) -> None:
        """Create task to display progress for execution.
        Attach progress tracking to all steps in the pipeline.
        """
        if self._prepared:
            return

        # Set default structure
        if self.task_structure == "undefined":
            self.enable_task_tree_structure()

        self.prepare_task(
            description=str(self.pipeline),
            total=self.pipeline.num_steps,
        )

        # Share progress and task for all steps
        self._prepare_steps(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )
        self._prepared = True

    def _prepare_steps(self, *steps: StepType) -> None:
        for step in steps:
            if isinstance(step, HasProgress):
                step.progress = self.progress

            if isinstance(step, ProgressMixin):
                self.propagate_task_structure(
                    instance=step,
                    is_last_subtask=(step is steps[-1]),
                )

            if isinstance(step, HasLifecycle):
                step.prepare()

    def on_cycle_start(self) -> None:
        # Note: this event may be triggered before the actual execution,
        # e.g., by the runner that owns this executor - in order to reset progress before each iteration.
        self.progress.update(self._task, completed=0)

        # Call cycle start on steps that support life cycle
        self._steps_on_cycle_start(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

    def _steps_on_cycle_start(self, *steps: StepType) -> None:
        for step in steps:
            if isinstance(step, HasLifecycle):
                step.on_cycle_start()

    def on_cycle_end(self) -> None:
        # Call cycle end on steps that support life cycle
        self._steps_on_cycle_end(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

    def _steps_on_cycle_end(self, *steps: StepType) -> None:
        for step in steps:
            if isinstance(step, HasLifecycle):
                step.on_cycle_end()

    def cleanup(self):
        if not self._prepared:
            return

        self.progress.remove_task(self._task)
        self._task = None

        self._steps_cleanup(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )
        self._prepared = False

    def _steps_cleanup(self, *steps: StepType) -> None:
        for step in steps:
            if isinstance(step, HasLifecycle):
                step.cleanup()

    # Main execution

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
