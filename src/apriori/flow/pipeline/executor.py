from typing import final

from apriori.flow.pipeline.types import (
    Pipeline,
    PipelineConfigType,
    PipelineContextType,
    PipelineControlMessage,
    PipelineExecutorProtocol,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
    Step,
    StepWithLifeCycle,
)
from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import (
    HasProgress,
    ProgressMixin,
)


@final
class PipelineExecutor(
    PipelineExecutorProtocol[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
    ProgressMixin,  # Add progress support
):
    __slots__ = ("_task", "pipeline")

    pipeline: Pipeline[PipelineConfigType, PipelineInputType, PipelineOutputType]
    context: PipelineContextType
    _task: int | None

    def __init__(
        self,
        pipeline: Pipeline[PipelineConfigType, PipelineInputType, PipelineOutputType],
        context: PipelineContextType,
    ) -> None:
        self.pipeline = pipeline
        self.context = context
        self.progress = NoOpProgress()
        self._task = None
        self._prepared = False

    def __str__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    def __repr__(self) -> str:
        return f"PipelineExecutor(pipeline={self.pipeline})"

    # Lifecycle hooks

    def prepare(self) -> None:
        """Create task to display progress for execution.
        Attach progress tracking to all steps in the pipeline.
        """
        if self._prepared:
            return

        # Create task for pipeline if not shared task is provided
        if self.has_shared_task:
            self._task = self.shared_task
            self.progress.update(
                self._task,
                description=str(self.pipeline),
                total=self.pipeline.num_steps,
                completed=0,
            )
        else:
            self._task = self.progress.add_task(
                description=str(self.pipeline),
                total=self.pipeline.num_steps,
                completed=0,
            )

        # Share progress and task for all steps
        self._attach_progress_to_steps(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )
        self._prepared = True

    def _attach_progress_to_steps(self, *steps: Step) -> None:
        for step in steps:
            if isinstance(step, HasProgress):
                step.progress = self.progress
                step.shared_task = self._task

    def on_cycle_start(self) -> None:
        # Note: this event may be triggered before the actual execution,
        # e.g., by the runner that owns this executor, in order to reset progress before each iteration.
        self.progress.update(self._task, completed=0)

        # Call cycle start on steps that support life cycle
        self._steps_on_cycle_start(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

    def _steps_on_cycle_start(self, *steps: Step) -> None:
        for step in steps:
            if isinstance(step, StepWithLifeCycle):
                step.on_cycle_start(self.context)

    def on_cycle_end(self) -> None:
        # Call cycle end on steps that support life cycle
        self._steps_on_cycle_end(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

    def _steps_on_cycle_end(self, *steps: Step) -> None:
        for step in steps:
            if isinstance(step, StepWithLifeCycle):
                step.on_cycle_end(self.context)

    def cleanup(self):
        if not self._prepared:
            return
        self._prepared = False

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
