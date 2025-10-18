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
    StepWithPrepare,
)
from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import (
    ProgressMixin,
    ProgressWithLevels,
    WithProgress,
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
        self._task = None

    # Lifecycle hooks
    def prepare(self) -> None:
        # Prepare progress tracking and create task for pipeline
        self._task = self._prepare_progress()

        # share task if available for steps
        self._attach_progress_to_steps(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

        # Run prepare if available for steps
        self._prepare_steps(
            self.pipeline.input_step,
            self.pipeline.output_step,
            *self.pipeline.steps,
        )

    def reset(self) -> None:
        pass

    # Main execution
    def run(self, input_data: PipelineInputType) -> PipelineResult[PipelineOutputType]:
        self.progress.update(
            self._task,
            description=str(self.pipeline.config),
            total=len(self.pipeline.steps) + 2,
            completed=0,
        )
        if isinstance(self.progress, ProgressWithLevels):
            self.progress.add_level(prefix=f"{input_data} ")

        # Convert input data to context
        self.pipeline.input_step(self.context, input_data)
        self.progress.advance(self._task, 1)
        control_message = PipelineControlMessage.Continue

        # Execute all steps
        for step in self.pipeline.steps:
            step_control_message = step(self.context)
            if (
                step_control_message is not None
                and step_control_message is PipelineControlMessage.Finalize
            ):
                control_message = step_control_message
            self.progress.advance(self._task, 1)

        # Produce output from context
        output = self.pipeline.output_step(self.context)
        self.progress.update(self._task, completed=len(self.pipeline.steps) + 2)

        if isinstance(self.progress, ProgressWithLevels):
            self.progress.remove_level()

        return PipelineResult(output, control_message)

    def _prepare_progress(self) -> int:
        self.progress = self.progress or NoOpProgress()
        return (
            self.progress.add_task(description="", total=0, completed=0)
            if self.shared_task is None
            else self.shared_task
        )

    def _prepare_steps(self, *steps: Step) -> None:
        for step in steps:
            if isinstance(step, StepWithPrepare):
                step.prepare(self.context)

    def _attach_progress_to_steps(self, *steps: Step) -> None:
        for step in steps:
            if isinstance(step, WithProgress):
                step.progress = self.progress
                step.shared_task = self._task
