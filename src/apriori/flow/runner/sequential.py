from collections.abc import Sized
from dataclasses import dataclass
from typing import Generic, final

from apriori.flow.pipeline.types import PipelineControlMessage, PipelineExecutorProtocol
from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import (
    ProgressMixin,
    ProgressWithLevels,
    WithProgress,
)
from apriori.flow.runner.types import (
    OnPipelineResultType,
    PipelineConfigType,
    PipelineContextType,
    PipelineExecutorType,
    PipelineInputType,
    PipelineOutputType,
    RunnereResult,
    RunnerInputType,
    StepWithRunner,
)


@dataclass(slots=True)
class SequentialRunnerConfig:
    title: str | None = None
    # Whether to show progress for each step in the pipeline
    show_pipeline_progress: bool = True


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
        "executor",
    )

    config: SequentialRunnerConfig
    executor: PipelineExecutorProtocol[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ]

    # Internal state
    _on_pipeline_result: OnPipelineResultType | None
    _prepared: bool
    _task: int | None
    _pipeline_task: int | None

    def __init__(
        self,
        config: SequentialRunnerConfig,
        executor: PipelineExecutorType,
        on_pipeline_result: OnPipelineResultType | None = None,
    ) -> None:
        self.config = config
        self.executor = executor
        self._prepared = False
        self._task = None
        self._pipeline_task = None
        self._on_pipeline_result = on_pipeline_result

    # lifecycle hooks

    def prepare(self) -> None:
        if self._prepared:
            return
        self._prepared = True
        self._prepare_progress()
        self.executor.prepare()

        # Prepare nested runners
        for step in self.executor.pipeline.steps:
            if isinstance(step, StepWithRunner):
                if isinstance(step.runner, WithProgress):
                    step.runner.progress = self.progress
                step.runner.prepare()

    def reset(self) -> None:
        self.progress.update(self._task, completed=0)
        self.executor.reset()

        # Reset nested runners
        for step in self.executor.pipeline.steps:
            if isinstance(step, StepWithRunner):
                step.runner.reset()

    # Main execution
    def run(self, input_item: RunnerInputType) -> None:
        self.prepare()
        self.reset()

        total = len(input_item) if isinstance(input_item, Sized) else None
        self.progress.update(
            self._task,
            description=f"▶️ {self.config.title or input_item}",
            total=total,
            completed=0,
        )

        for i, item in enumerate(input_item):
            # Reset pipeline state before each run
            for step in self.executor.pipeline.steps:
                if isinstance(step, StepWithRunner):
                    step.runner.reset()

            # Add hierarchy level if supported
            self._add_level()

            # Run the pipeline
            result = self.executor.run(item)

            # Callback with the result if provided
            if self._on_pipeline_result is not None:
                self._on_pipeline_result(
                    RunnereResult(
                        item_index=i,
                        context=self.executor.context,
                        pipeline=self.executor.pipeline,
                        pipeline_result=result,
                    ),
                )

            # Remove hierarchy level if supported
            self._remove_level()
            self.progress.advance(self._task)

            # Stop execution if the pipeline requests to finalize
            if result.control is PipelineControlMessage.Finalize:
                if total is not None:
                    self.progress.update(self._task, completed=total)
                break

        self.progress.print(f"✅ {self.executor.pipeline.config} completed!")

    def _prepare_progress(self):
        if self.progress is None:
            self.progress = NoOpProgress()  # Default no-op progress

        self._task = self.progress.add_task("", total=0, completed=0)
        self._pipeline_task = self.progress.add_task(
            description="",
            total=0,
            completed=0,
            visible=self.config.show_pipeline_progress,
        )

        # Propagate progress to the pipeline if it supports it
        if isinstance(self.executor, WithProgress):
            self.executor.progress = self.progress
            self.executor.shared_task = self._pipeline_task

    def _add_level(self, prefix: str | None = None) -> None:
        if isinstance(self.progress, ProgressWithLevels):
            self.progress.add_level(prefix=prefix)

    def _remove_level(self) -> None:
        if isinstance(self.progress, ProgressWithLevels):
            self.progress.remove_level()
