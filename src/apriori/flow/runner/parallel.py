from collections.abc import Callable, Sized
from dataclasses import dataclass
from typing import Generic

from apriori.flow.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    PipelineType,
    WithProgress,
)
from apriori.flow.progress.types import (
    ProgressProtocol,
    ProgressWithLevels,
    ProgressWithPrefix,
)
from apriori.flow.runner.types import RunnerInputType


@dataclass(slots=True)
class RunnerPipelineOutput:
    input: PipelineInputType
    output: PipelineOutputType


RunnerOutputCallbackType = Callable[[RunnerPipelineOutput], None]

# TODO: refactor parallel impementation to use PipelineType


class ParallelRunner(
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ]
):
    pipeline_factory: Callable[[], PipelineType]
    context_factory: Callable[[], PipelineContextType]
    progress: ProgressProtocol
    output_callback: RunnerOutputCallbackType

    def run(self, input_data: RunnerInputType) -> None:
        with self.progress as progress:
            total = len(input_data) if isinstance(input_data, Sized) else None
            task = progress.add_task(f"▶️ {input_data}", total=total)

            pipeline_with_progress = isinstance(self.pipeline, WithProgress)
            if pipeline_with_progress:
                self.pipeline.progress = progress

            has_prefix = isinstance(progress, ProgressWithPrefix)
            if isinstance(progress, ProgressWithLevels):
                progress.add_level()

            for i, item in enumerate(input_data):
                if has_prefix:
                    progress.prefix = f"Frame [{i}] "

                if pipeline_with_progress:
                    item_task = progress.add_task(f"Frame [{i}] ", total=1)
                    self.pipeline.task = item_task

                result = self.pipeline.run(item)

                self.output_callback(RunnerPipelineOutput(index=i, output=result))

                if pipeline_with_progress:
                    progress.remove_task(item_task)

                progress.advance(task)

            if has_prefix:
                progress.prefix = None
            if isinstance(progress, ProgressWithLevels):
                progress.remove_level()

            progress.print(f"✅ {self.pipeline.description}")
