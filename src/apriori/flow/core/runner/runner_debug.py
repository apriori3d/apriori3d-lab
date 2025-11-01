from typing import TypeAlias

from apriori.flow.core.executor.executor import PipelineExecutor
from apriori.flow.core.pipeline.pipeline import Pipeline
from apriori.flow.core.runner.parallel.parallel_runner import (
    ParallelRunner,
    ParallelRunnerConfig,
)
from apriori.flow.progress.rich.utils import create_progress

DebugPipelineConfig: TypeAlias = str
DebugPipelineInput: TypeAlias = int
DebugPipelineOutput: TypeAlias = int
DebugPipelineContext: TypeAlias = list[int]

DebugPipeline: TypeAlias = Pipeline[
    DebugPipelineConfig,
    DebugPipelineInput,
    DebugPipelineOutput,
]
DebugExecutor: TypeAlias = PipelineExecutor[
    DebugPipelineConfig,
    DebugPipelineContext,
    DebugPipelineInput,
    DebugPipelineOutput,
]

if __name__ == "__main__":

    def executor_factory_method() -> DebugExecutor:
        def input_step(
            context: DebugPipelineContext, input_item: DebugPipelineInput
        ) -> None:
            context[0] = input_item

        def output_step(context: DebugPipelineContext) -> DebugPipelineOutput:
            return context[0]

        def step1(context: DebugPipelineContext) -> None:
            context[0] += 1

        def step2(context: DebugPipelineContext) -> None:
            context[0] *= 2

        pipeline = DebugPipeline(
            config="Calculator",
            input_step=input_step,
            output_step=output_step,
            steps=[
                step1,
                step2,
            ],
        )
        return DebugExecutor(pipeline=pipeline, context=[0])

    with create_progress() as progress:
        executor = executor_factory_method()
        # runner = Runner(
        #     config=RunnerConfig(title="DebugRunner"),
        #     input=range(10),
        #     executor=executor,
        #     on_pipeline_result=lambda output: progress.print(
        #         f"Output: {output.pipeline_result.output}"
        #     ),
        # )
        runner = ParallelRunner(
            config=ParallelRunnerConfig(title="DebugParallelRunner"),
            input=range(10),
            executor_factory_method=executor_factory_method,
            num_workers=4,
        )
        runner.progress = progress
        runner.prepare()
        runner.run()
