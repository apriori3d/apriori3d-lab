from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from apriori.flow.executor.types import PipelineExecutorProtocol
from apriori.flow.pipeline.types import PipelineProtocol, PipelineResult
from apriori.flow.structure import HasFlowStructure

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")
RunnerInputType = TypeVar("RunnerInputType", bound=Iterable[PipelineInputType])


# Redefine a pipeline and executor type with local generic parameters

PipelineType: TypeAlias = PipelineProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
]

# Define a type alias for the pipeline used in the runner for better readability
PipelineExecutorType: TypeAlias = PipelineExecutorProtocol[
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


# Result type for runner execution
@dataclass(slots=True)
class RunnerResult(
    Generic[PipelineOutputType],
):
    input_index: int
    pipeline_result: PipelineResult[PipelineOutputType]


RunnerResultType: TypeAlias = RunnerResult[PipelineOutputType]

# Callback type for handling runner results

OnResultCallbackType: TypeAlias = Callable[[RunnerResultType], None]

# Define the main Runner Protocol


class RunnerProtocol(
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    HasFlowStructure,
):
    input: RunnerInputType
    pipeline_executor: PipelineExecutorType

    # Callback to be called on each result
    on_result: OnResultCallbackType | None

    # Main run method
    def run(self) -> None: ...


RunnerType: TypeAlias = RunnerProtocol[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


# Protocol for elements with nested runner
@runtime_checkable
class HasRunner(
    Protocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ]
):
    runner: RunnerType
