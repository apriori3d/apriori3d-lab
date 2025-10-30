from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from apriori.flow.pipeline.types import (
    PipelineExecutorProtocol,
    PipelineResult,
)
from apriori.flow.runner.lifecycle_mixin import LifecycleProtocol

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")
RunnerInputType = TypeVar("RunnerInputType", bound=Iterable[PipelineInputType])


@dataclass(slots=True)
class RunnerResult(
    Generic[PipelineOutputType,],
):
    input_index: int
    pipeline_result: PipelineResult[PipelineOutputType]


RunnerResultType: TypeAlias = RunnerResult[PipelineOutputType]

OnResultCallbackType: TypeAlias = Callable[[RunnerResultType], None]

# Define a type alias for the pipeline used in the runner for better readability
PipelineExecutorType: TypeAlias = PipelineExecutorProtocol[
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


class RunnerProtocol(
    LifecycleProtocol,
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
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


# Define a protocol for elements with nested runner to support pipeline hierarchies
@runtime_checkable
class HasRunner(Protocol):
    runner: RunnerType
