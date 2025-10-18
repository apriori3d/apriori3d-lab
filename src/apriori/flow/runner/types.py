from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from apriori.flow.pipeline.types import (
    Pipeline,
    PipelineExecutorProtocol,
    PipelineResult,
)

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")
RunnerInputType = TypeVar("RunnerInputType", bound=Iterable[PipelineInputType])


class RunnerProtocol(
    Protocol,
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
):
    pipeline: PipelineExecutorProtocol[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ]

    # Lifecycle hooks
    def prepare(self) -> None: ...
    def reset(self) -> None: ...

    # Main run method
    def run(self, stream: RunnerInputType) -> Any: ...


# Define a protocol for steps with nested runner to support pipeline hierarchies
@runtime_checkable
class StepWithRunner(Protocol):
    runner: RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ]


@dataclass(slots=True)
class RunnereResult(
    Generic[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
):
    item_index: int
    context: PipelineContextType
    pipeline: Pipeline[
        PipelineConfigType,
        PipelineInputType,
        PipelineOutputType,
    ]
    pipeline_result: PipelineResult[PipelineOutputType]


OnPipelineResultType: TypeAlias = Callable[[RunnereResult], None]

# Define a type alias for the pipeline used in the runner for better readability
PipelineExecutorType: TypeAlias = PipelineExecutorProtocol[
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]
