from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

from apriori.flow.core.executor.types import PipelineExecutorType
from apriori.flow.core.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
)

RunnerInputType = TypeVar("RunnerInputType", bound=Iterable[PipelineInputType])


# ──── Result type for runner execution ────
@dataclass(slots=True)
class RunnerResult(
    Generic[PipelineOutputType],
):
    input_item_index: int
    pipeline_result: PipelineResult[PipelineOutputType]


RunnerResultType: TypeAlias = RunnerResult[PipelineOutputType]

# ──── Callback type for handling runner results ────
OnResultCallbackType: TypeAlias = Callable[[RunnerResultType], None]


# ──── Main Runner Protocol ────
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


# ──── Nested runner protocol ────
@runtime_checkable
class HasRunner(Protocol):
    runner: RunnerType
