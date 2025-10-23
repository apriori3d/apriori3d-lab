from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, TypeAlias, TypeVar

from typing_extensions import runtime_checkable

# Generic type variables for Pipeline

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")


# Control messages for pipeline execution flow
class PipelineControlMessage(Enum):
    Continue = auto()
    Finalize = auto()


# Define protocols for steps
class InputStepProtocol(
    Protocol[PipelineConfigType, PipelineContextType, PipelineInputType]
):
    def __call__(
        self, context: PipelineContextType, input_data: PipelineInputType
    ) -> None: ...


class StepProtocol(Protocol[PipelineConfigType, PipelineContextType]):
    # Step can control execution flow by returning a control message
    def __call__(
        self, context: PipelineContextType
    ) -> PipelineControlMessage | None: ...


class OutputStepProtocol(
    Protocol[PipelineConfigType, PipelineContextType, PipelineOutputType]
):
    def __call__(self, context: PipelineContextType) -> PipelineOutputType: ...


# Type aliases for better readability (can be used in parameter annotations)
InputStep: TypeAlias = InputStepProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType
]
Step: TypeAlias = StepProtocol[PipelineConfigType, PipelineContextType]
OutputStep: TypeAlias = OutputStepProtocol[
    PipelineConfigType, PipelineContextType, PipelineOutputType
]


# Step Life cycle protocol
@runtime_checkable
class StepWithPrepare(Protocol[PipelineConfigType, PipelineContextType]):
    def prepare(self, context: PipelineContextType) -> None: ...


# Pipeline definition
@dataclass(slots=True)
class Pipeline(Generic[PipelineConfigType, PipelineInputType, PipelineOutputType]):
    config: PipelineConfigType
    input_step: InputStep
    steps: list[Step]
    output_step: OutputStep

    def __str__(self) -> str:
        return f"Pipeline(steps={len(self.steps)})"


# Executor protocol
class PipelineExecutorProtocol(
    Protocol,
    Generic[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
):
    pipeline: Pipeline[
        PipelineConfigType,
        PipelineInputType,
        PipelineOutputType,
    ]
    context: PipelineContextType

    # Lifecycle methods
    def prepare(self) -> None: ...
    def reset(self) -> None: ...

    # Main execution method
    def run(
        self, input_data: PipelineInputType
    ) -> "PipelineResult[PipelineOutputType]": ...


@dataclass(slots=True)
class PipelineResult(Generic[PipelineOutputType]):
    output: PipelineOutputType
    control: PipelineControlMessage
