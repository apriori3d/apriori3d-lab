from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, TypeAlias, TypeVar

# Generic type variables for Pipeline

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")


# Control messages for pipeline execution flow
class PipelineControlMessage(Enum):
    Continue = auto()
    StopCycle = auto()


# Steps Protocols and Type Aliases


class InputStepProtocol(
    Protocol[PipelineConfigType, PipelineContextType, PipelineInputType],
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
InputStepType: TypeAlias = InputStepProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType
]

StepType: TypeAlias = StepProtocol[PipelineConfigType, PipelineContextType]

OutputStepType: TypeAlias = OutputStepProtocol[
    PipelineConfigType, PipelineContextType, PipelineOutputType
]


# Pipeline Protocol


class PipelineProtocol(
    Protocol,
    Generic[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
):
    config: PipelineConfigType
    input_step: InputStepType
    steps: list[StepType]
    output_step: OutputStepType
    num_steps: int


PipelineType: TypeAlias = PipelineProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
]


@dataclass(slots=True)
class PipelineResult(Generic[PipelineOutputType]):
    output: PipelineOutputType
    control: PipelineControlMessage


# @runtime_checkable
# class HasState(Protocol):
#     def get_state(self) -> dict[str, torch.Tensor]:
#         """Get the current state of the executor as a dictionary of tensors."""
#         ...

#     def load_state(self, state: dict[str, torch.Tensor]) -> None:
#         """Load the executor state from a dictionary of tensors."""
#         ...
