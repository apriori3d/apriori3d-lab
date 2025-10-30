from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Protocol, TypeAlias, TypeVar

import torch
from typing_extensions import runtime_checkable

# Generic type variables for Pipeline

PipelineConfigType = TypeVar("PipelineConfigType")
PipelineContextType = TypeVar("PipelineContextType")
PipelineInputType = TypeVar("PipelineInputType")
PipelineOutputType = TypeVar("PipelineOutputType")


# Control messages for pipeline execution flow
class PipelineControlMessage(Enum):
    Continue = auto()
    StopCycle = auto()


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
class StepWithLifeCycle(Protocol[PipelineContextType]):
    def on_cycle_start(self, context: PipelineContextType) -> None:
        """Called at the start of each execution cycle to reset state if needed."""
        ...

    def on_cycle_end(self, context: PipelineContextType) -> None:
        """Called at the end of each execution cycle to finalize state if needed."""
        ...


# Pipeline definition
@dataclass(slots=True)
class Pipeline(Generic[PipelineConfigType, PipelineInputType, PipelineOutputType]):
    config: PipelineConfigType
    input_step: InputStep
    steps: list[Step]
    output_step: OutputStep

    @property
    def num_steps(self) -> int:
        return len(self.steps) + 2  # Including input and output steps

    def __str__(self) -> str:
        return f"Pipeline({str(self.config)})"

    def __repr__(self) -> str:
        return f"Pipeline(config={str(self.config)}, steps={len(self.steps)})"


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

    def prepare(self) -> None:
        """Prepare the executor for running the pipeline. Called once before execution."""
        ...

    def on_cycle_start(self) -> None:
        """Called at the start of each execution cycle to reset state if needed."""
        ...

    def on_cycle_end(self) -> None:
        """Called at the end of each execution cycle to finalize state if needed."""
        ...

    def cleanup(self) -> None:
        """Cleanup resources after execution is complete. Called once after execution."""
        ...

    # Main execution method
    def run(
        self, input_data: PipelineInputType
    ) -> "PipelineResult[PipelineOutputType]":
        """Execute the pipeline with the given input data and return the result."""
        ...


@dataclass(slots=True)
class PipelineResult(Generic[PipelineOutputType]):
    output: PipelineOutputType
    control: PipelineControlMessage


@runtime_checkable
class HasState(Protocol):
    def get_state(self) -> dict[str, torch.Tensor]:
        """Get the current state of the executor as a dictionary of tensors."""
        ...

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load the executor state from a dictionary of tensors."""
        ...
