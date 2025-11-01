from dataclasses import dataclass

from apriori.flow.core.pipeline.types import (
    InputStepType,
    OutputStepType,
    PipelineConfigType,
    PipelineInputType,
    PipelineOutputType,
    PipelineProtocol,
    StepType,
)
from apriori.flow.structure import FlowStructure, HasFlowStructure


@dataclass(slots=True)
class Pipeline(
    PipelineProtocol[PipelineConfigType, PipelineInputType, PipelineOutputType],
    HasFlowStructure,
):
    config: PipelineConfigType
    input_step: InputStepType
    steps: list[StepType]
    output_step: OutputStepType

    # Flow structure

    def describe_structure(self) -> FlowStructure:
        # Describe the structure of the steps
        steps_structures = []

        if isinstance(self.input_step, HasFlowStructure):
            steps_structures.append(self.input_step.describe_structure())
        else:
            steps_structures.append(
                FlowStructure(name=str(self.input_step), type="input_step")
            )

        for step in self.steps:
            if isinstance(step, HasFlowStructure):
                steps_structures.append(step.describe_structure())
            else:
                steps_structures.append(FlowStructure(name=str(step), type="step"))

        if isinstance(self.output_step, HasFlowStructure):
            steps_structures.append(self.output_step.describe_structure())
        else:
            steps_structures.append(
                FlowStructure(name=str(self.output_step), type="output_step")
            )

        return FlowStructure(
            name="Pipeline",
            type="pipeline",
            children=steps_structures,
        )

    # Properties

    @property
    def num_steps(self) -> int:
        return len(self.steps) + 2  # Including input and output steps

    def __str__(self) -> str:
        return f"Pipeline({str(self.config)})"

    def __repr__(self) -> str:
        return f"Pipeline(config={str(self.config)}, steps={len(self.steps)})"
