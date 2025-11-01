from typing import Generic, Protocol, TypeAlias

from apriori.flow.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
    PipelineType,
)


class PipelineExecutorProtocol(
    Protocol,
    Generic[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
):
    pipeline: PipelineType
    context: PipelineContextType

    def __init__(self):
        super().__init__()

    # Main execution method
    def run(self, input_data: PipelineInputType) -> PipelineResult[PipelineOutputType]:
        """Execute the pipeline with the given input data and return the result."""
        ...


PipelineExecutorType: TypeAlias = PipelineExecutorProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
]
