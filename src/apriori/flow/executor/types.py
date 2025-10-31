from typing import Generic, Protocol

from apriori.flow.lifecycle import HasLifecycle
from apriori.flow.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    PipelineProtocol,
    PipelineResult,
)
from apriori.flow.structure import HasFlowStructure


class PipelineExecutorProtocol(
    Protocol,
    Generic[
        PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
    ],
    HasLifecycle,
    HasFlowStructure,
):
    pipeline: PipelineProtocol[
        PipelineConfigType,
        PipelineInputType,
        PipelineOutputType,
    ]
    context: PipelineContextType

    # Main execution method
    def run(self, input_data: PipelineInputType) -> PipelineResult[PipelineOutputType]:
        """Execute the pipeline with the given input data and return the result."""
        ...
