from __future__ import annotations

from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import C, IcoOperatorProtocol, NodeType


@final
class IcoProcess(IcoOperator[C, C], Generic[C], IcoOperatorProtocol[C, C]):
    """
    Repeatedly applies an operator to the same context.

    Represents a fixed update rule applied several times to the same state.
    Useful for iterative refinement, optimization, or simulation steps.

    ICO form:
        C → C → C   (repeated `steps` times)

    Example:
        decay = IcoOperator[float, float](lambda x: x * 0.9)
        process = IcoProcess(decay, steps=3)
        result = process(1.0)  # 0.9³ = 0.729
    """

    __slots__ = (
        "body",
        "num_iterations",
    )

    body: IcoOperatorProtocol[C, C]

    def __init__(
        self,
        body: IcoOperatorProtocol[C, C],
        num_iterations: int,
    ):
        super().__init__(
            fn=self._run_loop,
            node_type=NodeType.process,
            children=[body],
        )
        self.body = body
        self.num_iterations = num_iterations

    def _run_loop(self, context: C) -> C:
        for _ in range(self.num_iterations):
            context = self.body(context)
        return context
