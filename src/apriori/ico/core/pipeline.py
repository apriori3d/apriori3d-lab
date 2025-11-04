from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import C, I, IcoOperatorProtocol, NodeType, O

# ──── Pipeline  ────


@final
class IcoPipeline(IcoOperator[I, O], Generic[I, C, O], IcoOperatorProtocol[I, O]):
    """
    A transformation flow following the ICO convention:
        I → C → O
          context: I → C
          flow:   [C → C]
          output:  C → O

    Each component can be any callable or nested operator.

    Example:
        >>> from apriori.ico.core import IcoOperator, IcoPipeline

        >>> to_float = IcoOperator(float)
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> to_string = IcoOperator(str)

        >>> pipeline = IcoPipeline(context=to_float, flow=[scale], output=to_string)
        >>> result = pipeline("21.5")
        >>> print(result)
        '43.0'

        # pipeline: pipeline
        #   operator: to_float
        #   operator: scale
        #   operator: to_string
    """

    __slots__ = ("context", "flow", "output")

    context: IcoOperatorProtocol[I, C]
    flow: Sequence[IcoOperatorProtocol[C, C]]
    output: IcoOperatorProtocol[C, O]

    def __init__(
        self,
        context: IcoOperatorProtocol[I, C],
        flow: Sequence[IcoOperatorProtocol[C, C]],
        output: IcoOperatorProtocol[C, O],
    ):
        def pipeline_fn(item: I) -> O:
            ctx = context(item)
            for step in flow:
                ctx = step(ctx)
            return output(ctx)

        super().__init__(
            fn=pipeline_fn,
            node_type=NodeType.pipeline,
            children=[context] + list(flow) + [output],
        )
        self.context = context
        self.flow = flow
        self.output = output

    def __len__(self) -> int:
        return len(self.flow)

    def __iter__(self) -> Iterator[IcoOperatorProtocol[C, C]]:
        yield from self.flow
