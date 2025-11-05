from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import C, I, IcoOperatorProtocol, NodeType, O


@final
class IcoPipeline(IcoOperator[I, O], Generic[I, C, O], IcoOperatorProtocol[I, O]):
    """
    A transformation pipeline following the ICO convention.

    ICO form:
        I → C → O
        context: I → C
        body:    [C → C]
        output:   C → O

    Each component can be any callable or nested IcoOperator.

    The `body` operates directly on the context `C`, possibly mutating it.
    This allows stateful or iterative transformations (e.g. model updates,
    accumulated metrics, cached buffers) within the same pipeline lifecycle.

    Example:
        >>> from apriori.ico.core import IcoOperator, IcoPipeline

        >>> to_float = IcoOperator(float)
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> to_string = IcoOperator(str)

        >>> pipeline = IcoPipeline(
        ...     context=to_float,
        ...     body=[scale],
        ...     output=to_string,
        ... )
        >>> print(pipeline("21.0"))
        '42.0'

        # ICO structure:
        # pipeline
        # ├── operator[to_float]
        # ├── operator[scale]
        # └── operator[to_string]
    """

    __slots__ = ("context", "body", "output")

    context: IcoOperatorProtocol[I, C]
    body: Sequence[IcoOperatorProtocol[C, C]]
    output: IcoOperatorProtocol[C, O]

    def __init__(
        self,
        context: IcoOperatorProtocol[I, C],
        body: Sequence[IcoOperatorProtocol[C, C]],
        output: IcoOperatorProtocol[C, O],
    ):
        super().__init__(
            fn=self._run_pipeline,
            node_type=NodeType.pipeline,
            children=[context] + list(body) + [output],
        )
        self.context = context
        self.body = body
        self.output = output

    def _run_pipeline(self, item: I) -> O:
        ctx = self.context(item)
        for step in self.body:
            ctx = step(ctx)
        return self.output(ctx)

    def __len__(self) -> int:
        return len(self.body)

    def __iter__(self) -> Iterator[IcoOperatorProtocol[C, C]]:
        yield from self.body
