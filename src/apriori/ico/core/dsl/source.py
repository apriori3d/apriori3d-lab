from collections.abc import Callable, Iterable
from typing import Generic, final

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.types import IcoOperatorProtocol, NodeType, O


@final
class IcoSource(
    IcoOperator[None, Iterable[O]],
    IcoOperatorProtocol[None, Iterable[O]],
    Generic[O],
):
    """
    A data source node in the ICO DSL.
    Produces data without requiring any input (acts as `() → Iterable[O]`).

    Example:
        >>> dataset = IcoSource(lambda: [1.0, 2.0, 3.0], name="dataset")
        >>> scale = IcoOperator(lambda x: x * 2, name="scale")
        >>> to_sum = IcoOperator(sum, name="sum")

        >>> flow = dataset | IcoStream(body=scale) | to_sum
        >>> result = flow()
        >>> print(result)
        12.0
    """

    def __init__(self, fn: Callable[[], Iterable[O]], name: str | None = None):
        # Note: we annotate the inner lambda explicitly to preserve type hints
        def wrapped(_: None) -> Iterable[O]:
            return fn()

        super().__init__(
            fn=wrapped,
            name=name or f"IcoSource[{str(fn)}]",
            node_type=NodeType.source,
            children=[],
        )
