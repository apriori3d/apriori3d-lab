from collections.abc import Callable, Iterable
from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import IcoOperatorProtocol, NodeType, O


@final
class IcoSource(
    IcoOperator[None, Iterable[O]], IcoOperatorProtocol[None, Iterable[O]], Generic[O]
):
    """
    A data source node in the ICO DSL.
    Produces data without requiring any input (acts as `() → Iterable[O]`).

    Typically used as the input of an ICO flow.

    Example:
        |> from apriori.ico.core import IcoSource, IcoOperator, IcoStream

        |> dataset = IcoSource(lambda: [1.0, 2.0, 3.0], name="dataset")
        |> scale = IcoOperator(lambda x: x * 2, name="scale")
        |> to_sum = IcoOperator(sum, name="sum")

        |> flow = dataset | IcoStream(body=scale) | to_sum

        # Sources are called without arguments
        |> result = flow()
        |> print(result)
        12.0

        # dataset: data
        #   operator: scale
        #   operator: sum
    """

    def __init__(self, fn: Callable[[], Iterable[O]], name: str | None = None):
        super().__init__(
            fn=lambda _: fn(),  # () → Iterable[O]
            name=name,
            node_type=NodeType.data,
            children=[],
        )
