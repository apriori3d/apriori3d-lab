from collections.abc import Callable, Iterable
from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import IcoOperatorProtocol, NodeType, O


@final
class IcoData(
    IcoOperator[None, Iterable[O]], IcoOperatorProtocol[None, Iterable[O]], Generic[O]
):
    """
    Source node in the ICO DSL.
    Produces data without input (acts as () -> O).

    Example:
        dataset = IcoData(lambda: [1.0, 2.0, 3.0], name="dataset")
        loader = IcoData(lambda: (i for i in range(3)), name="loader")
    """

    def __init__(self, fn: Callable[[], Iterable[O]], name: str | None = None):
        super().__init__(
            fn=lambda _: fn(),  # I -> O signature
            name=name,
            node_type=NodeType.data,
            children=[],
        )
