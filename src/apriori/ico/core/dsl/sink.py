from collections.abc import Callable, Iterable
from typing import Generic

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.types import I, NodeType


class IcoSink(IcoOperator[Iterable[I], None], Generic[I]):
    """Terminal operator: consumes output and ends the flow (I → ())."""

    def __init__(
        self, fn: Callable[[Iterable[I]], None], name: str | None = None
    ) -> None:
        super().__init__(fn=fn, name=name or "IcoSink", node_type=NodeType.sink)
