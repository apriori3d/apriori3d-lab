from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from typing_extensions import Self

from apriori.ico.core.dsl.operator import O2
from apriori.ico.core.runtime.runtime_hierarchy import IcoRuntimeHierarchyMixin
from apriori.ico.core.runtime.runtime_lifecycle import IcoRuntimeLifecycleMixin
from apriori.ico.core.runtime.runtime_state import IcoRuntimeStateMixin
from apriori.ico.core.runtime.types import IcoRuntimeProtocol
from apriori.ico.core.types import IcoOperatorProtocol, NodeType


class IcoRuntimeOperator(
    IcoRuntimeStateMixin,
    IcoRuntimeHierarchyMixin,
    IcoRuntimeLifecycleMixin,
    IcoRuntimeProtocol,
):
    name: str
    children: list[IcoOperatorProtocol[Any, Any]]
    parent: IcoOperatorProtocol[Any, Any] | None
    node_type: NodeType
    fn: Callable[[None], None]

    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.children = []
        self.parent = None
        self.node_type = NodeType.runtime
        self.fn = self._noop_fn

    # ─── Execution ───

    def _noop_fn(self, _: None) -> None:
        pass

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        return self

    # ─── Declarative sync execution path ───

    def __call__(self, _: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Imperative async execution path ───

    async def run_async(self, item: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Operator composition ───

    def chain(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def __or__(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def map(self) -> IcoOperatorProtocol[Iterator[None], Iterator[None]]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )
