from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar

from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

# ──── Generic type variables for composition ────

I2 = TypeVar("I2")
O2 = TypeVar("O2")


# ─── Base Operator ───
class IcoOperator(IcoOperatorProtocol[I, O], Generic[I, O]):
    """
    A composable callable transformation: I → O.

    The Operator is the atomic building block of the ICO flow.
    It wraps any function `fn: I → O` and allows:
      • functional composition: `a >> b` or `a | b`
      • lazy mapping over iterables: `op.map()`
      • hierarchical structure introspection (FlowStructure)

    Example:
        >>> from apriori.ico import IcoOperator

        # Define basic transformations
        >>> to_float = IcoOperator(float, name="to_float")
        >>> scale = IcoOperator(lambda x: x * 2, name="scale")
        >>> to_str = IcoOperator(str, name="to_string")

        # Compose them: I → O, O → O2, O2 → O3, I → O3
        >>> composed = to_float >> scale >> to_str
        >>> print(composed("21.5"))
        '43.0'

        # Apply lazily over an iterable
        >>> mapped = scale.map()
        >>> print(list(mapped([1, 2, 3])))
        [2, 4, 6]

        # Inspect flow
        >>> flow = composed.describe_flow()
        >>> print(flow.name)
        to_float >> scale >> to_string
    """

    __slots__ = ("fn", "name", "node_type", "children")

    fn: Callable[[I], O]

    # -── Structure ───
    name: str | None
    node_type: NodeType
    children: list[IcoOperatorProtocol[Any, Any]]

    def __init__(
        self,
        fn: Callable[[I], O],
        name: str | None = None,
        node_type: NodeType = NodeType.operator,
        children: list[IcoOperatorProtocol[Any, Any]] | None = None,
    ):
        self.fn = fn
        self.name = name
        self.node_type = node_type
        self.children = children if children is not None else []

    def __call__(self, item: I) -> O:
        return self.fn(item)

    # ─── Composition ───

    def compose(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperator[I, O2]:
        """Function composition: I → O, O → O2 → I → O2."""

        def composed(x: I) -> O2:
            return other(self(x))

        return IcoOperator(
            fn=composed,
            name=f"{self.name} >> {other.name}",
            node_type=NodeType.compose,
            children=[self, other],
        )

    def then(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperator[I, O2]:
        """Alias for compose, improves readability."""
        return self.compose(other)

    __or__ = compose
    __rshift__ = compose

    # ─── Map ───

    def map(self) -> IcoOperator[Iterable[I], Iterable[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """

        def map_fn(xs: Iterable[I]) -> Iterable[O]:
            for x in xs:
                yield self(x)

        return IcoOperator(
            fn=map_fn,
            name=f"{self.name}.map",
            node_type=NodeType.map,
            children=[self],
        )
