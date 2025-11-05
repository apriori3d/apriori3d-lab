from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, overload

from apriori.ico.core.runtime.execution import IcoExecutionMixin
from apriori.ico.core.runtime.lifecycle import IcoLifecycleMixin
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

# ──── Generic type variables for composition ────

I2 = TypeVar("I2")
O2 = TypeVar("O2")


# ─── Operator Class ───
class IcoOperator(
    IcoOperatorProtocol[I, O],
    Generic[I, O],
    IcoLifecycleMixin,  # Added lifecycle management
    IcoExecutionMixin[I, O],  # Added execution state tracking
):
    """
    An atomic transformation unit following the ICO convention.

    ICO form:
        I → O
        fn: I → O

    An `IcoOperator` wraps a callable and provides:
    • composable transformations via `>>` or `|`
    • lazy mapping over iterables with `.map()`
    • structured graph representation for flow inspection

    Operators are the fundamental building blocks of ICO pipelines.
    They can represent stateless or stateful transformations,
    depending on the behavior of the wrapped callable.

    Example:
        >>> from apriori.ico import IcoOperator

        >>> to_float = IcoOperator(float)
        >>> scale = IcoOperator(lambda x: x * 2)
        >>> to_str = IcoOperator(str)

        # Compose: I → O → O2 → O3 == I → O3
        >>> pipeline = to_float | scale | to_str
        >>> print(pipeline("21.0"))
        '42.0'

        # Lazy map over iterable
        >>> mapped = scale.map()
        >>> print(list(mapped([1, 2, 3])))
        [2, 4, 6]

        # Inspect flow
        >>> flow = pipeline.describe_flow()
        >>> print(flow.name)
        to_float | scale | to_string
    """

    __slots__ = ("fn", "name", "node_type", "children")

    fn: Callable[[I], O]

    # -── Flow introspection ───
    name: str
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
        self.name = name or f"IcoOperator[{str(self.fn)}]"
        self.node_type = node_type
        self.children = children if children is not None else []

    # ─── Properties ───

    def __str__(self) -> str:
        return self.name

    # ─── Operator execution ───

    @overload
    def __call__(self, item: I) -> O:
        # Method overload for standard call with input
        ...

    @overload
    def __call__(self) -> O:
        # Method overload for no-argument call in flow with IcoSource
        ...

    def __call__(self, item: I | None = None, *args: Any) -> O:
        if item is not None:
            # Call for standard operator with input
            return self.track(self.fn, item)

        # Call for IcoSource with no input
        return self.track(self.fn, None)  # type: ignore

    # ─── Composition ───

    def compose(self, other: IcoOperatorProtocol[O, O2]) -> IcoOperator[I, O2]:
        """Function composition: (I → O, O → O2) == I → O2."""

        def composed(x: I) -> O2:
            return other(self(x))

        return IcoOperator(
            fn=composed,
            name=f"{self.name} | {other.name}",
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

        return IcoOperator(
            fn=self._map_fn,
            name=f"{self.name}.map",
            node_type=NodeType.map,
            children=[self],
        )

    def _map_fn(self, xs: Iterable[I]) -> Iterable[O]:
        for x in xs:
            yield self(x)


def wrap_operator(
    fn: Callable[[I], O],
) -> IcoOperatorProtocol[I, O]:
    """Wrap a callable into an IcoOperator if it is not already one."""
    return fn if isinstance(fn, IcoOperatorProtocol) else IcoOperator(fn=fn)
