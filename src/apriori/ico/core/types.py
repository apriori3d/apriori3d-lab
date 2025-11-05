from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Protocol, TypeVar, runtime_checkable

# ──── Generic type variables for ICO model ────

I = TypeVar("I")  # noqa: E741
C = TypeVar("C")
O = TypeVar("O")  # noqa: E741


# ─── Node Types ───


class NodeType(Enum):
    operator = auto()
    compose = auto()
    map = auto()
    pipeline = auto()
    stream = auto()
    process = auto()
    source = auto()


# ─── Operator Protocol ───
@runtime_checkable
class IcoOperatorProtocol(Protocol[I, O]):
    """
    Protocol for ICO Operators, defining the expected interface.
    """

    fn: Callable[[I], O]
    name: str
    node_type: NodeType
    children: list[IcoOperatorProtocol[Any, Any]]

    def __call__(self, item: I) -> O: ...
