from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Protocol, TypeVar

# ──── Generic type variables for ICO model ────

I = TypeVar("I")  # noqa: E741
C = TypeVar("C")
O = TypeVar("O")  # noqa: E741

# ─── Node Types ───


class NodeType(Enum):
    operator = auto()
    map = auto()
    compose = auto()
    pipeline = auto()
    context = auto()
    step = auto()
    output = auto()
    stream = auto()
    data = auto()
    process = auto()


# ─── Operator Protocol ───


class IcoOperatorProtocol(Protocol[I, O]):
    """
    Protocol for ICO Operators, defining the expected interface.
    """

    fn: Callable[[I], O]
    name: str | None
    node_type: NodeType
    children: list[IcoOperatorProtocol[Any, Any]]

    def __call__(self, item: I) -> O: ...
