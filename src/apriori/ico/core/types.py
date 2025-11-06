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
    chain = auto()
    map = auto()
    pipeline = auto()
    stream = auto()
    process = auto()
    source = auto()
    agent = auto()
    agent_host = auto()


# ─── Operator Protocol ───
@runtime_checkable
class IcoOperatorProtocol(Protocol[I, O]):
    """
    Protocol for ICO Operators, defining the expected interface.
    """

    # ─── Core callable function ───

    fn: Callable[[I], O]

    # -── Structural attributes for graph representation ───

    name: str
    node_type: NodeType
    children: list[IcoOperatorProtocol[Any, Any]]

    # ─── Declarative sync execution path ───

    def __call__(self, item: I) -> O: ...

    # ─── Imperative async execution path ───

    async def run_async(self, item: I) -> O: ...
