from collections.abc import Callable
from typing import Any

from apriori.ico.core import IcoOperator, IcoPipeline
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeMixin,
    IcoRuntimeState,
)
from apriori.ico.core.types import IcoOperatorProtocol, NodeType


def test_lifecycle_broadcast_updates_nested_states() -> None:
    """
    Test that lifecycle events correctly propagate through nested ICO operators.

    Structure:
        pipeline
        ├── stateful_a
        ├── stateless
        └── stateful_b
            └── child_stateful

    Expected:
        - prepare → all stateful operators become 'prepared'
        - reset → all stateful operators become 'ready'
        - cleanup → all stateful operators become 'cleaned'
    """

    class StatefulOp(
        IcoOperator[int, int],
        IcoRuntimeMixin,
    ):
        """A simple stateful operator that records received events."""

        def __init__(self, name: str):
            IcoOperator.__init__(self, lambda x: x, name=name)
            IcoRuntimeMixin.__init__(self)

            self.events: list[IcoRuntimeCommand] = []

        def on_event(self, event: IcoRuntimeCommand) -> None:
            super().on_event(event)
            self.events.append(event)

    class StatelessOp:
        """A simple stateless operator that ignores lifecycle events."""

        fn: Callable[[int], int]
        name: str
        node_type: NodeType
        children: list[IcoOperatorProtocol[Any, Any]]

        def __init__(self, name: str):
            self.name = name
            self.node_type = NodeType.operator
            self.children = []
            self.fn = self.__call__

        def __call__(self, input: int) -> int:
            return input

    # Create nested operator structure
    stateful_a = StatefulOp("A")
    stateful_b = StatefulOp("B")
    stateful_child = StatefulOp("B_child")
    stateless = StatelessOp(name="noop")

    # Nest B_child under B
    stateful_b.children.append(stateful_child)

    # Build pipeline with mixed children
    pipeline = IcoPipeline[int, int, int](
        context=stateful_a,
        body=[stateless, stateful_b],
        output=lambda x: x,
    )

    # ── 1. Prepare phase ──────────────────────────────
    pipeline.broadcast_event(IcoRuntimeCommand.activate)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoRuntimeState.running
        assert IcoRuntimeCommand.activate in op.events

    # ── 2. Reset phase ────────────────────────────────
    pipeline.broadcast_event(IcoRuntimeCommand.reset)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoRuntimeState.running
        assert IcoRuntimeCommand.reset in op.events

    # ── 3. Cleanup phase ──────────────────────────────
    pipeline.broadcast_event(IcoRuntimeCommand.deavtivate)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoRuntimeState.cleaned
        assert IcoRuntimeCommand.deavtivate in op.events

    # Ensure stateless node never changed state (not lifecycle-aware)
    assert not hasattr(stateless, "state")


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
