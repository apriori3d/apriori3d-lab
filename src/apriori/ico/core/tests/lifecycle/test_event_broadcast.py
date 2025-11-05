from apriori.ico.core import IcoOperator, IcoPipeline
from apriori.ico.core.lifecycle import (
    IcoLifecycleEvent,
    IcoLifecycleMixin,
    IcoLifecycleState,
)


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
        IcoLifecycleMixin,
    ):
        """A simple stateful operator that records received events."""

        def __init__(self, name: str):
            IcoOperator.__init__(self, lambda x: x, name=name)
            IcoLifecycleMixin.__init__(self)

            self.events: list[IcoLifecycleEvent] = []

        def on_event(self, event: IcoLifecycleEvent) -> None:
            self.events.append(event)

    # Create nested operator structure
    stateful_a = StatefulOp("A")
    stateful_b = StatefulOp("B")
    stateful_child = StatefulOp("B_child")
    stateless = IcoOperator[int, int](lambda x: x, name="noop")

    # Nest B_child under B
    stateful_b.children.append(stateful_child)

    # Build pipeline with mixed children
    pipeline = IcoPipeline[int, int, int](
        context=stateful_a,
        body=[stateless, stateful_b],
        output=lambda x: x,
    )

    # ── 1. Prepare phase ──────────────────────────────
    IcoLifecycleMixin.broadcast_event(pipeline, IcoLifecycleEvent.prepare)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoLifecycleState.prepared
        assert IcoLifecycleEvent.prepare in op.events

    # ── 2. Reset phase ────────────────────────────────
    IcoLifecycleMixin.broadcast_event(pipeline, IcoLifecycleEvent.reset)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoLifecycleState.ready
        assert IcoLifecycleEvent.reset in op.events

    # ── 3. Cleanup phase ──────────────────────────────
    IcoLifecycleMixin.broadcast_event(pipeline, IcoLifecycleEvent.cleanup)
    for op in (stateful_a, stateful_b, stateful_child):
        assert op.state == IcoLifecycleState.cleaned
        assert IcoLifecycleEvent.cleanup in op.events

    # Ensure stateless node never changed state (not lifecycle-aware)
    assert not hasattr(stateless, "state")
