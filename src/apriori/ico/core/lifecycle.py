from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from apriori.ico.core.types import IcoOperatorProtocol

# ──── Lifecycle Events & States ────


class IcoLifecycleEvent(Enum):
    """Discrete lifecycle events broadcast through the ICO operator tree."""

    prepare = auto()  # Initialize or allocate resources
    reset = auto()  # Reset internal state for stateful operators (weights, cache, etc.)
    cleanup = auto()  # Release resources and temporary buffers


class IcoLifecycleState(Enum):
    """Declarative lifecycle state of an ICO operator."""

    unknown = auto()  # Default state (uninitialized)
    prepared = auto()  # Operator initialized and ready to start work
    ready = auto()  # Operator actively usable (after reset)
    cleaned = auto()  # Operator has released all resources


# ──── Mapping between events and resulting states ────

EVENT_TO_STATE = {
    IcoLifecycleEvent.prepare: IcoLifecycleState.prepared,
    IcoLifecycleEvent.reset: IcoLifecycleState.ready,
    IcoLifecycleEvent.cleanup: IcoLifecycleState.cleaned,
}


# ──── Protocol for lifecycle-capable operators ────


@runtime_checkable
class SupportsIcoLifecycle(Protocol):
    """
    Interface for operators that react to lifecycle events.

    Each implementing operator should:
      • maintain a `.state` field (of type IcoLifecycleState)
      • implement `.on_event(event)` to handle incoming lifecycle events

    Lifecycle events are typically propagated using `IcoLifecycleMixin.broadcast_event()`.
    """

    state: IcoLifecycleState

    def on_event(self, event: IcoLifecycleEvent) -> None: ...


# ──── Lifecycle mixin (event broadcaster) ────


class IcoLifecycleMixin:
    """
    Utility mixin for propagating lifecycle events through an ICO operator tree.

    This mixin implements a minimal default behavior:
      • each lifecycle event updates the operator's `.state`
      • events are recursively broadcast to child operators

    Example:
        >>> from apriori.ico.core.lifecycle import IcoLifecycleMixin, IcoLifecycleEvent
        >>> IcoLifecycleMixin.broadcast_event(pipeline, IcoLifecycleEvent.prepare)
    """

    state: IcoLifecycleState

    def __init__(self) -> None:
        super().__init__()
        self.state = IcoLifecycleState.unknown

    def on_event(self, event: IcoLifecycleEvent) -> None:
        """Default no-op handler for lifecycle events. Override in subclasses if needed."""
        return None

    @staticmethod
    def broadcast_event(
        operator: IcoOperatorProtocol[Any, Any],
        event: IcoLifecycleEvent,
    ) -> None:
        """
        Recursively propagate a lifecycle event through the operator tree.

        Each node implementing `SupportsIcoLifecycle`:
          • receives the event via `.on_event(event)`
          • updates its `.state` according to EVENT_TO_STATE
        """
        if isinstance(operator, SupportsIcoLifecycle):
            operator.on_event(event)
            operator.state = EVENT_TO_STATE.get(event, operator.state)

        for child in getattr(operator, "children", []):
            IcoLifecycleMixin.broadcast_event(child, event)
