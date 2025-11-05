from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Generic

from apriori.ico.core.types import I, O


class IcoExecutionState(Enum):
    """Local, runtime-level execution state of an operator."""

    idle = auto()
    running = auto()
    done = auto()
    faulted = auto()


class IcoExecutionMixin(Generic[I, O]):
    """
    Mixin for tracking and reacting to an operator's execution state.

    Emits execution events via `on_exec_event(state)` at each transition.

    Example:
        >>> class DebugOp(IcoExecutionMixin[int, int]):
        ...     def on_exec_event(self, state):  # optional hook
        ...         print(f"State changed to: {state.name}")
        ...     def __call__(self, x: int) -> int:
        ...         return self.track(lambda v: v * 2, x)
        >>> op = DebugOp()
        >>> op(3)
        State changed to: running
        State changed to: done
    """

    exec_state: IcoExecutionState

    def __init__(self) -> None:
        super().__init__()
        self.exec_state = IcoExecutionState.idle

    # ─── Execution tracking ───

    def track(self, fn: Callable[[I], O], item: I) -> O:
        """Execute fn(item) while updating state and emitting events."""
        self._set_state(IcoExecutionState.running)
        try:
            result = fn(item)
        except Exception:
            self._set_state(IcoExecutionState.faulted)
            raise
        else:
            self._set_state(IcoExecutionState.done)
            return result

    # ─── Event handling ───

    def _set_state(self, state: IcoExecutionState) -> None:
        self.exec_state = state
        self.on_exec_event(state)

    def on_exec_event(self, state: IcoExecutionState) -> None:
        """Optional hook called on each execution state change."""
        pass
