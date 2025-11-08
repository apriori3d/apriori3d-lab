from __future__ import annotations

from typing import Any

from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeState,
    SupportsIcoRuntime,
)
from apriori.ico.core.types import IcoOperatorProtocol

COMMAND_TO_STATE = {
    IcoRuntimeCommand.activate: IcoRuntimeState.ready,
    IcoRuntimeCommand.reset: IcoRuntimeState.ready,
    IcoRuntimeCommand.deactivate: IcoRuntimeState.inactive,
    IcoRuntimeCommand.pause: IcoRuntimeState.paused,
    IcoRuntimeCommand.resume: IcoRuntimeState.running,
}


class IcoRuntimeMixin:
    """
    Mixin for runtime command propagation in both directions.

     Downward:
         `broadcast_command()` — spread commands (activate, reset, deactivate) to all children.
     Upward:
         `bubble_command()` — send runtime signals (stop_iteration, fault) to nearest runtime host.

     This allows full bidirectional control flow across the operator tree.

     Example:
         >>> from apriori.ico.core.runtime import IcoRuntimeMixin, IcoRuntimeCommand
         >>> IcoRuntimeMixin.broadcast_command(pipeline, IcoRuntimeCommand.activate)
    """

    _state: IcoRuntimeState
    _last_command: IcoRuntimeCommand | None

    def __init__(self) -> None:
        super().__init__()
        self._state = IcoRuntimeState.inactive
        self._last_command = None

    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state of the operator."""
        return self._state

    @property
    def last_command(self) -> IcoRuntimeCommand | None:
        """Last received runtime command."""
        return self._last_command

    # ─── Command Handling ───

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self._state = COMMAND_TO_STATE.get(command, self._state)
        self._last_command = command

    # ─── Recursive Broadcast ───

    @staticmethod
    def broadcast_command_static(
        operator: IcoOperatorProtocol[Any, Any],
        command: IcoRuntimeCommand,
    ) -> None:
        """
        Recursively propagate a runtime command through the operator tree.

        Each node implementing `SupportsIcoRuntime` receives `on_command(command)`.
        """
        if isinstance(operator, SupportsIcoRuntime):
            operator.on_command(command)

        for child in getattr(operator, "children", []):
            IcoRuntimeMixin.broadcast_command_static(child, command)

    @staticmethod
    def bubble_command_static(
        operator: IcoOperatorProtocol[Any, Any], command: IcoRuntimeCommand
    ) -> None:
        """
        Propagate a runtime command upward until a contour or agent host is reached.
        """
        node = operator
        while node.parent is not None:
            node = node.parent
            if isinstance(node, SupportsIcoRuntime):
                node.on_command(command)
                break
