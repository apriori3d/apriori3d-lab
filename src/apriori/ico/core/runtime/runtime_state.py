from __future__ import annotations

from collections.abc import Callable

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeProtocol,
    IcoRuntimeState,
    IcoRuntimeStateProtocol,
)

COMMAND_TO_STATE = {
    IcoRuntimeCommand.activate: IcoRuntimeState.ready,
    IcoRuntimeCommand.reset: IcoRuntimeState.ready,
    IcoRuntimeCommand.deactivate: IcoRuntimeState.inactive,
    IcoRuntimeCommand.pause: IcoRuntimeState.paused,
    IcoRuntimeCommand.resume: IcoRuntimeState.running,
}


class IcoRuntimeStateMixin(IcoRuntimeStateProtocol):
    _state: IcoRuntimeState
    _last_command: IcoRuntimeCommand | None
    _last_event: IcoRuntimeEvent | None
    __as_runtime: IcoRuntimeProtocol

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(self, IcoRuntimeProtocol):
            raise TypeError(
                "IcoRuntimeLifecycleMixin can only be used with IcoRuntimeProtocol instances"
            )
        self.__as_runtime = self
        self._state = IcoRuntimeState.inactive
        self._last_command = None
        self._last_event = None

    # ─── Runtime State tracking ───

    def _track(self, fn: Callable[[None], None]) -> None:
        """Execute function while managing runtime state transitions."""
        try:
            self._state = IcoRuntimeState.running
            fn(None)
            self._state = IcoRuntimeState.ready
        except Exception:
            self._state = IcoRuntimeState.error
            raise

    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state of the operator."""
        return self._state

    @property
    def last_command(self) -> IcoRuntimeCommand | None:
        """Last received runtime command."""
        return self._last_command

    @property
    def last_event(self) -> IcoRuntimeEvent | None:
        """Last received runtime event."""
        return self._last_event

    # ─── Command Handling ───

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """
        Handle a single runtime command.

        Subclasses may override to implement additional behavior
        (e.g., resource allocation, reset hooks, or teardown logic).
        """
        self._state = COMMAND_TO_STATE.get(command, self._state)
        self._last_command = command
        self.__as_runtime.broadcast_command(command)

    # ─── Event Handling ───

    def on_event(self, event: IcoRuntimeEvent) -> None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
        self._last_event = event
        self.__as_runtime.bubble_event(event)
