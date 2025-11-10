from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from typing_extensions import Self

from apriori.ico.core.dsl.operator import O2, iterate_nodes
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.types import ProgressProtocol, SupportsProgress
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeHost,
    IcoRuntimeProtocol,
    IcoRuntimeState,
)
from apriori.ico.core.types import IcoOperatorProtocol, NodeType

COMMAND_TO_STATE = {
    IcoRuntimeCommand.activate: IcoRuntimeState.ready,
    IcoRuntimeCommand.reset: IcoRuntimeState.ready,
    IcoRuntimeCommand.deactivate: IcoRuntimeState.inactive,
    IcoRuntimeCommand.pause: IcoRuntimeState.paused,
    IcoRuntimeCommand.resume: IcoRuntimeState.running,
}


class IcoRuntimeMixin(IcoRuntimeProtocol):
    name: str
    children: list[IcoOperatorProtocol[Any, Any]]
    parent: IcoOperatorProtocol[Any, Any] | None
    node_type: NodeType
    fn: Callable[[None], None]

    _state: IcoRuntimeState
    _last_command: IcoRuntimeCommand | None
    _last_event: IcoRuntimeEvent | None

    def __init__(self) -> None:
        self._state = IcoRuntimeState.inactive
        self._last_command = None
        self._last_event = None
        self.name = self.__class__.__name__
        self.children = []
        self.parent = None
        self.node_type = NodeType.runtime
        self.fn = lambda _: None

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
        self.broadcast_command(command)

    # ─── Command propagation ───

    def broadcast_command(
        self: IcoRuntimeProtocol,
        command: IcoRuntimeCommand,
    ) -> None:
        """
        Recursively propagate a runtime command through the operator tree.

        Each node implementing `SupportsIcoRuntime` receives `on_command(command)`.
        """
        for child in self.children:
            if not isinstance(child, IcoRuntimeProtocol):
                raise TypeError(
                    f"Child operator {child} in runtime should be an instance of IcoRuntimeProtocol"
                )
            self.on_command(command)
            child.broadcast_command(command)

    # ─── Event Handling ───

    def on_event(self, event: IcoRuntimeEvent) -> None:
        """
        Handle a single runtime event.

        Subclasses may override to implement additional behavior
        (e.g., logging, metrics, or alerting).
        """
        self._last_event = event
        self.bubble_event(event)

    # ─── Event propagation ───

    def bubble_event(self, event: IcoRuntimeEvent) -> None:
        """
        Propagate a runtime event upward until a contour or agent host is reached.
        """
        node: IcoRuntimeProtocol = self
        while node.parent is not None:
            if not isinstance(node.parent, IcoRuntimeProtocol):
                raise TypeError(
                    f"Parent operator {node} in runtime should be an instance of IcoRuntimeProtocol"
                )
            node = node.parent
            node.on_event(event)

    # ─── Runtime Discovery and Connection ───

    def discover_runtime(
        self, closure: IcoOperatorProtocol[None, None]
    ) -> Iterator[IcoRuntimeProtocol]:
        """Discover all runtime hosts within the given closure."""
        if isinstance(closure, IcoRuntimeHost):
            raise ValueError(
                "Cannot discover runtime within a closure that is itself a runtime"
            )
        yield from self._discover_runtime_deep(closure)

    def _discover_runtime_deep(
        self, operator: IcoOperatorProtocol[Any, Any], in_runtime_scope: bool = False
    ) -> Iterator[IcoRuntimeProtocol]:
        """Discover all runtime hosts within the given closure."""

        if isinstance(operator, IcoRuntimeHost):
            # If we are already in a runtime scope, do not yield nested hosts
            if in_runtime_scope:
                return
            yield operator.runtime
            in_runtime_scope = True

        for child in operator.children:
            if isinstance(child, IcoRuntimeProtocol):
                yield from self._discover_runtime_deep(child, in_runtime_scope)

    def connect_runtime(self, runtime: IcoRuntimeProtocol) -> None:
        """Connect to a runtime host for command propagation."""
        if runtime not in self.children and (runtime.parent and runtime.parent != self):
            self.children.append(runtime)
            runtime.parent = self

    def disconnect_runtime(self, runtime: IcoRuntimeProtocol) -> None:
        """Disconnect from a runtime host."""
        if runtime in self.children:
            self.children.remove(runtime)
            runtime.parent = None

    def discover_and_connect_runtimes(
        self, closure: IcoOperatorProtocol[None, None]
    ) -> None:
        """Discover and connect all runtime hosts within the given closure."""
        for runtime in self.discover_runtime(closure):
            self.connect_runtime(runtime)

    def disconnect_all_runtimes(self) -> None:
        """Disconnect from all connected runtime hosts."""
        for runtime in list(self.children):
            if isinstance(runtime, IcoRuntimeProtocol):
                self.disconnect_runtime(runtime)

    # ─── Execution ───

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        return self

    # ─── Lifecycle ───

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.activate)
        return self

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.reset)
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.deactivate)
        return self

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.pause)
        return self

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.resume)
        return self

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        self.on_command(IcoRuntimeCommand.stop)
        return self

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self:
        """
        Bind a shared progress relay to all progress-capable nodes.

        Returns:
            Self — allows chaining: contour.bind_progress().ready().run().idle()
        """
        self.progress = progress

        for runtime in iterate_nodes(self):
            if isinstance(runtime, SupportsProgress):
                runtime.progress = self.progress

        return self

    # ─── Declarative sync execution path ───

    def __call__(self, _: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Imperative async execution path ───

    async def run_async(self, item: None) -> None:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    # ─── Operator composition ───

    def chain(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def __or__(
        self, other: IcoOperatorProtocol[None, O2]
    ) -> IcoOperatorProtocol[None, O2]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )

    def map(self) -> IcoOperatorProtocol[Iterator[None], Iterator[None]]:
        raise RuntimeError(
            "IcoRuntimeMixin does not implement data flow and has ICO form () → ()"
        )
