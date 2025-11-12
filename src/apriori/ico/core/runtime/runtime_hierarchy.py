from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from typing_extensions import Self

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.types import ProgressProtocol, SupportsProgress
from apriori.ico.core.runtime.types import (
    ConnectedToIcoRuntime,
    IcoRuntimeCommand,
    IcoRuntimeHierarchyProtocol,
    IcoRuntimeProtocol,
)
from apriori.ico.core.types import IcoOperatorProtocol


class IcoRuntimeHierarchyMixin(IcoRuntimeHierarchyProtocol):
    runtime_children: list[IcoRuntimeProtocol]
    runtime_parent: IcoRuntimeProtocol | None
    __as_runtime: IcoRuntimeProtocol

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(self, IcoRuntimeProtocol):
            raise TypeError(
                "IcoRuntimeLifecycleMixin can only be used with IcoRuntimeProtocol instances"
            )
        self.__as_runtime = self
        self.runtime_children = []
        self.runtime_parent = None

    # ─── Command propagation ───

    def broadcast_command(
        self,
        command: IcoRuntimeCommand,
    ) -> None:
        """
        Recursively propagate a runtime command through the operator tree.

        Each node implementing `SupportsIcoRuntime` receives `on_command(command)`.
        """
        for child in self.runtime_children:
            if not isinstance(child, IcoRuntimeProtocol):
                raise TypeError(
                    f"Child operator {child} in runtime should be an instance of IcoRuntimeProtocol"
                )
            child.on_command(command)
            child.broadcast_command(command)

    # ─── Event propagation ───

    def bubble_event(self, event: IcoRuntimeEvent) -> None:
        """
        Propagate a runtime event upward until a contour or agent host is reached.
        """
        node: IcoRuntimeProtocol = self.__as_runtime
        while node.runtime_parent is not None:
            if not isinstance(node.runtime_parent, IcoRuntimeProtocol):
                raise TypeError(
                    f"Parent operator {node} in runtime should be an instance of IcoRuntimeProtocol"
                )
            node = node.runtime_parent
            node.on_event(event)

    # ─── Runtime Discovery and Connection ───

    def discover_runtime(
        self, closure: IcoOperatorProtocol[None, None]
    ) -> Iterator[IcoRuntimeProtocol]:
        """Discover all runtime hosts within the given closure."""
        if isinstance(closure, ConnectedToIcoRuntime):
            raise ValueError(
                "Cannot discover runtime within a closure that is itself a runtime"
            )
        yield from self._discover_runtime_deep(closure)

    def _discover_runtime_deep(
        self, operator: IcoOperatorProtocol[Any, Any], in_runtime_scope: bool = False
    ) -> Iterator[IcoRuntimeProtocol]:
        """Discover all runtime hosts within the given closure."""

        if isinstance(operator, ConnectedToIcoRuntime):
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
        if runtime not in self.__as_runtime.runtime_children and (
            runtime.runtime_parent and runtime.runtime_parent != self.__as_runtime
        ):
            self.__as_runtime.runtime_children.append(runtime)
            runtime.runtime_parent = self.__as_runtime

    def disconnect_runtime(self, runtime: IcoRuntimeProtocol) -> None:
        """Disconnect from a runtime host."""
        if runtime in self.runtime_children:
            self.runtime_children.remove(runtime)
            runtime.parent = None

    def discover_and_connect_runtimes(
        self, closure: IcoOperatorProtocol[None, None]
    ) -> None:
        """Discover and connect all runtime hosts within the given closure."""
        for runtime in self.discover_runtime(closure):
            self.connect_runtime(runtime)

    def disconnect_all_runtimes(self) -> None:
        """Disconnect from all connected runtime hosts."""
        for runtime in list(self.runtime_children):
            if isinstance(runtime, IcoRuntimeProtocol):
                self.disconnect_runtime(runtime)

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


def iterate_nodes(node: Any, strict: bool = True) -> Iterator[IcoRuntimeProtocol]:
    """Recursively yield all children operators in the flow tree."""
    if strict and not isinstance(node, IcoRuntimeProtocol):
        raise TypeError(
            f"Operator {node} in runtime should be an instance of IcoRuntimeProtocol"
        )
    yield node
    for c in node.runtime_children:
        yield from iterate_nodes(c, strict)
