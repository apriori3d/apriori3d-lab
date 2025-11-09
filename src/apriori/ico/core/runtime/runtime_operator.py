from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.meta.ico_form import infer_ico_form
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_mixin import IcoRuntimeMixin
from apriori.ico.core.runtime.types import IcoRuntimeCommand, IcoRuntimeState
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O


class IcoRuntimeOperator(IcoOperator[I, O], Generic[I, O], IcoRuntimeMixin):
    """
    Extension of `IcoOperator` with runtime command handling.

    This operator participates in the ICO runtime layer — it can
    receive, process, and react to runtime commands such as
    activation, reset, pause, or stop, modifying its internal state
    accordingly.

    Purpose:
        Bridge between declarative operators (I → O)
        and the runtime control system managing state and lifecycle.

    ICO form:
        I → O
        fn: I → O

    Features:
        • Tracks runtime state transitions during execution.
        • Reacts to runtime commands (activate, reset, stop).
        • Automatically sets `error` state on exceptions.
        • Integrates with runtime contours and agents.
    """

    def __init__(
        self,
        fn: Callable[[I], O],
        name: str | None = None,
        node_type: NodeType = NodeType.operator,
        children: list[IcoOperatorProtocol[Any, Any]] | None = None,
    ):
        IcoOperator.__init__(
            self,
            fn=fn,
            name=name,
            node_type=node_type,
            children=children,
        )
        IcoRuntimeMixin.__init__(self)

    # ─── Execution ───

    def __call__(self, item: I | None = None, *args: Any) -> O:
        """Execute the wrapped function and update runtime state."""
        if item is not None:
            return self._track(self.fn, item)
        return self._track(self.fn, None)  # type: ignore

    # ─── Runtime State tracking ───

    def _track(self, fn: Callable[[I], O], item: I) -> O:
        """Execute function while managing runtime state transitions."""
        try:
            self._state = IcoRuntimeState.running
            output = fn(item)
            self._state = IcoRuntimeState.ready
            return output
        except Exception:
            self._state = IcoRuntimeState.error
            raise

    # ─── Command Handling ───

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """
        Handle an incoming runtime command.

        Default behavior delegates to `IcoRuntimeMixin` and
        updates internal state. Subclasses can override this
        to react to specific runtime commands.
        """
        IcoRuntimeMixin.on_command(self, command)

    # ─── Command Propagation ───

    def broadcast_command(self, command: IcoRuntimeCommand) -> None:
        """
        Broadcast a runtime command to all child operators.

        This propagates the command downward through the operator tree,
        allowing all children to react accordingly.
        """
        IcoRuntimeMixin.broadcast_command_static(self, command)

    # ─── Event Propagation ───

    def bubble_event(self, event: IcoRuntimeEvent) -> None:
        """
        Bubble a runtime event up to the nearest runtime host.

        This sends the event upward through the operator tree,
        allowing parent operators to handle it.
        """
        IcoRuntimeMixin.bubble_event_static(self, event)

    # ─── Runtime Endpoint Attachment ───

    def connect_runtime(self, endpoint: IcoOperatorProtocol[Any, Any]) -> None:
        """Connect a runtime endpoint (boundary operator) for command propagation.

        Endpoints are boundary operators with ICO forms:
            () → Any  (runtime source)
            Any → ()  (runtime sink)

        This allows runtime commands to cross contours (e.g. host ↔ agent links)
        without breaking declarative isolation.
        """
        ico_form = infer_ico_form(endpoint)
        if ico_form.i != "()" or ico_form.o != "()":
            raise ValueError(
                f"Invalid runtime endpoint form: expected (() → Any) or (Any → ()), got {ico_form}"
            )
        if endpoint not in self.children:
            self.children.append(endpoint)
            endpoint.parent = self
