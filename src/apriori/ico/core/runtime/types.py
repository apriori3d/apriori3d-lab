from __future__ import annotations

from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.types import I, IcoOperatorProtocol, O

# ──── Runtime Commands ────


class IcoRuntimeCommand(Enum):
    """
    Runtime commands controlling activation and resource lifecycle
    across agents and runtime contours.

    Commands:
        • activate   - Allocate resources and prepare for execution
        • reset      - Reinitialize internal state (weights, cache, buffers)
        • deactivate - Release resources and mark as inactive
        • pause      - Temporarily suspend execution without teardown
        • resume     - Resume execution after pause
        • stop       - Stop signal for iterative or streaming operators
    """

    activate = auto()
    reset = auto()
    deactivate = auto()
    pause = auto()
    resume = auto()
    stop = auto()


# ──── State for runtime operators ────


class IcoRuntimeState(Enum):
    """
    Current runtime state of an agent and connected contour.

    States:
        • inactive - Operator uninitialized or fully released
        • ready    - Initialized and ready to run
        • running  - Currently executing or active
        • paused   - Temporarily suspended, resources preserved
        • error    - Faulted state after unrecoverable failure
    """

    inactive = auto()
    ready = auto()
    running = auto()
    paused = auto()
    error = auto()


# ──── Event types for runtime signaling ────


class IcoRuntimeEventType(Enum):
    fault = auto()
    heartbeat = auto()


# ──── Protocol for runtime operators ────


@runtime_checkable
class IcoRuntimeProtocol(Protocol):
    """
    Protocol for runtime-controllable ICO operators.

    Defines the minimal interface for operators that can
    participate in the runtime control flow — receiving and
    propagating execution commands such as activation,
    reset, or stop.

    Attributes:
        state: Current runtime state of the operator.
        last_command: The most recently received runtime command.

    Methods:
        on_command(command) -> None:
            Handle an incoming runtime command.
    """

    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state."""
        ...

    @property
    def last_event(self) -> IcoRuntimeEvent | None:
        """Last received runtime event."""

    # ─── Handlers ───

    def on_command(self, command: IcoRuntimeCommand) -> None: ...

    def on_event(self, event: IcoRuntimeEvent) -> None: ...

    # ─── Runtime Endpoint Attachment ───

    def connect_runtime(self, endpoint: IcoOperatorProtocol[Any, Any]) -> None: ...


# ──── Support
class SupportsDownstream(IcoRuntimeProtocol):
    def broadcast_command(self, command: IcoRuntimeCommand) -> None: ...


class SupportsUpstream(IcoRuntimeProtocol):
    def bubble_event(self, event: IcoRuntimeEvent) -> None: ...


class IcoRuntimeOperatorProtocol(
    IcoOperatorProtocol[I, O],
    Protocol[I, O],
    IcoRuntimeProtocol,
):
    """Operator that also supports runtime commands."""

    ...

    # # ─── Command Propagation ───

    # def broadcast_command(self, command: IcoRuntimeCommand) -> None:
    #     """
    #     Broadcast a runtime command to all child operators.

    #     This propagates the command downward through the operator tree,
    #     allowing all children to react accordingly.
    #     """
    #     ...

    # def bubble_command(self, command: IcoRuntimeCommand) -> None:
    #     """
    #     Bubble a runtime command up to the nearest runtime host.

    #     This sends the command upward through the operator tree,
    #     allowing parent operators to handle it.
    #     """
    #     ...

    # # ─── Event Propagation ───

    # def broadcast_event(self, event: IcoRuntimeEvent) -> None:
    #     """
    #     Broadcast a runtime event to all child operators.

    #     This propagates the event downward through the operator tree,
    #     allowing all children to react accordingly.
    #     """
    #     ...

    # def bubble_event(self, event: IcoRuntimeEvent) -> None:
    #     """
    #     Bubble a runtime event up to the nearest runtime host.

    #     This sends the event upward through the operator tree,
    #     allowing parent operators to handle it.
    #     """
    #     ...

    # ─── Runtime Endpoint Attachment ───

    # def attach_runtime(self, contour: IcoOperatorProtocol[None, None]) -> None:
    #     """Attach coontour to enable command propagation."""
    #     ...
