from __future__ import annotations

from collections.abc import Iterator
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from typing_extensions import Self

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.progress.types import ProgressProtocol
from apriori.ico.core.types import IcoOperatorProtocol

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


# ──── Protocols for runtime operators ────


class IcoRuntimeStateProtocol(Protocol):
    # ─── Properties ───

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state."""
        ...

    @property
    def last_event(self) -> IcoRuntimeEvent | None:
        """Last received runtime event."""


@runtime_checkable
class IcoRuntimePortProtocol(Protocol):
    """Operator responsible for pushing data and runtime events downstream."""

    def on_command(self, command: IcoRuntimeCommand) -> None: ...

    def on_event(self, event: IcoRuntimeEvent) -> None: ...


class IcoRuntimeHierarchyProtocol(Protocol):
    runtime_children: list[IcoRuntimeProtocol]
    runtime_parent: IcoRuntimeProtocol | None

    # ─── Runtime Discovery and Connection ───

    def discover_runtime(
        self, closure: IcoOperatorProtocol[None, None]
    ) -> Iterator[IcoRuntimeProtocol]: ...

    def connect_runtime(self, runtime: IcoRuntimeProtocol) -> None: ...

    def disconnect_runtime(self, runtime: IcoRuntimeProtocol) -> None: ...

    # ─── Command & Event Propagation ───

    def broadcast_command(self, command: IcoRuntimeCommand) -> None: ...

    def bubble_event(self, event: IcoRuntimeEvent) -> None: ...

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self: ...


class IcoRuntimeLifecycleProtocol(Protocol):
    def run(self) -> Self:
        """Execute the contour by calling itself."""
        ...

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        ...

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        ...

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        ...

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        ...

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        ...

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        ...


@runtime_checkable
class IcoRuntimeProtocol(
    IcoRuntimeStateProtocol,
    IcoRuntimeHierarchyProtocol,
    IcoRuntimeLifecycleProtocol,
    IcoRuntimePortProtocol,
    IcoOperatorProtocol[None, None],
    Protocol,
): ...


@runtime_checkable
class ConnectedToIcoRuntime(Protocol):
    @property
    def runtime(self) -> IcoRuntimeProtocol:
        """Get the associated runtime protocol."""
        ...
