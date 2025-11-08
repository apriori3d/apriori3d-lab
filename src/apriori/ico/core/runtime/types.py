from __future__ import annotations

from enum import Enum, auto
from typing import Protocol, runtime_checkable

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
    # fault = auto()


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


# ──── Protocol for runtime operators ────


@runtime_checkable
class SupportsIcoRuntime(Protocol):
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

    @property
    def state(self) -> IcoRuntimeState:
        """Current runtime state."""
        ...

    @property
    def last_command(self) -> IcoRuntimeCommand | None:
        """Last received runtime command."""

    def on_command(self, command: IcoRuntimeCommand) -> None: ...


class IcoRuntimeOperatorProtocol(
    IcoOperatorProtocol[I, O],
    Protocol[I, O],
    SupportsIcoRuntime,
):
    """Operator that also supports runtime commands."""

    ...
