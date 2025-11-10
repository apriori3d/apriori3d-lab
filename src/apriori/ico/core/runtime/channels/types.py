from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from typing import Protocol

from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.types import IcoRuntimeCommand, IcoRuntimeProtocol
from apriori.ico.core.types import I, IcoOperatorProtocol


class IcoSendEndpointProtocol(IcoOperatorProtocol[I, None], Protocol[I]):
    """Operator responsible for pushing data and runtime events downstream."""

    def send_command(self, command: IcoRuntimeCommand) -> None: ...
    def send_event(self, event: IcoRuntimeEvent) -> None: ...


class IcoReceiveEndpointProtocol(IcoOperatorProtocol[None, I], Protocol[I]):
    """Operator responsible for pulling data and runtime events."""

    command_port: Callable[[IcoRuntimeCommand], None] | None
    event_port: Callable[[IcoRuntimeEvent], None] | None


class IcoRuntimeChannelRole(Enum):
    input = auto()
    output = auto()
    duplex = auto()


class IcoRuntimeChannelProtocol(Protocol[I], IcoRuntimeProtocol):
    role: IcoRuntimeChannelRole
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[I]
