from __future__ import annotations

from typing import Generic

from apriori.ico.core.runtime.channels.types import (
    IcoReceiveEndpointProtocol,
    IcoRuntimeChannelProtocol,
    IcoSendEndpointProtocol,
)
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommandType, IcoRuntimeFlowProtocol
from apriori.ico.core.types import I, O


class IcoRuntimeChannelMixin(
    Generic[I, O],
    IcoRuntimeOperator,
    IcoRuntimeChannelProtocol[I, O],
):
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[O]

    def __init__(
        self,
        send: IcoSendEndpointProtocol[I],
        receive: IcoReceiveEndpointProtocol[O],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.send = send
        self.receive = receive
        self.name = name or f"MPQueueChannel-{id(self)}"

        # Connect remote runtime via endpoint runtime port
        self.receive.runtime = self

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        super().on_command(command)

        # Send command to runtime port of send endpoint
        self.send.on_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        super().on_event(event)

        # Send event to runtime port of send endpoint
        self.send.on_event(event)


class IcoReceiveEndpointMixin(
    Generic[O],
    IcoReceiveEndpointProtocol[O],
):
    def on_command(self, command: IcoRuntimeCommandType) -> None:
        if self.runtime:
            self.runtime.broadcast_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        if self.runtime:
            self.runtime.bubble_event(event)


class IcoSendEndpointMixin(
    Generic[I],
    IcoSendEndpointProtocol[I],
    IcoRuntimeFlowProtocol,
): ...
