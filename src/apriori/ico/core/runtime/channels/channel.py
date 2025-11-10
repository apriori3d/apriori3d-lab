from __future__ import annotations

from typing import Generic

from apriori.ico.core.runtime.channels.types import (
    IcoReceiveEndpointProtocol,
    IcoRuntimeChannelProtocol,
    IcoRuntimeChannelRole,
    IcoSendEndpointProtocol,
)
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I


class IcoRuntimeChannelMixin(
    Generic[I], IcoRuntimeOperator, IcoRuntimeChannelProtocol[I]
):
    role: IcoRuntimeChannelRole
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[I]

    def __init__(
        self,
        role: IcoRuntimeChannelRole,
        send: IcoSendEndpointProtocol[I],
        receive: IcoReceiveEndpointProtocol[I],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.role = role
        self.send = send
        self.receive = receive
        self.name = name or f"MPQueueChannel-{id(self)}"

        # Set up receive semantics based on channel role
        if role in [IcoRuntimeChannelRole.input, IcoRuntimeChannelRole.duplex]:
            self.receive.command_port = self.on_command

        if role in [IcoRuntimeChannelRole.output, IcoRuntimeChannelRole.duplex]:
            self.receive.event_port = self.on_event

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        # Execute send semantics based on channel role
        if self.role in [IcoRuntimeChannelRole.input, IcoRuntimeChannelRole.duplex]:
            self.send.send_command(command)

    def on_event(self, event: IcoRuntimeEvent) -> None:
        super().on_event(event)

        # Execute send semantics based on channel role
        if self.role in [IcoRuntimeChannelRole.output, IcoRuntimeChannelRole.duplex]:
            self.send.send_event(event)
