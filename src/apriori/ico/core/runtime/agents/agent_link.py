from __future__ import annotations

from typing import Generic

from apriori.ico.core.runtime.agents.types import IcoAgentLinkProtocol
from apriori.ico.core.runtime.channels.types import IcoChannelProtocol
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I, O


class IcoAgentLinkMixin(
    Generic[I, O],
    IcoRuntimeOperator,
    IcoAgentLinkProtocol[I, O],
):
    input_channel: IcoChannelProtocol[I]
    output_channel: IcoChannelProtocol[O]

    def __init__(
        self,
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
    ) -> None:
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel

        output_channel.receive.event_port = self.on_event

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        self.input_channel.send.send_command(command)

        if command == IcoRuntimeCommand.deactivate:
            self.input_channel.send.close()
            self.output_channel.receive.close()
