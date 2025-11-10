from __future__ import annotations

from typing import Generic, Protocol

from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.types import IcoRuntimeProtocol
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class IcoAgentProtocol(
    Protocol[I, O],
    IcoRuntimeProtocol,
):
    input_channel: IcoChannelProtocol[I]
    output_channel: IcoChannelProtocol[O]


class IcoAgent(
    Generic[I, O],
    IcoRuntimeContour,
    IcoAgentProtocol[I, O],
):
    input_channel: IcoChannelProtocol[I]
    output_channel: IcoChannelProtocol[O]

    def __init__(
        self,
        closure: IcoOperatorProtocol[None, None],
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
    ) -> None:
        super().__init__(closure)
        self.input_channel = input_channel
        self.output_channel = output_channel

        input_channel.receive.command_port = self.on_command

    def on_event(self, event: IcoRuntimeEvent) -> None:
        self.output_channel.send.send_event(event)
