from __future__ import annotations

from typing import Generic, Protocol

from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.types import (
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, O


class AgentLinkProtocol(
    IcoRuntimeOperatorProtocol[I, O],
    Protocol,
    Generic[I, O],
):
    """
    Runtime operator bridging host and agent contours.

    Purpose:
        Acts as the runtime bridge between two independent contours:
          • Host → Agent (input channel)
          • Agent → Host (output channel)
        Each channel is itself a runtime operator participating in
        the event propagation system.

    ICO form:
        I → O
        send: I → ()
        receive: () → O

    Runtime semantics:
        • on_command(command) — manage lifecycle (activate/deactivate)
        • broadcast_command(command) — propagate downward to channels
        • bubble_command(command) — send upward to host contour
    """

    # Channels composing this link
    input_channel: IcoChannelProtocol[I]
    output_channel: IcoChannelProtocol[O]
