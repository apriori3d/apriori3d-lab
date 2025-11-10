from __future__ import annotations

from typing import Protocol

from apriori.ico.core.runtime.channels.types import IcoRuntimeChannelProtocol
from apriori.ico.core.runtime.types import IcoRuntimeProtocol
from apriori.ico.core.types import I, O


class IcoAgentProtocol(
    Protocol[I, O],
    IcoRuntimeProtocol,
):
    input_channel: IcoRuntimeChannelProtocol[I]
    output_channel: IcoRuntimeChannelProtocol[O]


class IcoAgentLinkProtocol(
    Protocol[I, O],
    IcoRuntimeProtocol,
):
    input_channel: IcoRuntimeChannelProtocol[I]
    output_channel: IcoRuntimeChannelProtocol[O]
