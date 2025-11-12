from __future__ import annotations

from typing import Protocol

from apriori.ico.core.runtime.types import (
    ConnectedToIcoRuntime,
    IcoRuntimePortProtocol,
    IcoRuntimeProtocol,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class IcoSendEndpointProtocol(
    IcoOperatorProtocol[I, None],
    Protocol[I],
    IcoRuntimePortProtocol,
):
    """Operator responsible for pushing data and runtime events downstream."""

    ...


class IcoReceiveEndpointProtocol(
    IcoOperatorProtocol[None, O],
    Protocol[O],
    ConnectedToIcoRuntime,
):
    """Operator responsible for pulling data and runtime events."""

    ...


class IcoRuntimeChannelProtocol(Protocol[I, O], IcoRuntimeProtocol):
    send: IcoSendEndpointProtocol[I]
    receive: IcoReceiveEndpointProtocol[O]
