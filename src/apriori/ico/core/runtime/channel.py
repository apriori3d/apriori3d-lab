from typing import Protocol

from apriori.ico.core.runtime.types import (
    IcoRuntimeOperatorProtocol,
    SupportsIcoRuntime,
)
from apriori.ico.core.types import I


class IcoChannelProtocol(Protocol[I], SupportsIcoRuntime):
    """
    Runtime communication channel connecting two ICO contours.

    Send: I → ()
    Receive: () → I

    Responsibilities:
      • Transmit data and runtime events downstream (via send)
      • Receive data and runtime events upstream (via receive)
      • Propagate lifecycle events through both sides of the channel

    Channels do not execute computations directly and are *not operators*.
    They represent runtime transport links between distributed contours
    (e.g. Agent ↔ Worker, Host ↔ Remote Process).

    Examples of concrete implementations:
        • MPQueueChannel  — based on multiprocessing.Queue
        • ZMQChannel      — ZeroMQ socket transport (TBD)
        • TCPChannel      — network socket transport (TBD)
    """

    # --- Data transmission endpoints ---

    send: IcoRuntimeOperatorProtocol[I, None]
    """Operator responsible for pushing data and runtime events downstream."""

    receive: IcoRuntimeOperatorProtocol[None, I]
    """Operator responsible for pulling data and runtime events upstream."""

    def attach_runtime(
        self, contour: IcoRuntimeOperatorProtocol[None, None]
    ) -> None: ...
