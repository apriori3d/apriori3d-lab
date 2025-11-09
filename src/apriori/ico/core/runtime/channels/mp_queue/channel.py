# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import TYPE_CHECKING, Generic, final

from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.channels.messages import (
    ChannelMessage,
)
from apriori.ico.core.runtime.channels.mp_queue.receive_endpoint import (
    MPQueueReceiveEndpoint,
)
from apriori.ico.core.runtime.channels.mp_queue.send_endpoint import MPQueueSendEndpoint
from apriori.ico.core.runtime.runtime_mixin import IcoRuntimeMixin
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage]
else:
    ChannelQueue = Queue  # noqa: F401


@final
class MPQueueChannel(
    Generic[I],
    IcoRuntimeMixin,
    IcoChannelProtocol[I],
):
    send: IcoRuntimeOperatorProtocol[I, None]
    receive: IcoRuntimeOperatorProtocol[None, I]

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(self, *, mp_context: SpawnContext) -> None:
        IcoRuntimeMixin.__init__(self)

        self._main_queue = mp_context.Queue()
        self._ack_queue = mp_context.Queue()

        # Define endpoints
        self.send = MPQueueSendEndpoint[I](
            main_queue=self._main_queue,
            ack_queue=self._ack_queue,
        )

        self.receive = MPQueueReceiveEndpoint[I](
            main_queue=self._main_queue,
            ack_queue=self._ack_queue,
        )

    def on_command(self, command: IcoRuntimeCommand) -> None:
        # Send command downstream via send endpoint
        self.send.on_command(command)

    @property
    def main_queue(self) -> ChannelQueue:
        return self._main_queue

    @property
    def ack_queue(self) -> ChannelQueue:
        return self._ack_queue
