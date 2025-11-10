# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import TYPE_CHECKING, Generic, final

from apriori.ico.core.runtime.channels.channel import IcoRuntimeChannelMixin
from apriori.ico.core.runtime.channels.messages import (
    ChannelMessage,
)
from apriori.ico.core.runtime.channels.mp_queue.receive_endpoint import (
    MPQueueReceiveEndpoint,
)
from apriori.ico.core.runtime.channels.mp_queue.send_endpoint import MPQueueSendEndpoint
from apriori.ico.core.runtime.channels.types import (
    IcoRuntimeChannelRole,
)
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage]
else:
    ChannelQueue = Queue  # noqa: F401


@final
class MPQueueChannel(
    Generic[I],
    IcoRuntimeChannelMixin[I],
):
    send: MPQueueSendEndpoint[I]
    receive: MPQueueReceiveEndpoint[I]
    _mp_context: SpawnContext

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(
        self,
        role: IcoRuntimeChannelRole,
        mp_context: SpawnContext,
        name: str | None = None,
    ) -> None:
        main_queue = mp_context.Queue()
        ack_queue = mp_context.Queue()

        # Define endpoints
        send = MPQueueSendEndpoint[I](
            main_queue=main_queue,
            ack_queue=ack_queue,
            name=f"{name}_send_endpoint" if name else None,
        )

        receive = MPQueueReceiveEndpoint[I](
            main_queue=main_queue,
            ack_queue=ack_queue,
            name=f"{name}_receive_endpoint" if name else None,
        )

        super().__init__(
            role=role,
            send=send,
            receive=receive,
            name=name or "mp_queue_channel",
        )
        self._mp_context = mp_context
        self._main_queue = main_queue
        self._ack_queue = ack_queue

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        # Handle close command
        if command == IcoRuntimeCommand.deactivate:
            self.send.close()
            self.receive.close()
