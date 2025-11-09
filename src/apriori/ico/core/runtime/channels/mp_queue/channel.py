# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

import queue
from multiprocessing import Queue
from multiprocessing.context import SpawnContext
from typing import TYPE_CHECKING, Any, Generic, cast, final

from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessageType,
    ErrorPayload,
    InputPayload,
    PayloadT,
    RuntimeCommandPayload,
)
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.runtime_mixin import IcoRuntimeMixin
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType

if TYPE_CHECKING:
    ChannelQueue = Queue[ChannelMessage[Any]]
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

        # define endpoints
        self.send = MPQueueSendOperator[I](
            main_queue=self._main_queue,
            ack_queue=self._ack_queue,
        )

        self.receive = MPQueueReceiveOperator[I](
            main_queue=self._main_queue,
            ack_queue=self._ack_queue,
        )

    def attach_runtime(self, contour: IcoOperatorProtocol[None, None]) -> None:
        if contour not in self.receive.children:
            self.receive.children.append(contour)
            self.receive.parent = contour

    def on_command(self, command: IcoRuntimeCommand) -> None:
        self.send.on_command(command)

    @property
    def main_queue(self) -> ChannelQueue:
        return self._main_queue

    @property
    def ack_queue(self) -> ChannelQueue:
        return self._ack_queue


class MPQueueSendOperator(
    Generic[I],
    IcoRuntimeOperator[I, None],
    IcoRuntimeOperatorProtocol[I, None],
):
    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(self, main_queue: ChannelQueue, ack_queue: ChannelQueue) -> None:
        IcoRuntimeMixin.__init__(self)
        super().__init__(
            fn=self._send_fn,
            name="mp_queue_send",
            node_type=NodeType.operator,
        )
        self._main_queue = main_queue
        self._ack_queue = ack_queue

    def _send_fn(self, item: I) -> None:
        self._main_queue.put(InputPayload[I](item).wrap())
        self._wait_for_ack(InputPayload)

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        self._main_queue.put(RuntimeCommandPayload(command).wrap())
        self._wait_for_ack(RuntimeCommandPayload)

    def _wait_for_ack(self, payload_type: type[PayloadT], timeout: int = 5) -> None:
        while True:
            try:
                message = self._ack_queue.get(timeout=timeout)
            except queue.Empty as e:
                raise TimeoutError(
                    f"No ACK received for {payload_type.__name__} within {timeout}s"
                ) from e

            if not isinstance(message, ChannelMessage):
                raise TypeError(f"Expected ChannelMessage, got {type(message)}")

            match message.message_type:
                case ChannelMessageType.acknowledge:
                    ack_payload = message.unwrap(AcknowledgePayload)

                    if ack_payload.ack_message_type != payload_type.get_message_type():
                        raise RuntimeError(
                            f"Unexpected acknowledgment: expected {payload_type.get_message_type()}, got {ack_payload.ack_message_type}"
                        )
                    return

                case ChannelMessageType.error:
                    error_payload = message.unwrap(ErrorPayload)
                    raise RuntimeError(f"Remote error: {error_payload.error}")

                case _:
                    raise RuntimeError(
                        f"Unexpected message type in ack queue: {message.message_type}"
                    )


class MPQueueReceiveOperator(
    Generic[I],
    IcoRuntimeOperator[None, I],
    IcoRuntimeOperatorProtocol[None, I],
    ProgressMixin,
):
    main_queue: ChannelQueue
    ack_queue: ChannelQueue

    def __init__(self, main_queue: ChannelQueue, ack_queue: ChannelQueue) -> None:
        super().__init__(
            fn=self._receive_fn,
            name="mp_queue_receive",
            node_type=NodeType.operator,
        )
        self.main_queue = main_queue
        self.ack_queue = ack_queue

    def _receive_fn(self, _: None = None) -> I:
        """Blocking receive for a single item.

        Waits until an input payload arrives, filtering out
        runtime and progress messages. Returns the next data item.
        """
        while True:
            try:
                message = self.main_queue.get()

                # ─── Handle progress relay ───
                if ProgressRelay.handle_message(self.progress, message):
                    continue

                # ─── Validate message ───
                if not isinstance(message, ChannelMessage):
                    raise TypeError(
                        f"Expected ChannelMessage, got {type(message).__name__}"
                    )

                # ─── Handle message types ───
                match message.message_type:
                    # ─── Input item ───
                    case ChannelMessageType.input:
                        self.ack_queue.put(
                            AcknowledgePayload(message.message_type).wrap()
                        )
                        return cast(I, message.payload.input)

                    # ─── Runtime command ───
                    case ChannelMessageType.runtime_command:
                        command = message.unwrap(RuntimeCommandPayload).command
                        self.broadcast_command(command)
                        self.ack_queue.put(
                            AcknowledgePayload(message.message_type).wrap()
                        )

                        # Stop receiving if instructed
                        if command in (
                            IcoRuntimeCommand.stop,
                            IcoRuntimeCommand.reset,
                            IcoRuntimeCommand.deactivate,
                        ):
                            raise RuntimeError("Receiving halted by runtime command")

                    # ─── Remote error ───
                    case ChannelMessageType.error:
                        error = message.unwrap(ErrorPayload).error
                        raise RuntimeError(f"Remote error: {error}")

                    # ─── Ignore unknown ───
                    case _:
                        continue

            except Exception as e:
                # Report local failure to the peer
                self.ack_queue.put(ErrorPayload(repr(e)).wrap())
                raise
