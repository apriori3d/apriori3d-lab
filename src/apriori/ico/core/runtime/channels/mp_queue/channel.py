# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

import queue
from collections.abc import Iterator
from multiprocessing import Queue
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
from apriori.ico.core.runtime.progress import ProgressMixin
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
    send: IcoRuntimeOperatorProtocol[Iterator[I], None]
    receive: IcoRuntimeOperatorProtocol[None, Iterator[I]]

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(
        self,
        *,
        main_queue: ChannelQueue | None = None,
        ack_queue: ChannelQueue | None = None,
    ) -> None:
        IcoRuntimeMixin.__init__(self)
        self._main_queue = main_queue or ChannelQueue()
        self._ack_queue = ack_queue or ChannelQueue()

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
    IcoRuntimeOperator[Iterator[I], None],
    IcoRuntimeOperatorProtocol[Iterator[I], None],
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

    def _send_fn(self, item: Iterator[I]) -> None:
        for input in item:
            self._main_queue.put(InputPayload[I](input=input).wrap())
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
    IcoRuntimeOperator[None, Iterator[I]],
    IcoRuntimeOperatorProtocol[None, Iterator[I]],
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

    def _receive_fn(self, _: None = None) -> Iterator[I]:
        while True:
            try:
                message = self.main_queue.get()

                # Handle progress messages
                if ProgressRelay.handle_message(self.progress, message):
                    continue

                if not isinstance(message, ChannelMessage):
                    raise TypeError(
                        f"Expected ChannelMessage, got {type(message).__name__}"
                    )

                match message.message_type:
                    case ChannelMessageType.input:
                        # Acknowledge then yield input
                        self.ack_queue.put(
                            AcknowledgePayload(message.message_type).wrap()
                        )
                        yield cast(I, message.payload.input)

                    case ChannelMessageType.runtime_command:
                        # Broadcast command then acknowledge
                        command = message.unwrap(RuntimeCommandPayload).command
                        self.broadcast_command(command)
                        self.ack_queue.put(
                            AcknowledgePayload(message.message_type).wrap()
                        )
                        # Stop iteration if command is stop or reset
                        if command in [IcoRuntimeCommand.stop, IcoRuntimeCommand.reset]:
                            raise StopIteration

                    case ChannelMessageType.error:
                        raise RuntimeError(
                            f"Remote error: {cast(ErrorPayload, message.payload).error}"
                        )

                    case _:
                        # Ignore other message types
                        continue

            except StopIteration:
                return  # Exit the generator
            except Exception as e:
                self.ack_queue.put(ErrorPayload(repr(e)).wrap())
                raise
