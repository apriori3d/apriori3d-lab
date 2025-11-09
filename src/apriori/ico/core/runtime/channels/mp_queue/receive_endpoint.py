# apriori/ico/core/runtime/mp_queue_channel.py
from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Queue
from typing import TYPE_CHECKING, Generic, cast, final

from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.ico.core.runtime.channels.messages import (
    AcknowledgePayload,
    ChannelMessage,
    ChannelMessageType,
    InputPayload,
    RuntimeCommandPayload,
    RuntimeEventPayload,
)
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoRuntimeError, IcoStopExecutionSignal
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
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
class MPQueueReceiveEndpoint(
    Generic[I],
    IcoRuntimeOperator[None, I],
    IcoRuntimeOperatorProtocol[None, I],
    ProgressMixin,
):
    """
    ReceiveEndpoint for multiprocessing Queue-based communication.

    Responsibilities:
      • Blocking wait for incoming messages
      • Dispatch by message type
      • Handle runtime commands and events
      • Acknowledge receipt to the sender
    """

    _main_queue: ChannelQueue
    _ack_queue: ChannelQueue

    def __init__(self, main_queue: ChannelQueue, ack_queue: ChannelQueue) -> None:
        super().__init__(
            fn=self._receive_fn,
            name="mp_queue_receive",
        )
        self._main_queue = main_queue
        self._ack_queue = ack_queue

    # ────────────────────────────────
    # Main receive loop
    # ────────────────────────────────

    def _receive_fn(self, _: None = None) -> I:
        """Blocking receive for a single item.

        Waits for a message in the queue, routes it to a handler,
        and returns the next input item to the runtime contour flow.
        """
        while True:
            try:
                message = self._main_queue.get()

                # Handle progress messages transparently
                if ProgressRelay.handle_message(self.progress, message):
                    continue

                if not isinstance(message, ChannelMessage):
                    raise TypeError(
                        f"Expected ChannelMessage, got {type(message).__name__}"
                    )

                # Dispatch based on message type
                handler = self._dispatch_table().get(message.message_type)

                if handler is None:
                    self._log_unknown(message)
                    continue

                result = handler(message)
                if result is not None:
                    return result

            except (IcoStopExecutionSignal, IcoRuntimeError):
                # Propagate runtime signals upward
                raise
            except Exception as e:
                # Wrap and send back fault event
                fault_event = IcoRuntimeEvent.exception(e)
                self._ack_queue.put(RuntimeEventPayload(fault_event).wrap())
                raise

    # ────────────────────────────────
    # Dispatch table
    # ────────────────────────────────

    def _dispatch_table(
        self,
    ) -> dict[ChannelMessageType, Callable[[ChannelMessage], None | I]]:
        """Return mapping of message types to handler methods."""
        return {
            ChannelMessageType.input: self._handle_input,
            ChannelMessageType.runtime_command: self._handle_command,
            ChannelMessageType.runtime_event: self._handle_event,
        }

    # ────────────────────────────────
    # Handlers
    # ────────────────────────────────

    def _handle_input(self, message: ChannelMessage) -> I:
        """Handle normal data input message."""
        self._ack(ChannelMessageType.input)
        input_payload = cast(InputPayload, message.unwrap())
        return cast(I, input_payload.input)

    def _handle_command(self, message: ChannelMessage) -> None:
        """Handle runtime command (activate, reset, stop, etc.)."""
        payload = cast(RuntimeCommandPayload, message.unwrap())
        command = payload.command

        # Propagate downstream before acknowledging
        self.broadcast_command(command)
        self._ack(ChannelMessageType.runtime_command)

        # Stop or deactivate ends receive loop
        if command in {IcoRuntimeCommand.deactivate, IcoRuntimeCommand.stop}:
            raise IcoStopExecutionSignal

    def _handle_event(self, message: ChannelMessage) -> None:
        """Handle runtime events (faults, metrics, progress, etc.)."""
        payload = cast(RuntimeEventPayload, message.unwrap())
        event = payload.event
        self._ack(ChannelMessageType.runtime_event)

        if event.is_fault:
            event.raise_if_fault()
        else:
            self.broadcast_event(event)

    # ────────────────────────────────
    # Runtime command handling
    # ────────────────────────────────

    def on_command(self, command: IcoRuntimeCommand) -> None:
        """Handle deactivate to close queues."""
        super().on_command(command)

        if command is IcoRuntimeCommand.deactivate:
            self._main_queue.close()
            self._main_queue.join_thread()
            self._ack_queue.close()
            self._ack_queue.join_thread()

    # ────────────────────────────────
    # Utilities
    # ────────────────────────────────

    def _ack(self, msg_type: ChannelMessageType) -> None:
        """Send acknowledgment to the sender."""
        self._ack_queue.put(AcknowledgePayload(msg_type).wrap())

    def _log_unknown(self, message: ChannelMessage) -> None:
        self.progress.log(
            f"{self.name}: Ignoring unknown message type {message.message_type}"
        )
