from __future__ import annotations

from collections.abc import Callable
from typing import Any

from apriori.ico.core.agent.process.messages import (
    AcknowledgePayload,
    ErrorPayload,
    InputPayload,
    LifecycleEventPayload,
    MessageType,
    OutputPayload,
    ShutdownPayload,
    WorkerMessage,
)
from apriori.ico.core.dsl.operator import IcoOperator, wrap_operator
from apriori.ico.core.types import I, O


class WorkerProtocol(IcoOperator[I, O]):
    """
    Handles routing of WorkerMessages for a given operator.

    Provides a clean, extensible protocol between Agent and Worker.

    Supported message flow:
        • InputPayload[I]          → OutputPayload[O]
        • LifecycleEventPayload    → AcknowledgePayload
        • ExecutionStatePayload    → emitted during execution
        • ShutdownPayload          → AcknowledgePayload
        • ErrorPayload             ← on fault
    """

    def __init__(self, fn: Callable[[I], O], name: str | None = None) -> None:
        operator = wrap_operator(fn)
        super().__init__(fn=fn, name=name, children=[operator])

        # map MessageType → handler method
        self._handlers: dict[MessageType, Callable[[Any], WorkerMessage[Any]]] = {
            MessageType.input: self._on_input,
            MessageType.lifecycle_event: self._on_lifecycle,
            MessageType.shutdown: self._on_shutdown,
        }

    # ──── Routing entry ────

    def handle(self, msg: WorkerMessage[Any]) -> WorkerMessage[Any]:
        """Main router: dispatch message to handler and return a response message."""
        try:
            handler = self._handlers[msg.message_type]
        except KeyError:
            return WorkerMessage.create(
                ErrorPayload(f"Unsupported message {msg.message_type}")
            )

        try:
            return handler(msg.payload)
        except Exception as e:
            return WorkerMessage.create(ErrorPayload(error=repr(e)))

    # ──── Handlers ────

    def _on_input(self, payload: InputPayload[I]) -> WorkerMessage[Any]:
        """Execute operator and return result."""
        output = self(payload.input)
        return WorkerMessage.create(OutputPayload(output))

    def _on_lifecycle(self, payload: LifecycleEventPayload) -> WorkerMessage[Any]:
        """Apply lifecycle event."""
        self.broadcast_event(payload.event)
        return WorkerMessage.create(AcknowledgePayload(MessageType.lifecycle_event))

    def _on_shutdown(self, _: ShutdownPayload) -> WorkerMessage[Any]:
        """Acknowledge shutdown request."""
        return WorkerMessage.create(AcknowledgePayload(MessageType.shutdown))
