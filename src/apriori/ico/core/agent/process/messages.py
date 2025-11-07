from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    ClassVar,
    Generic,
    TypeVar,
    final,
)

from apriori.ico.core.runtime.execution import IcoExecutionState
from apriori.ico.core.runtime.lifecycle import IcoLifecycleEvent
from apriori.ico.core.types import I, O


class MessageType(Enum):
    # ─── Message from Host to Worker ───
    lifecycle_event = auto()  # Lifecycle event for broadcasting
    input = auto()  # Input for hosted operator
    shutdown = auto()  # Shutdown signal for a worker

    # ─── Message from Worker to Host ───
    acknowledge = auto()  # Acknowledgment of input messages
    execution_event = auto()  # Execution state update
    output = auto()  # Output from hosted operator
    fault = auto()  # Error reporting


# ──── Payload definitions ────


@dataclass(slots=True)
class MessagePayload:
    __message_type__: ClassVar[MessageType]

    def __init__(self) -> None:
        if not hasattr(self, "__message_type__"):
            raise NotImplementedError(
                "Subclasses must define __message_type__ class variable."
            )

    @property
    def message_type(self) -> MessageType:
        return self.__message_type__

    @classmethod
    def get_message_type(cls) -> MessageType:
        return cls.__message_type__


@final
@dataclass(slots=True)
class LifecycleEventPayload(MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.lifecycle_event
    event: IcoLifecycleEvent


@final
@dataclass(slots=True)
class InputPayload(Generic[I], MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.input
    input: I


@final
@dataclass(slots=True)
class ShutdownPayload(MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.shutdown


@final
@dataclass(slots=True)
class AcknowledgePayload(MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.acknowledge
    ack_message_type: MessageType


@final
@dataclass(slots=True)
class ExecutionStatePayload(MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.execution_event
    state: IcoExecutionState


@final
@dataclass(slots=True)
class OutputPayload(Generic[O], MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.output
    output: O


@final
@dataclass(slots=True)
class ErrorPayload(MessagePayload):
    __message_type__: ClassVar[MessageType] = MessageType.fault
    error: str


# ──── Worker Message Definition ────

PayloadT = TypeVar("PayloadT", bound=MessagePayload)


@final
@dataclass(slots=True)
class WorkerMessage(Generic[PayloadT]):
    message_type: MessageType
    payload: PayloadT

    @staticmethod
    def create(payload: PayloadT) -> WorkerMessage[PayloadT]:
        return WorkerMessage(message_type=payload.message_type, payload=payload)
