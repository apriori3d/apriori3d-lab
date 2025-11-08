from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeVar,
    final,
)

from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I

# ──────────────────────────────────────────────────────────────
# Channel message types
# ──────────────────────────────────────────────────────────────


class ChannelMessageType(Enum):
    """Classifies all messages exchanged between runtime endpoints."""

    input = auto()  # Data payload (I)
    runtime_command = auto()  # Runtime control (activate/reset/stop)
    acknowledge = auto()  # Confirmation of delivery
    error = auto()  # Error reporting
    system = auto()  # Optional system-level event (metrics, heartbeat)


# ──────────────────────────────────────────────────────────────
# Decorator for assigning message types
# ──────────────────────────────────────────────────────────────


def message(t: ChannelMessageType) -> Callable[[type[Any]], type[Any]]:
    """Decorator that assigns `__message_type__` to a payload class."""

    def decorator(cls: type[Any]) -> type[Any]:
        cls.__message_type__ = t
        return cls

    return decorator


# ──────────────────────────────────────────────────────────────
# Base payload class
# ──────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChannelMessagePayload:
    """
    Base class for all channel payloads.
    Subclasses must be annotated with @message_type.
    """

    __message_type__: ClassVar[ChannelMessageType]

    @property
    def message_type(self) -> ChannelMessageType:
        return self.__message_type__

    @classmethod
    def get_message_type(cls) -> ChannelMessageType:
        return cls.__message_type__

    def wrap(self) -> ChannelMessage[Any]:
        """Create a ChannelMessage wrapping this payload instance."""
        return ChannelMessage(self.message_type, self)


# ──────────────────────────────────────────────────────────────
# Payloads
# ──────────────────────────────────────────────────────────────


@message(ChannelMessageType.runtime_command)
@dataclass(slots=True)
class RuntimeCommandPayload(ChannelMessagePayload):
    command: IcoRuntimeCommand


@message(ChannelMessageType.input)
@dataclass(slots=True)
class InputPayload(Generic[I], ChannelMessagePayload):
    input: I


@message(ChannelMessageType.acknowledge)
@dataclass(slots=True)
class AcknowledgePayload(ChannelMessagePayload):
    ack_message_type: ChannelMessageType


@message(ChannelMessageType.error)
@dataclass(slots=True)
class ErrorPayload(ChannelMessagePayload):
    error: str


@message(ChannelMessageType.system)
@dataclass(slots=True)
class SystemPayload(ChannelMessagePayload):
    """Optional generic system event."""

    data: dict[str, Any]


# ──────────────────────────────────────────────────────────────
# Message wrapper
# ──────────────────────────────────────────────────────────────

PayloadT = TypeVar("PayloadT", bound=ChannelMessagePayload)


@final
@dataclass(slots=True)
class ChannelMessage(Generic[PayloadT]):
    """
    A unified message envelope exchanged between two endpoints.

    Each message carries:
      • message_type — tag of ChannelMessageType
      • payload      — typed payload data
    """

    message_type: ChannelMessageType
    payload: PayloadT

    @staticmethod
    def create(payload: PayloadT) -> ChannelMessage[PayloadT]:
        """Wrap a payload into a typed message."""
        return ChannelMessage(message_type=payload.message_type, payload=payload)

    def unwrap(self, expected_type: type[PayloadT]) -> PayloadT:
        """Type-safe access to payload."""
        if self.message_type != expected_type.get_message_type():
            raise TypeError(
                f"Expected {expected_type.get_message_type()}, got {self.message_type}"
            )
        return self.payload
