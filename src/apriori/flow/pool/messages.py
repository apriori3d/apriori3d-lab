from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    final,
    overload,
    runtime_checkable,
)

import torch

from apriori.flow.core.pipeline.types import (
    PipelineControlMessage,
    PipelineInputType,
    PipelineOutputType,
    PipelineResult,
)

# ──── Literal-based message type definitions ────


class AgentMessageType(Enum):
    lifecycle = auto()
    lifecycle_response = auto()
    restore_state = auto()
    restore_state_response = auto()
    run = auto()
    run_response = auto()
    run_response_consumed = auto()
    fault = auto()


# ──── Payload decorator ────


def message_payload(message: AgentMessageType):
    def wrapper(cls: type[PayloadT]) -> type[PayloadT]:
        cls.__agent_message_type__ = message
        return cls

    return wrapper


@runtime_checkable
class SupportsAgentMessageType(Protocol):
    __agent_message_type__: AgentMessageType


# ──── Payload definitions ────


class LifecyclePhasesType(Enum):
    prepare = auto()
    on_cycle_start = auto()
    on_cycle_end = auto()
    cleanup = auto()


@final
@dataclass()
@message_payload(AgentMessageType.lifecycle)
class LifecyclePayload:
    phase: LifecyclePhasesType


@final
@dataclass()
@message_payload(AgentMessageType.lifecycle_response)
class LifecycleResponsePayload:
    phase: LifecyclePhasesType


@final
@dataclass()
@message_payload(AgentMessageType.restore_state)
class RestoreStatePayload:
    state: dict[str, torch.Tensor]


@final
@dataclass()
@message_payload(AgentMessageType.restore_state_response)
class RestoreStateResponsePayload:
    state: dict[str, torch.Tensor]


@final
@dataclass()
@message_payload(AgentMessageType.run)
class RunPayload(Generic[PipelineInputType]):
    item: PipelineInputType


@final
@dataclass()
@message_payload(AgentMessageType.run_response)
class RunResponsePayload(Generic[PipelineOutputType]):
    result: PipelineResult[PipelineOutputType]


@final
@dataclass()
@message_payload(AgentMessageType.run_response_consumed)
class ResultConsumedAckPayload(Generic[PipelineOutputType]):
    control: PipelineControlMessage


@final
@dataclass()
@message_payload(AgentMessageType.fault)
class AgentFaultPayload:
    error: str


# ──── Generic message container ────

PayloadT = TypeVar("PayloadT")


@final
@dataclass()
class AgentMessage(Generic[PayloadT]):
    type: AgentMessageType
    agent_id: int
    payload: PayloadT


# ──── Message factory methods with type hints ────


@overload
def create_message(
    agent_id: int, payload: LifecyclePayload
) -> AgentMessage[LifecyclePayload]: ...


@overload
def create_message(
    agent_id: int, payload: LifecycleResponsePayload
) -> AgentMessage[LifecycleResponsePayload]: ...


@overload
def create_message(
    agent_id: int, payload: RestoreStatePayload
) -> AgentMessage[RestoreStatePayload]: ...


@overload
def create_message(
    agent_id: int, payload: RestoreStateResponsePayload
) -> AgentMessage[RestoreStateResponsePayload]: ...


@overload
def create_message(agent_id: int, payload: RunPayload) -> AgentMessage[RunPayload]: ...


@overload
def create_message(
    agent_id: int, payload: RunResponsePayload
) -> AgentMessage[RunResponsePayload]: ...


@overload
def create_message(
    agent_id: int, payload: AgentFaultPayload
) -> AgentMessage[AgentFaultPayload]: ...


def create_message(agent_id: int, payload: Any) -> AgentMessage:
    if not isinstance(payload, SupportsAgentMessageType):
        raise ValueError(
            f"Payload {type(payload)} does not support agent message type annotation"
        )
    return AgentMessage(
        type=payload.__agent_message_type__,
        agent_id=agent_id,
        payload=payload,
    )
