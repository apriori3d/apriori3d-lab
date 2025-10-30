from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    final,
    overload,
    runtime_checkable,
)

import torch

from apriori.flow.pipeline.types import (
    PipelineResult,
)
from apriori.flow.runner.types import (
    PipelineInputType,
    PipelineOutputType,
)

# ──── Literal-based message type definitions ────

AgentRequestType = Literal[
    "lifecycle",
    "restore_state",
    "run",
]
AgentResponseType = Literal[
    "lifecycle_response",
    "restore_state_response",
    "run_response",
]

AgentEventType = Literal["fault",]

AgentMessageType = AgentRequestType | AgentResponseType | AgentEventType

# ──── Payload decorator ────


def message_payload(message: AgentMessageType):
    def wrapper(cls: type[PayloadT]) -> type[PayloadT]:
        cls.__agent_message_type__ = message
        return cls

    return wrapper


# Mapping of payload classes to message types


@runtime_checkable
class SupportsAgentMessageType(Protocol):
    __agent_message_type__: AgentMessageType


# ──── Payload definitions ────


PhasesType = Literal["prepare", "on_cycle_start", "on_cycle_end", "cleanup"]


@final
@dataclass()
@message_payload("lifecycle")
class LifecyclePayload:
    phase: PhasesType


@final
@dataclass()
@message_payload("lifecycle_response")
class LifecycleResponsePayload:
    phase: PhasesType


@final
@dataclass()
@message_payload("restore_state")
class RestoreStatePayload:
    state: dict[str, torch.Tensor]


@final
@dataclass()
@message_payload("restore_state_response")
class RestoreStateResponsePayload:
    state: dict[str, torch.Tensor]


@final
@dataclass()
@message_payload("run")
class RunPayload(Generic[PipelineInputType]):
    items: list[PipelineInputType]


@final
@dataclass()
@message_payload("run_response")
class RunResponsePayload(Generic[PipelineOutputType]):
    result: list[PipelineResult[PipelineOutputType]]


@final
@dataclass()
@message_payload("fault")
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


# ──── Factory methods with type hints ────


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
