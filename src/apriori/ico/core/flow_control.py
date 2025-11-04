# ──── Control messages for pipeline execution flow ────


from enum import Enum, auto


class FlowControlMessage(Enum):
    Continue = auto()
    StopOperator = auto()
    StopRunner = auto()
    Pause = auto()
    Skip = auto()


class HasFlowControl:
    flow_control_msg: FlowControlMessage


class FlowControlMixin:
    flow_control_msg: FlowControlMessage = FlowControlMessage.Continue

    def set_flow_control(self, msg: FlowControlMessage) -> None:
        self.flow_control_msg = msg

    def reset_flow_control(self) -> None:
        self.flow_control_msg = FlowControlMessage.Continue
