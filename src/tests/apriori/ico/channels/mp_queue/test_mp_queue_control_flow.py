# ───────────────────────────────────────────────
#  Test: Runtime command propagation (roundtrip)
# ───────────────────────────────────────────────
import time
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess

from apriori.ico.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommandType,
    IcoRuntimeEventType,
)


class ControlFlowTestingRuntime(IcoRuntimeOperator):
    commands_received: list[IcoRuntimeCommandType]
    events_received: list[IcoRuntimeEventType]

    def __init__(self):
        super().__init__()
        self.commands_received = []
        self.events_received = []

    def on_command(self, command: IcoRuntimeCommandType) -> None:
        # Echo command back through send endpoint
        self.commands_received.append(command)

    def on_event(self, event: IcoRuntimeEventType) -> None:
        self.events_received.append(event)


def recording_agent(channel: MPQueueChannel[str, str]) -> None:
    """Agent process that records received commands and sends back acknowledgements."""

    # Create runtime that records commands and events and sends them back
    runtime = ControlFlowTestingRuntime()
    # Connect runtime to channel receive endpoint to receive bubbled-up events
    channel.receive.runtime = runtime

    def reporting_fn(item: str) -> str:
        # Send heartbeat event to host runtime
        print("Agent sending heartbeat event")
        runtime.on_event(IcoRuntimeEvent.heartbeat())
        if item == "report":
            print("Agent reporting recorded commands and events")
            # Return recorded commands and events
            return {
                "commands": runtime.commands_received,
                "events": runtime.events_received,
            }
        # Raise error for unknown items to test exception propagation
        print(f"Agent received unknown item: {item}")
        raise ValueError(f"Unknown item: {item}")

    # Create agent flow to record and report commands/events
    reporting_operator = IcoOperator[str, str](reporting_fn)
    closure = channel.receive | reporting_operator | channel.send

    try:
        # Execute the agent closure
        closure()
    except Exception as e:
        # Send exception event back to host runtime
        channel.send.on_event(IcoRuntimeEvent.exception(e))


def test_runtime_flow_propagation() -> None:
    """Ensure runtime commands travel to agent and responses return back."""

    # Create communication channel between host and agent
    ctx = get_context("spawn")
    channel = MPQueueChannel[str, str](ctx)

    # Create host runtime to aggregate bubble-up events via channel
    host_runtime = ControlFlowTestingRuntime()
    host_runtime.connect_runtime(channel)  # Setup link for commands/events flow

    # Strat agent process
    process: SpawnProcess = ctx.Process(
        target=recording_agent, args=(channel,), daemon=True
    )
    process.start()
    time.sleep(0.05)

    try:
        # Send commands to remote agent
        host_runtime.activate().pause().resume()

        # Send 'report' to agent to get back recorded commands and events
        flow = channel.send | channel.receive
        agent_runtime_recorting = flow("report")
        print(agent_runtime_recorting)
        # Проверяем, что все команды отразились обратно
        assert (
            agent_runtime_recorting["revceived_commands"]
            == host_runtime.commands_received
        )

    finally:
        process.terminate()
        process.join(timeout=0.5)

    assert not process.is_alive(), "Agent process did not exit cleanly"


if __name__ == "__main__":
    test_runtime_flow_propagation()
