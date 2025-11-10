from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from typing_extensions import Self

from apriori.ico.core.runtime.agents.agent import IcoAgent
from apriori.ico.core.runtime.agents.types import IcoAgentProtocol
from apriori.ico.core.runtime.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.runtime.channels.types import IcoRuntimeChannelProtocol
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoStopExecutionSignal
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I, IcoOperatorProtocol, O


@final
class MPProcessAgent(
    Generic[I, O],
    IcoAgent[I, O],
    IcoAgentProtocol[I, O],
    ProgressMixin,
):
    _contour: IcoRuntimeContour

    def __init__(
        self,
        *,
        input_channel: IcoRuntimeChannelProtocol[I],
        output_channel: IcoRuntimeChannelProtocol[O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
    ) -> None:
        flow = flow_factory()
        closure = input_channel.receive | flow | output_channel.send

        super().__init__(
            closure=closure,
            input_channel=input_channel,
            output_channel=output_channel,
        )
        self.name = name or f"MPProcessAgent-{id(self)}"

    def run_loop(self) -> Self:
        """
        Main execution loop of the process agent.

        The agent runs a blocking loop, continuously executing its runtime contour.
        Each iteration processes one input payload received from the input channel
        and produces one output payload via the output channel.

        The loop terminates on runtime commands:
            - stop
            - reset
            - deactivate
        """
        while True:
            try:
                # Execute the contour (receive → flow → send)
                # Blocks internally until new input arrives in the input channel.
                self.run()

            except IcoStopExecutionSignal:
                # Flow has completed naturally via runtime command deactivate
                break

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.output_channel.send.send_event(IcoRuntimeEvent.exception(e))
                break

        return self

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        if command == IcoRuntimeCommand.deactivate:
            # Stop the main execution loop
            raise IcoStopExecutionSignal()

    @staticmethod
    def spawn(
        *,
        mp_context: SpawnContext,
        input_channel: MPQueueChannel[I],
        output_channel: MPQueueChannel[O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> SpawnProcess:
        process = mp_context.Process(
            target=MPProcessAgent._process_fn,
            args=(input_channel, output_channel, flow_factory, name, relay_progress),
        )
        # TODO: relay_progress
        process.start()
        return process

    @staticmethod
    def _process_fn(
        input_channel: IcoRuntimeChannelProtocol[I],
        output_channel: IcoRuntimeChannelProtocol[O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
        relay_progress: bool = True,
    ) -> None:
        agent = MPProcessAgent[I, O](
            input_channel=input_channel,
            output_channel=output_channel,
            flow_factory=flow_factory,
            name=name,
        )
        # Run agent to start receiving and processing commands and items
        agent.run_loop()
