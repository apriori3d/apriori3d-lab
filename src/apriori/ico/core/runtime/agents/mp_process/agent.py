from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic

from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.contour import IcoRuntimeContour
from apriori.ico.core.runtime.events import IcoRuntimeEvent
from apriori.ico.core.runtime.exceptions import IcoStopExecutionSignal
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class MPProcessAgent(
    Generic[I, O],
    IcoRuntimeOperator[None, None],
    IcoRuntimeOperatorProtocol[None, None],
    ProgressMixin,
):
    # Channels composing this link
    input_channel: IcoChannelProtocol[I]
    output_channel: IcoChannelProtocol[O]
    _flow_factory: Callable[[], IcoOperatorProtocol[I, O]]
    _contour: IcoRuntimeContour

    def __init__(
        self,
        *,
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._agent_fn,
            name=name,
        )
        self.input_channel = input_channel
        self.output_channel = output_channel
        self._flow_factory = flow_factory

        flow = self._flow_factory()
        closure = self.input_channel.receive | flow | self.output_channel.send
        self._contour = IcoRuntimeContour(closure).attach_progress(self.progress)

        # Enable incoming command broadcasting via input channel futher downstream to the contour
        self.input_channel.receive.connect_runtime(self)
        self._contour.connect_runtime(self)

    def _agent_fn(self, _: None) -> None:
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
                self._contour.run()

            except IcoStopExecutionSignal:
                # Flow has completed naturally via runtime command deactivate
                break

            except Exception as e:
                # Report runtime errors downstream to output channel and terminate
                self.output_channel.send.on_event(IcoRuntimeEvent.exception(e))
                break

        # Graceful shutdown: mark the contour as inactive
        self._contour.on_command(IcoRuntimeCommand.deactivate)

    @staticmethod
    def spawn(
        *,
        mp_context: SpawnContext,
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
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
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
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
        agent()
