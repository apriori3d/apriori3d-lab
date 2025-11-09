from __future__ import annotations

from collections.abc import Callable
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic

from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.channels.messages import ErrorPayload
from apriori.ico.core.runtime.contour import IcoRuntimeContour
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
    _contour: IcoRuntimeContour | None

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
        if self._contour is None:
            raise RuntimeError("Agent contour not initialized (activate required)")

        while True:
            try:
                # Execute the contour (receive → flow → send)
                # Blocks internally until new input arrives in the input channel.
                self._contour.run()

                # Check if contour received a termination command
                if self._contour.last_command in (
                    IcoRuntimeCommand.stop,
                    IcoRuntimeCommand.reset,
                    IcoRuntimeCommand.deactivate,
                ):
                    break

            except StopIteration:
                # Flow has completed naturally (no more data)
                break

            except Exception as e:
                # Report runtime errors upstream and terminate
                self.output_channel.send(ErrorPayload(repr(e)).wrap())
                break

        # Graceful shutdown: mark the contour as inactive
        self._contour.on_command(IcoRuntimeCommand.deactivate)

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        match command:
            case IcoRuntimeCommand.activate:
                if self._contour is None:
                    flow = self._flow_factory()
                    closure = (
                        self.input_channel.receive | flow | self.output_channel.send
                    )
                    self._contour = IcoRuntimeContour(closure)
                    self._contour.attach_progress(self.progress)
                    self.input_channel.attach_runtime(self._contour)
                    self._contour.activate()

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
        agent()
