from __future__ import annotations

from collections.abc import Callable, Iterator
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic

from apriori.ico.core.runtime.agents.mp_process.agent import MPProcessAgent
from apriori.ico.core.runtime.channel import IcoChannelProtocol
from apriori.ico.core.runtime.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeOperatorProtocol,
)
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class MPProcessAgentLink(
    Generic[I, O],
    IcoRuntimeOperator[Iterator[I], Iterator[O]],
    ProgressMixin,
    IcoRuntimeOperatorProtocol[Iterator[I], Iterator[O]],
):
    # Channels composing this link
    _mp_context: SpawnContext
    _flow_factory: Callable[[], IcoOperatorProtocol[I, O]]
    _input_channel: IcoChannelProtocol[I]
    _output_channel: IcoChannelProtocol[O]
    _agent_process: SpawnProcess | None

    def __init__(
        self,
        mp_context: SpawnContext,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        input_channel: IcoChannelProtocol[I],
        output_channel: IcoChannelProtocol[O],
        name: str | None = None,
    ) -> None:
        super().__init__(
            fn=self._link_fn,
            name=name,
        )
        self._mp_context = mp_context
        self._flow_factory = flow_factory
        self._input_channel = input_channel
        self._output_channel = output_channel
        self._agent_process = None

        # Attach two enpoints for command propagation.
        # Input channel will broadcast commands downstream to the agent and further to the contour.
        # Output channel.receive will handle upsteam commands (e.g. deactivate to close queues).
        self.connect_runtime(input_channel.send)
        self.connect_runtime(output_channel.receive)

    def _link_fn(self, items: Iterator[I]) -> Iterator[O]:
        for item in items:
            self._input_channel.send(item)
            yield self._output_channel.receive(None)

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        match command:
            case IcoRuntimeCommand.activate:
                self._agent_process = self._spawn_agent()

            case IcoRuntimeCommand.deactivate:
                self._shutdown_agent()

    # ─── Agent process management ───

    def _spawn_agent(self) -> SpawnProcess:
        return MPProcessAgent.spawn(
            mp_context=self._mp_context,
            input_channel=self._input_channel,
            output_channel=self._output_channel,
            flow_factory=self._flow_factory,
        )

    def _shutdown_agent(self) -> None:
        if self._agent_process is None:
            return
        try:
            # Gracefully join the worker process
            if self._agent_process.is_alive():
                # Notify agent to shutdown befor closing agent process and channels queues
                self._input_channel.send.broadcast_command(IcoRuntimeCommand.deactivate)

                # Wait for agent process to exit
                self._agent_process.join(timeout=5)

                # Note: Queues will be closed by the channels themselves after command propagation downstream
                print(f"Process Agent {self.name} worker joined.")

                # Check if worker exited properly
                if self._agent_process.exitcode is None:
                    self.progress.print("⚠️ Worker did not exit (possibly stuck)")

                elif self._agent_process.exitcode != 0:
                    self.progress.print(
                        f"⚠️ Process Agent {self.name} worker exited with code {self._agent_process.exitcode}."
                    )
        except Exception as e:
            self.progress.print(f"❌ Error while stopping agent {self.name}: {e}")

        finally:
            if self._agent_process.is_alive():
                self.progress.print(
                    f"⚠️ Process Agent {self.name} did not terminate worker gracefully."
                )
                self._agent_process.terminate()

    # ─── Factory helper ───

    @classmethod
    def create_with_context(
        cls,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        *,
        context: SpawnContext | None = None,
        name: str | None = None,
    ) -> MPProcessAgentLink[I, O]:
        """
        Factory for creating a process link with a shared context.

        Ensures both channels share the same multiprocessing context,
        allowing synchronized data exchange between host and agent.
        """
        ctx = context or get_context("spawn")
        input_channel = MPQueueChannel[I](mp_context=ctx)
        output_channel = MPQueueChannel[O](mp_context=ctx)
        return cls(
            mp_context=ctx,
            flow_factory=flow_factory,
            input_channel=input_channel,
            output_channel=output_channel,
            name=name,
        )
