from __future__ import annotations

from collections.abc import Callable
from multiprocessing import get_context
from multiprocessing.context import SpawnContext, SpawnProcess
from typing import Generic, final

from apriori.ico.core.runtime.agents.mp_process.agent import MPProcessAgent
from apriori.ico.core.runtime.channels.mp_queue.channel import MPQueueChannel
from apriori.ico.core.runtime.channels.types import IcoRuntimeChannelRole
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import I, IcoOperatorProtocol, O


@final
class MPProcessAgentHost(
    Generic[I, O],
    IcoRuntimeOperator,
    ProgressMixin,
):
    input_channel: MPQueueChannel[I]
    output_channel: MPQueueChannel[O]
    mp_context: SpawnContext
    flow_factory: Callable[[], IcoOperatorProtocol[I, O]]
    _agent_process: SpawnProcess | None

    def __init__(
        self,
        input_channel: MPQueueChannel[I],
        output_channel: MPQueueChannel[O],
        mp_context: SpawnContext,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name or f"mp_process_agent_host_{id(self)}"
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.mp_context = mp_context
        self.flow_factory = flow_factory
        self._agent_process = None

    def on_command(self, command: IcoRuntimeCommand) -> None:
        super().on_command(command)

        match command:
            case IcoRuntimeCommand.activate:
                self._spawn_agent()

            case IcoRuntimeCommand.deactivate:
                self._shutdown_agent()

    # ─── Agent process management ───

    def _spawn_agent(self) -> None:
        self._agent_process = MPProcessAgent.spawn(
            mp_context=self.mp_context,
            input_channel=self.input_channel,
            output_channel=self.output_channel,
            flow_factory=self.flow_factory,
        )

    def _shutdown_agent(self) -> None:
        if self._agent_process is None:
            return
        try:
            # Gracefully join the worker process
            if self._agent_process.is_alive():
                # Notify agent to shutdown befor closing agent process and channels queues
                self.input_channel.send.send_command(IcoRuntimeCommand.deactivate)

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
    def create(
        cls,
        flow_factory: Callable[[], IcoOperatorProtocol[I, O]],
        name: str | None = None,
    ) -> MPProcessAgentHost[I, O]:
        mp_context = get_context("spawn")
        host_name = name or f"mp_process_agent_host_{id(cls)}"

        input_channel = MPQueueChannel[I](
            IcoRuntimeChannelRole.input,
            mp_context,
            name=f"{host_name}_input_channel",
        )
        output_channel = MPQueueChannel[O](
            IcoRuntimeChannelRole.output,
            mp_context,
            name=f"{host_name}_output_channel",
        )

        host = MPProcessAgentHost(
            mp_context=mp_context,
            flow_factory=flow_factory,
            input_channel=input_channel,
            output_channel=output_channel,
            name=name,
        )
        input_channel.connect_runtime(host)
        return host
