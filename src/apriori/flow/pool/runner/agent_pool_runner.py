import queue
from collections import OrderedDict
from collections.abc import Callable, Iterator
from itertools import islice
from typing import Any, TypeAlias, final

from torch.multiprocessing import Queue

from apriori.flow.core.executor.agent.agent_link import ExecutorAgentLink
from apriori.flow.core.executor.agent.messages import (
    AgentFaultPayload,
    AgentMessage,
    AgentMessageType,
    LifecycleResponsePayload,
    RunResponsePayload,
)
from apriori.flow.core.executor.types import PipelineExecutorType
from apriori.flow.core.pipeline.types import (
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
)
from apriori.flow.core.runner.types import (
    OnResultCallbackType,
    RunnerInputType,
    RunnerProtocol,
    RunnerResultType,
)
from apriori.flow.core.runner.utils import get_input_len_or_zero
from apriori.flow.lifecycle import SupportsLifecycle
from apriori.flow.progress.progress_mixin import ProgressMixin
from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.flow.structure import HasFlowStructure


@final
class AgentPoolRunner(
    RunnerProtocol[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    ProgressMixin,
    SupportsLifecycle,
    HasFlowStructure,
):
    __slots__ = (
        "input",
        "executor_factory_method",
        "num_agents",
        "agent_batch_size",
        "_task",
        "on_result",
        "_agents",
        "_response_queue",
        "_prepared",
        "_input_iterator",
    )
    # Runner protocol fields
    input: RunnerInputType
    on_result: OnResultCallbackType | None = None

    # Factory method to create new runner instances for agents
    executor_factory_method: Callable[[], PipelineExecutorType]

    # Number of agents in the pool
    num_agents: int

    # Agents pool configuration
    _agents: OrderedDict[int, ExecutorAgentLink]

    # Shared queue for receiving messages from agents
    _response_queue: Queue

    # Iterator over input items, initialized in run()
    _input_iterator: Iterator[PipelineInputType] | None

    # Current index of the input item being processed, using for mapping results
    _current_input_index: int

    # Lifecycle state
    _prepared: bool

    def __init__(
        self,
        input: RunnerInputType,
        executor_factory_method: Callable[[], PipelineExecutorType],
        num_agents: int,
    ) -> None:
        self.input = input
        self.executor_factory_method = executor_factory_method
        self.num_agents = num_agents
        self._response_queue = Queue()

        # Create agent links in init to represent whole comunication graph
        self._agents = {
            i: ExecutorAgentLink(
                agent_id=i,
                response_queue=self._response_queue,
                executor_factory_method=self.executor_factory_method,
            )
            for i in range(self.num_agents)
        }

        # Input iterator will be initialized in run()
        self._input_iterator = None
        self._current_input_index = 0
        self._prepared = False

    # ──── Main execution ────

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        self._input_iterator = iter(self.input)
        self._current_input_index = 0

        if not self._dispatch_next_input():
            self.progress.print("✔️ No input items to process.")
            return

        self._run_loop()

    def _dispatch_next_input(self) -> bool:
        # Dispatch input items to available agents

        if self._input_iterator is None:
            raise RuntimeError("Input iterator is not initialized.")

        available_agents = [
            agent for agent in self._agents.values() if agent.is_waiting
        ]
        if not available_agents:
            raise RuntimeError("No available agents to dispatch input to.")

        next_items = list(islice(self._input_iterator, len(available_agents)))
        if not next_items:
            self.progress.print("✔️ All items have been dispatched to workers.")
            return False

        for agent_link, item in zip(available_agents, next_items, strict=True):
            agent_link.run(item, self._current_input_index)
            self._current_input_index += 1

        return True

    # ──── Lifecycle support ────

    def prepare(self) -> None:
        if self._prepared:
            return

        # Prepare each agent via its link, which spawns the worker process
        for agent_link in self._agents.values():
            agent_link.prepare()

        #  Process responses until all agents have acknowledged
        self._run_loop()
        self._prepared = True

    def on_cycle_start(self) -> None:
        # Send event to the agents
        for agent_link in self._agents.values():
            agent_link.on_cycle_start()

        # Process responses until all agents have acknowledged
        self._run_loop()

    def on_cycle_end(self) -> None:
        # Send event to the agents
        for agent_link in self._agents.values():
            agent_link.on_cycle_end()

        # Process responses until all agents have acknowledged
        self._run_loop()

    def cleanup(self) -> None:
        if not self._prepared:
            return

        # Process cleanup for each agent
        for agent_link in self._agents.values():
            # Send exit signal to each agent
            agent_link.cleanup()

        self._run_loop()

        # Ensure all agent processes have exited
        for agent_link in self._agents.values():
            agent_link.process.join()
            # Remove agent tasks from progress tracker
            self.progress.remove_task(agent_link.task)

        self._prepared = False

    # ──── Progress task support ────

    def setup_process_task(self):
        super().setup_process_task()

        # Create or update task for this runner
        self.prepare_task(
            description=str(self.input), total=get_input_len_or_zero(self.input)
        )
        # Each agent has its own task under the pool task
        agent_links = self._agents.values()
        for agent_link in agent_links:
            agent_link.enable_task_tree_structure(
                self.task, is_last_subtask=agent_link is agent_links[-1]
            )
            agent_link.setup_process_task()

    # ──── Message handling ────

    def _run_loop(self) -> None:
        while True:
            # Process responses from running agents
            if not any(agent.is_running for agent in self._agents.values()):
                break

            try:
                message = self._response_queue.get(timeout=5)
            except queue.Empty:
                self.progress.print("⚠️ Timeout waiting for agent response.")
                break

            # Redirect progress calls from agents to the progress instance of this runner
            if ProgressRelay.relay_to(self.progress, message):
                continue

            # Fail-fast strategy on agent faults
            if not self._handle_message(message):
                self._handle_global_fault()
                raise RuntimeError("❌ ParallelRunner aborted due to agent failure.")

    def _handle_message(self, message: AgentMessage[Any]) -> bool:
        if message.agent_id not in self._agents:
            raise ValueError(
                f"Message agent_id {message.agent_id} does not match any known agent."
            )

        match message.type:
            case AgentMessageType.lifecycle_response:
                self._handle_lifecycle_response(message)
                return True  # Continue processing

            case AgentMessageType.run_response:
                self._handle_run_response(message)
                self._dispatch_next_input()
                return True  # Continue processing

            case AgentMessageType.fault:
                self._handle_fault_message(message)
                return False  # Stop processing on fault

            case _:
                raise TypeError(f"Unknown message type: {message.type}")

        raise RuntimeError("Unhandled message processing case.")

    def _handle_lifecycle_response(
        self, message: AgentMessage[LifecycleResponsePayload]
    ) -> None:
        self.progress.print(
            f"✔️ Agent {message.agent_id} has completed lifecycle phase: {message.payload.phase}."
        )
        # Update agent state
        self._agents[message.agent_id].waiting()

    def _handle_run_response(self, message: AgentMessage[RunResponsePayload]) -> None:
        self.progress.print(
            f"✔️ Agent {message.agent_id} has completed {message.item_index} processing."
        )
        # Update global progress
        self.progress.advance(self._task)

        agent_link = self._agents[message.agent_id]

        # Callback results
        if self.on_result is not None:
            self.on_result(
                RunnerResultType(
                    input_item_index=agent_link.input_item_index,
                    pipeline_result=message.payload.result,
                )
            )

        # Flag agent as waiting for new input
        agent_link.waiting()

    def _handle_fault_message(self, message: AgentMessage[AgentFaultPayload]) -> None:
        self.progress.print(f"❌ Agent fault: {message.payload.error}")

        # Update agent state
        self._agents[message.agent_id].failed()

    def _handle_global_fault(self) -> None:
        self.progress.print("💥 Global fault detected — stopping all agents.")

        # Notify all agents to stop
        for agent_link in self._agents.values():
            try:
                if agent_link.is_running:
                    agent_link.cleanup()  # graceful stop
                    # TODO: add support for graceful stop message in agent
            except Exception as e:
                self.progress.print(f"⚠️ Error during agent cleanup: {e}")

        # Force terminate any remaining agents
        for agent_link in self._agents.values():
            try:
                if agent_link.process.is_alive():
                    agent_link.process.terminate()
                    agent_link.process.join(timeout=1)

                    if agent_link.process.exitcode is None:
                        self.progress.print(
                            f"⚠️ Agent {agent_link.agent_id} did not exit cleanly."
                        )
            except Exception:
                pass

        self.progress.print("🛑 All agents terminated after fault.")


ParallelRunnerType: TypeAlias = AgentPoolRunner[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]
