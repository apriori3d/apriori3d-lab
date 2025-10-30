from collections import OrderedDict
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeAlias, final

from torch.multiprocessing import Process, Queue

from apriori.flow.progress.progress_relay import ProgressRelay
from apriori.flow.progress.types import (
    ProgressMixin,
)
from apriori.flow.runner.agent.agent import (
    RunnerAgentLink,
    RunnerAgentType,
)
from apriori.flow.runner.agent.messages import (
    AgentFaultPayload,
    AgentMessage,
    LifecycleResponsePayload,
    RunResponsePayload,
)
from apriori.flow.runner.types import (
    OnResultCallbackType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
    RunnerInputType,
    RunnerProtocol,
    RunnerResultType,
    RunnerType,
)


@final
class RunnerAgentPool(
    RunnerProtocol,
    Generic[
        RunnerInputType,
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
    ProgressMixin,
):
    __slots__ = (
        "input",
        "runner_factory_method",
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
    runner_factory_method: Callable[[], RunnerType]

    # Number of agents in the pool
    num_agents: int
    # Number of input items to process per agent
    agent_batch_size: int

    # Task for tracking overall progress
    _task: int | None

    # Agents pool configuration
    _agents: OrderedDict[int, RunnerAgentLink]

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
        runner_factory_method: Callable[[], RunnerType],
        num_agents: int,
        agent_batch_size: int = 1,
    ) -> None:
        self.input = input
        self.runner_factory_method = runner_factory_method
        self.num_agents = num_agents
        self.agent_batch_size = agent_batch_size

        self._agents = OrderedDict()
        self._response_queue = Queue()

        # Task will be initialized in prepare()
        self._task = None

        # Input iterator will be initialized in run()
        self._input_iterator = None
        self._current_input_index = 0

        self._prepared = False

    # Main execution

    def run(self) -> None:
        self.prepare()
        self.on_cycle_start()

        self._input_iterator = iter(self.input)
        self._current_input_index = 0

        if not self._dispatch_input():
            self.progress.print("✔️ No input items to process.")
            return

        self._run_loop()

    # RunnerLifecycleProtocol methods

    def prepare(self) -> None:
        if self._prepared:
            return

        self._task = self.progress.add_task(
            str(self.input), total=self.input_len, completed=0
        )
        self._spawn_agents()

        # Send prepare signal to each worker
        for worker_link in self._agents.values():
            worker_link.prepare()

        # Process responses until all workers have acknowledged
        self._run_loop()
        self._prepared = True

    def on_cycle_start(self) -> None:
        # Send event to the workers
        for worker_link in self._agents.values():
            worker_link.on_cycle_start()

        # Process responses until all workers have acknowledged
        self._run_loop()

    def on_cycle_end(self) -> None:
        # Send event to the workers
        for worker_link in self._agents.values():
            worker_link.on_cycle_end()

        # Process responses until all workers have acknowledged
        self._run_loop()

    def cleanup(self) -> None:
        if not self._prepared:
            return

        # Process cleanup for each worker
        for worker_link in self._agents.values():
            # Send exit signal to each worker
            worker_link.cleanup()

        self._run_loop()

        # Ensure all worker processes have exited
        for worker_link in self._agents.values():
            worker_link.process.join()
            # Remove worker tasks from progress tracker
            self.progress.remove_task(worker_link.task)

        self._prepared = False

    # Run loop and message handling

    def _run_loop(self) -> None:
        # Process responses from agents until all input items are processed
        while any([agent.is_running for agent in self._agents.values()]):
            message = self._response_queue.get()

            # Redirect progress calls from agents to the main progress
            if ProgressRelay.relay_to(self.progress, message):
                continue

            should_continue = self._handle_message(message)
            if not should_continue:
                break

    def _handle_message(self, message: AgentMessage[Any]) -> bool:
        if message.agent_id not in self._agents:
            raise ValueError(
                f"Message agent_id {message.agent_id} does not match any known agent."
            )

        match message.type:
            case "lifecycle_response":
                self._handle_lifecycle_response(message)
                return True  # Continue processing

            case "run_response":
                new_items_dispatched = self._handle_run_response(message)
                return new_items_dispatched  # Continue processing if new items were dispatched

            case "fault":
                self._handle_fault_message(message)
                return False  # Stop processing on fault

            case _:
                raise TypeError(f"Unknown message type: {message.type}")

        return True  # Continue processing

    def _handle_lifecycle_response(
        self, message: AgentMessage[LifecycleResponsePayload]
    ) -> None:
        self.progress.print(
            f"✔️ Agent {message.agent_id} has completed lifecycle phase: {message.payload.phase}."
        )

        # Update agent state
        agent_link = self._agents[message.agent_id]
        agent_link.waiting()

    def _handle_run_response(self, message: AgentMessage[RunResponsePayload]) -> bool:
        self.progress.print(
            f"✔️ Agent {message.agent_id} has completed {message.item_index} processing."
        )

        # Update global progress
        self.progress.advance(self._task, self.agent_batch_size)

        agent_link = self._agents[message.agent_id]

        if self.on_result is not None:
            # Callback with each result
            for item_index, result in zip(
                agent_link.input_indices, message.payload.result, strict=False
            ):
                self.on_result(
                    RunnerResultType(
                        input_index=item_index,
                        pipeline_result=result,
                    )
                )

        # Update agent state
        agent_link = self._agents[message.agent_id]
        agent_link.waiting()

        # Dispatch next item to the agent that has just finished processing
        return self._dispatch_input()

    def _handle_fault_message(self, message: AgentMessage[AgentFaultPayload]) -> None:
        self.progress.print(f"❌ Agent fault: {message.payload.error}")

        # Update agent state
        agent_link = self._agents[message.agent_id]
        agent_link.failed()

    # Dispatch input items to available agents

    def _dispatch_input(self) -> bool:
        if self._input_iterator is None:
            raise RuntimeError("Input iterator is not initialized.")

        available_agents = [
            agent for agent in self._agents.values() if agent.is_waiting
        ]
        if not available_agents:
            raise RuntimeError("No available agents to dispatch input to.")

        has_dispatched = False

        for agent_link in available_agents:
            # Fetch next batch of items
            next_batch = []
            try:
                for _ in range(self.agent_batch_size):
                    next_batch.append(self._input_iterator.__next__())

            except StopIteration:
                if len(next_batch) == 0:
                    self.progress.print("✔️ All items have been dispatched to workers.")
                    return has_dispatched

            input_indices = list(
                range(
                    self._current_input_index,
                    self._current_input_index + len(next_batch),
                )
            )
            agent_link.run(next_batch, input_indices=input_indices)
            self._current_input_index += len(next_batch)
            has_dispatched = True

        return has_dispatched

    # Agent spawning

    def _spawn_agents(self) -> None:
        self._agents = OrderedDict(
            {
                agent_id: self._spawn_agent(
                    agent_id,
                    # Show livestatus only for the last agent
                    show_status=(agent_id == self.num_agents - 1),
                )
                for agent_id in range(self.num_agents)
            }
        )

    def _spawn_agent(self, agent_id: int, show_status: bool) -> RunnerAgentLink:
        # Add task for the agent to the progress tracker
        task = self.progress.add_task(
            f"Worker {agent_id}",
            total=0,
            show_status=show_status,
            parent_task=self._task,
            is_last_subtask=show_status,
        )
        # Create queue for communication with the agent
        agent_queue = Queue()
        # Create and start agent process
        agent_process = Process(
            target=self.agent_fn,
            args=(
                agent_id,
                agent_queue,
                self._response_queue,
                self.runner_factory_method,
                task,
            ),
        )
        agent_process.start()

        return RunnerAgentLink(
            agent_id=agent_id,
            process=agent_process,
            queue=agent_queue,
            task=task,
        )

    @staticmethod
    def agent_fn(
        agent_id: int,
        agent_queue: Queue,
        response_queue: Queue,
        runner_factory_method: Callable[[], RunnerType],
        task: int,
    ):
        runner = runner_factory_method()
        agent = RunnerAgentType(
            agent_id=agent_id,
            runner=runner,
            request_queue=agent_queue,
            response_queue=response_queue,
        )
        # Progress relay will forward progress back to the main process via worker's response queue.
        agent.progress = ProgressRelay(response_queue)
        # Use single task for displaying runner and executor progress
        agent.shared_task = task
        # Start worker loop
        agent.run_loop()


WorkersPoolType: TypeAlias = RunnerAgentPool[
    RunnerInputType,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]
