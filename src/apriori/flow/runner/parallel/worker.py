import builtins
from ast import TypeAlias
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeAlias

import torch
from torch.multiprocessing import Queue

from apriori.flow.pipeline.types import (
    PipelineContextType,
    PipelineExecutorProtocol,
)
from apriori.flow.progress.types import ProgressProtocol
from apriori.flow.runner.sequential import SequentialRunner, SequentialRunnerConfig
from apriori.flow.runner.types import (
    PipelineConfigType,
    PipelineInputType,
    PipelineOutputType,
    RunnereResult,
)


@dataclass()
class WorkerPrepareRequest(Generic[PipelineConfigType]):
    task: int
    pipeline_config: PipelineConfigType


@dataclass()
class WorkerExecuteRequest(Generic[PipelineInputType]):
    items: Iterable[PipelineInputType]
    state: dict[str, torch.Tensor] | None = None


@dataclass()
class WorkerProgressUpdate:
    progress_method: Callable[[Any], None]
    progress_args: tuple
    progress_kwargs: dict


@dataclass()
class WorkerExecuteResponse(Generic[PipelineOutputType]):
    task: int
    item_index: int
    output: PipelineOutputType


class UpdatableInput(Iterable[PipelineInputType]):
    def update(self, items: Iterable[PipelineInputType]) -> None:
        self.items = items

    def __iter__(self):
        return iter(self.items)


WorkerExecutor: TypeAlias = PipelineExecutorProtocol[
    PipelineConfigType, PipelineContextType, PipelineInputType, PipelineOutputType
]

WorkerRunner: TypeAlias = SequentialRunner[
    UpdatableInput,
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]

WorkerRunnerResult: TypeAlias = RunnereResult[
    PipelineConfigType,
    PipelineContextType,
    PipelineInputType,
    PipelineOutputType,
]


class Worker(
    ProgressProtocol,
    Generic[
        PipelineConfigType,
        PipelineContextType,
        PipelineInputType,
        PipelineOutputType,
    ],
):
    __slots__ = (
        "_runner",
        "_number",
        "_input_queue",
        "_output_queue",
        "_task",
    )
    _runner: WorkerRunner
    _number: int
    _input_queue: Queue
    _output_queue: Queue
    _task: int

    def __init__(
        self,
        executor: WorkerExecutor,
        number: int,
        input_queue: Queue,
        output_queue: Queue,
        task: int,
    ):
        self._number = number
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._task = task

        self.runner_input = UpdatableInput()
        self._runner = WorkerRunner(
            config=SequentialRunnerConfig(),
            input=self.runner_input,
            executor=executor,
            on_pipeline_result=self._on_pipeline_result,
        )
        self._runner.progress = self
        self._items_processed = 0
        self._items_total = 0

        # Redirect print to progress routine, because print interferes with rich.Progress
        builtins.print = self.print

    def run(self) -> None:
        # Process requests until a None request is received
        while True:
            request = self._input_queue.get()

            if request is None:
                break

            # Prepare the pipeline (e.g. load models to the device)
            if isinstance(request, WorkerPrepareRequest):
                self._runner.prepare()
                continue

            if isinstance(request, WorkerExecuteRequest):
                if self._items_total > 0:
                    raise RuntimeError(
                        "Worker is already processing items, cannot accept new execute request."
                    )
                # Process an input item in the pipeline
                self.runner_input.update(request.items)
                self._items_total = len(list(request.items))
                self._items_processed = 0

                # Restore context state if provided
                if request.state is not None:
                    self._runner.load_state_dict(request.state)
                    self.print(f"Worker {self._number} restored state from request")

                self.print(
                    f"Worker {self._number} start to process items {list(str(request.items))}"
                )

                # TODO: manage task context for progress reporting

                # Run the pipeline
                self._runner.run()

                self.print(
                    f"Worker {self._number} finished processing items {list(str(request.items))}"
                )
                # Ensure all items were processed
                if self._items_processed != self._items_total:
                    raise RuntimeError(
                        f"Worker processed {self._items_processed} items, "
                        f"but expected {self._items_total}."
                    )
                # Reset counters and wait for the next request
                self._items_total = 0
                self._items_processed = 0

            # TODO: send error response back
            raise TypeError(f"Unknown request type: {type(request)}")

    def _on_pipeline_result(self, runner_result: WorkerRunnerResult) -> None:
        self._items_processed += 1

        # TODO: Move output to cpu and serialize?
        self._output_queue.put(
            WorkerExecuteResponse(
                task=self._task,
                item_index=runner_result.item_index,
                output=runner_result.pipeline_result.output,
            ),
        )

    # ProgressProtocol methods

    def print(self, message: str):
        self._output_queue.put(
            WorkerProgressUpdate(
                item_index=self.item_index,
                progress_method="print",
                progress_args=(message,),
                progress_kwargs={},
            ),
        )

    def update(
        self,
        task_id: int,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if task_id != self._task:
            raise ValueError(
                f"Worker progress update task_id {task_id} does not match worker task {self._task}"
            )
        self._output_queue.put(
            WorkerProgressUpdate(
                item_index=self.item_index,
                progress_method="update",
                progress_args=(self._task,),
                progress_kwargs={
                    "description": description,
                    "total": total,
                    "completed": completed,
                    "advance": advance,
                    "visible": visible,
                    "refresh": refresh,
                    **fields,
                },
            ),
        )

    def advance(self, task: int, n: float = 1.0):
        self._output_queue.put(
            WorkerProgressUpdate(
                item_index=self.item_index,
                progress_method="advance",
                progress_args=(task, n),
                progress_kwargs={},
            ),
        )

    def remove_task(self, task_id: int):
        if task_id != self._task:
            raise ValueError(
                f"Worker remove_task task_id {task_id} does not match worker task {self._task}"
            )
        self._output_queue.put(
            WorkerProgressUpdate(
                item_index=self.item_index,
                progress_method="remove_task",
                progress_args=(task_id,),
                progress_kwargs={},
            ),
        )
