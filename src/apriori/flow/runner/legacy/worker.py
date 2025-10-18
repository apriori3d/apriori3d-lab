import builtins
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.multiprocessing import Queue


@dataclass()
class WorkerPrepareRequest:
    task: int
    plan: Any  # PipelinePlan


@dataclass()
class WorkerPipelineRequest:
    frame_index: int
    images: np.ndarray
    body_model_state: dict | None = None


@dataclass()
class WorkerProgressResponse:
    frame_index: int | None
    method: Callable[[Any], None]
    args: tuple
    kwargs: dict


@dataclass()
class WorkerResultsResponse:
    task: int
    frame_index: int
    context: MultiViewSceneContext[np.ndarray]
    body_model_state: dict
    body: Body3D


class Worker(ProgressProtocol):
    def __init__(
        self,
        pipeline: PipelineProtocol,
        num: int,
        input_queue: Queue,
        output_queue: Queue,
        device_name: str,
        task: int,
    ):
        self.pipeline = pipeline
        self.num = num
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.task = task

        self.device = torch.device(device_name)
        self.frame_index: int | None = None

        if (
            self.pipeline.interactive
            and threading.current_thread() is not threading.main_thread()
        ):
            self.print(
                "⚠️ Interactive mode is only supported in the main thread and will be disabled."
            )
            pipeline.interactive = False

        # Redirect print to progress routine, because print interferes with rich.Progress
        builtins.print = self.print

    def run(self) -> None:
        # Process requests until a None request is received
        while True:
            request = self.input_queue.get()

            if request is None:
                break

            # Prepare the pipeline (e.g. load models to the device)
            if isinstance(request, WorkerPrepareRequest):
                self.pipeline.prepare(
                    plan=request.plan,
                    device=self.device,
                    progress=self,
                    task=self.task,
                )
                continue

            if not isinstance(request, WorkerPipelineRequest):
                raise TypeError(f"Unknown request type: {type(request)}")

            # Process a frame in the pipeline
            self.frame_index = request.frame_index
            self.print(
                f"Worker {self.num} start to process frame {request.frame_index}"
            )

            # Convert body model state back to tensors
            if request.body_model_state is not None:
                # Update pipeline body model state
                body_model_state = numpy_to_tensor(request.body_model_state)
                self.pipeline.scene.body_model.to("cpu")
                self.pipeline.scene.body_model.load_state_dict(body_model_state)

            # Run the pipeline
            final_body = self.pipeline(request.frame_index, request.images)

            # Move result to cpu and send back to main process
            # 1. Body SMPL parameters
            final_body = move_tensors_to_cpu(final_body)
            # 2. Predicted scene content in original scale (e.g. keypoints, body masks)
            context = self.pipeline.scene.original.context.to_numpy()
            # 3. State of the body model
            self.pipeline.scene.body_model.to("cpu")
            body_model_state = tensors_to_numpy(
                self.pipeline.scene.body_model.state_dict()
            )

            self.output_queue.put(
                WorkerResultsResponse(
                    task=self.task,
                    frame_index=request.frame_index,
                    context=context,
                    body_model_state=body_model_state,
                    body=final_body,
                ),
            )

    def print(self, message: str):
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="print",
                args=(message,),
                kwargs={},
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
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="update",
                args=(self.task,),
                kwargs={
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
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="advance",
                args=(task, n),
                kwargs={},
            ),
        )

    def add_level(self):
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="add_level",
                args=(),
                kwargs={},
            ),
        )

    def remove_level(self):
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="remove_level",
                args=(),
                kwargs={},
            ),
        )

    def remove_task(self):
        self.output_queue.put(
            WorkerProgressResponse(
                frame_index=self.frame_index,
                method="remove_task",
                args=(self.task,),
                kwargs={},
            ),
        )
