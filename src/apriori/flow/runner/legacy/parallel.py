import builtins
import faulthandler
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.multiprocessing import Process, Queue

faulthandler.enable()


def worker_fn(
    num: int,
    input_queue: Queue,
    output_queue: Queue,
    settings: Settings,
    device_name: str,
    task: int,
):
    pipeline = MainPipeline(settings=settings, interactive=False)
    worker = Worker(
        pipeline=pipeline,
        num=num,
        input_queue=input_queue,
        output_queue=output_queue,
        device_name=device_name,
        task=task,
    )
    worker.run()


# Add body model to the callback arguments, to be able to pass the body model across frames via queue
RunnerResultCallback = Callable[
    [int, MultiViewSceneContext[np.ndarray], Body3D, BodyModel | None], None
]


class ParallelRunner:
    def __init__(
        self,
        video_sources: list[AbstractVideoReader],
        settings: Settings,
        num_workers: int = 1,
    ):
        self.video_source = BatchVideoReader(video_sources)
        self.settings = settings
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self.num_workers = num_workers
        self._prepared = False

    def _prepare(self, progress: ProgressProtocol) -> None:
        if self._prepared:
            return
        self._prepared = True
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)

        if self.device.type == "cuda":
            progress.print(f"🚀 Using GPU {torch.cuda.get_device_name(0)}")
        else:
            progress.print("🐌 Using CPU")
            # TODO: move logic to PipelinePlan
            # if not (self.settings.use_gt_keypoints3d or self.settings.use_gt_body_masks):
            #    progress.print('⚠️ WARNING: Running on CPU without ground truth can be very slow!')

    def run(
        self,
        on_frame_results_callback: PipelineResultCallback | None = None,
    ) -> None:
        # Processing each video frame
        with create_live_progress() as live:
            # Redirect print to progress routine, because print interferes with rich.Progress bars
            builtins.print = live.print

            self._prepare(live)

            """
            The plan for executing the pipeline is as follows:

            Step 1: Spawn single worker and process the first frame
            with full shape optimization (including segmentation).

            Step 2: Spawn multiple workers and process the remaining frames
            with pose-only optimization (without segmentation).

            Note: Using segmentation of the full-resolution images produces better results,
            but currently consumes ~ 10Gb of GPU memory.
            Terminating the worker after processing the first frame frees up memory for the next workers.
            """

            # The progress routine:
            # The last task status displays in the live progress bar

            # All workers will send their progress and results to this queue
            response_queue = Queue()

            # Step 1: full shape optimization with segmentation.

            # Add global task on to to track the progress for step
            total = 1
            global_task = live.add_task(
                "Fitting the shape on the first frame", total=total
            )

            # All workers will be displayed as sub-tasks
            live.progress.add_level()

            initial_body_model = self._run_pose_and_shape_plan(
                live,
                global_task,
                response_queue,
                on_frame_results_callback,
            )
            live.progress.remove_level()
            # Reset completed count for the next step,
            # because all frames will be processed on the second step.

            # Step 2: pose-only optimization without segmentation
            # Note: The body shape from the first frame is used as initial shape for the remaining frames.
            total = len(self.settings.frame_range)
            live.update(
                global_task,
                total=total,
                completed=0,
                description="Fitting poses on the video sequence",
            )
            live.progress.add_level()
            next_frame = 0  # All frames will be processed on second step, to produce consistent results

            self._run_pose_only_plan(
                live,
                next_frame,
                total,
                global_task,
                response_queue,
                initial_body_model,
                on_frame_results_callback,
            )

            live.progress.remove_level()
            live.remove_task(global_task)
            live.print("✅ Video sequence processing is completed.")

    def _run_pose_and_shape_plan(
        self, progress, global_task, response_queue, on_frame_results_callback
    ) -> BodyModel:
        plan = PipelinePlan(
            align_body_by_keypoints3d=True,
            apply_body_segmentation=True,
            apply_hair_mask=True,
            optimization_plan="pose_and_shape",
        )

        # Spawn single worker to process the first frame with full shape optimization
        worker, request_queue, task = self._spawn_worker(
            progress,
            response_queue,
            plan,
            num=0,
            show_status=True,
        )

        # Send request for the first frame with full shape optimization.
        # Note: body state is None to use the default initial body shape.
        frame_index = self.settings.frame_range.start
        images = self.video_source[frame_index].copy()
        request_queue.put(WorkerPipelineRequest(frame_index, images))

        # Receive the body model from the first frame to use as initial shape for the remaining frames
        initial_body_model: BodyModel | None = None

        def on_frame_results(
            frame_index: int,
            context: MultiViewSceneContext[np.ndarray],
            body: Body3D,
            body_model: BodyModel | None,  # added body_model argument
        ) -> None:
            nonlocal initial_body_model
            if body_model is None:
                raise ValueError("Body model must be provided for the first frame")

            initial_body_model = body_model

            if on_frame_results_callback is not None:
                on_frame_results_callback(frame_index, context, body)

        # Process responses from worker
        self._process_response_queue(
            progress=progress,
            global_task=global_task,
            worker_queue=None,  # worker_queue is None because there is only one frame to process
            response_queue=response_queue,
            next_frame=1,
            completed=0,
            total=1,
            return_body_model=True,
            on_frame_results_callback=on_frame_results,
        )

        # Terminate the first worker to free up GPU memory
        self._shut_down_workers(progress, [(worker, request_queue, task)])

        return initial_body_model

    def _run_pose_only_plan(
        self,
        progress,
        next_frame,
        total,
        global_task,
        response_queue,
        initial_body_model,
        on_frame_results_callback,
    ):
        plan = PipelinePlan(
            align_body_by_keypoints3d=False,
            apply_body_segmentation=False,
            apply_hair_mask=False,
            optimization_plan="pose_only",
        )
        workers = []
        worker_queue = {}

        for num in range(self.num_workers):
            worker, request_queue, task = self._spawn_worker(
                progress,
                response_queue,
                plan,
                num=num,
                show_status=(
                    num == self.num_workers - 1
                ),  # Show livestatus only for the last worker
            )
            worker_queue[task] = request_queue
            workers.append((worker, request_queue, task))

        # Use predicted body as initial for other frames
        initial_body_model_state = initial_body_model.cpu().state_dict()
        # Convert body model state to numpy for inter-process transfer
        initial_body_model_state = {
            key: value.cpu().numpy() for key, value in initial_body_model_state.items()
        }

        # Distribute first video frames between all workers (one frame per worker).
        # Note: the first frame has already been processed in the previous step.
        # Note: other frames will be distributed to workers dynamically as they finish processing.

        for worker_index in range(self.num_workers):
            # Get the next worker in round-robin fashion to make evenly distributed load
            _, request_queue, _ = workers[worker_index % self.num_workers]
            request_queue.put(
                WorkerPipelineRequest(
                    frame_index=next_frame,
                    images=self.video_source[next_frame].copy(),
                    body_model_state=initial_body_model_state,
                ),
            )
            next_frame += 1

        def on_frame_results(
            frame_index: int,
            context: MultiViewSceneContext[np.ndarray],
            body: Body3D,
            *_: BodyModel
            | None,  # added body_model argument, but ignored (only used in the first step)
        ) -> None:
            if on_frame_results_callback is not None:
                on_frame_results_callback(frame_index, context, body)

        # Process responses from workers
        self._process_response_queue(
            progress=progress,
            global_task=global_task,
            worker_queue=worker_queue,
            response_queue=response_queue,
            next_frame=next_frame,
            completed=0,
            total=total,
            body_model_state=initial_body_model_state,
            return_body_model=False,
            on_frame_results_callback=on_frame_results,
        )

        # Shutdown workers
        self._shut_down_workers(progress, workers)

    def _spawn_worker(
        self,
        progress: ProgressProtocol,
        response_queue: Queue,
        plan: Any,
        num: int,
        send_initial_prepare: bool = True,
        show_status: bool = False,
    ):
        task = progress.add_task(f"Worker {num + 1}", total=0, show_status=show_status)
        request_queue = Queue()

        worker = Process(
            target=worker_fn,
            args=(
                num,
                request_queue,
                response_queue,
                self.settings,
                self.device_name,
                task,
            ),
        )
        worker.start()

        if send_initial_prepare:
            request_queue.put(WorkerPrepareRequest(task, plan))

        return worker, request_queue, task

    def _shut_down_workers(self, live, workers):
        for worker, request_queue, task in workers:
            request_queue.put(None)  # Send exit signal to each worker
            live.remove_task(task)
            worker.join()

    def _process_response_queue(
        self,
        progress: ProgressProtocol,
        global_task: int,
        worker_queue: dict[int, Queue],
        response_queue: Queue,
        next_frame: int,
        completed: int,
        total: int,
        body_model_state: dict | None = None,
        return_body_model: bool = False,
        on_frame_results_callback: RunnerResultCallback | None = None,
    ) -> None:
        while True:
            item = response_queue.get()

            # Redirect progress calls from workers to the main progress routine
            if isinstance(item, WorkerProgressResponse):
                if (
                    item.method == "print"
                    and item.frame_index is not None
                    and len(item.args) == 1
                    and isinstance(item.args[0], str)
                ):
                    # Single string argument - add prefix with frame index
                    progress.print(f"Frame {item.frame_index}: {item.args[0]}")
                else:
                    getattr(progress, item.method)(*item.args, **item.kwargs)

            # Collect results from workers
            elif isinstance(item, WorkerResultsResponse):
                progress.print(f"✅ Frame {item.frame_index} processing is completed.")

                # Pass results to callback (to save results or display interactive visualization)
                if on_frame_results_callback is not None:
                    # Create body model from the returned state
                    if return_body_model:
                        body_model = load_body_model(self.settings, "cpu")
                        body_model_state = numpy_to_tensor(item.body_model_state)
                        body_model.load_state_dict(body_model_state)
                    else:
                        body_model = None
                    # Pass to callback
                    on_frame_results_callback(
                        item.frame_index, item.context, item.body, body_model
                    )

                # Update global progress
                progress.advance(global_task)
                completed += 1
                if completed >= total:
                    break

                # Send next frame to the worker that has just finished processing
                if next_frame < total:
                    worker_queue[item.task].put(
                        WorkerPipelineRequest(
                            frame_index=next_frame,
                            images=self.video_source[next_frame].copy(),
                            body_model_state=body_model_state,  # Use default initial body shape
                        ),
                    )
                    next_frame += 1
