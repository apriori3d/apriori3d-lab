import argparse
import contextlib

# from ai_vision.pipelines.body3d import recordings
# from ai_vision.pipelines.body3d.pipeline.result_saver import ResultSaver
# from ai_vision.pipelines.body3d.run.parallel import ParallelRunner
# from ai_vision.pipelines.body3d.run.single import SingleRunner
# from ai_vision.pipelines.body3d.settings import Settings
# from ai_vision.pipelines.body3d.utils.script_utils import is_running_in_jupyter
# from ai_vision.pipelines.body3d.utils.utils import load_video_source


# def run(
#     source_name: str,
#     num_frames: int | None = None,
#     use_gt_keypoints3d: bool = False,
#     use_gt_body_masks: bool = False,
#     use_gt_cameras: bool = False,
#     run_evaluation: bool = False,
#     interactive: bool | None = None,
#     num_workers: int | None = None,
#     **pipeline_kwargs,
# ):
#     settings = recordings[source_name]()

#     if num_frames is not None:
#         if num_frames <= 0:
#             raise ValueError('num_frames must be positive or None')
#         settings.end_frame = min(settings.num_frames - 1, settings.start_frame + num_frames - 1)

#     settings.use_gt_keypoints3d = use_gt_keypoints3d
#     settings.use_gt_body_masks = use_gt_body_masks
#     settings.use_gt_cameras = use_gt_cameras

#     run_with_settings(
#         settings=settings,
#         run_evaluation=run_evaluation,
#         interactive=interactive,
#         num_workers=num_workers,
#         **pipeline_kwargs,
#     )


# def run_with_settings(
#     settings: Settings,
#     save_results: bool = True,
#     interactive: bool | None = None,
#     run_evaluation: bool = False,
#     num_workers: int | None = None,
#     **plan_kwargs,
# ):
#     if num_workers is not None:
#         if num_workers > 0:
#             if is_running_in_jupyter():
#                 print('⚠️ Multiprocessing is not supported in Jupyter notebooks. Setting num_workers=0.')
#                 num_workers = 0
#             if interactive is True:
#                 print('⚠️ Interactive mode is not supported with multiprocessing. Setting interactive=False.')
#                 interactive = False
#     else:
#         num_workers = 0 if is_running_in_jupyter() or interactive else 4  # Optimal loading for 3090 GPU for main pipeline

#     interactive = interactive if interactive is not None else is_running_in_jupyter()
#     video_source = load_video_source(settings)
#     if save_results:
#         saver = ResultSaver(settings)
#         result_saver_callback = saver.on_frame_results
#     else:
#         result_saver_callback = None

#     if settings.is_synthetic:
#         default_plan_kwargs = dict(
#             use_gt_cameras=True,
#             use_gt_keypoints=True,
#             use_gt_body_masks=True,
#         )
#         plan_kwargs = {**default_plan_kwargs, **plan_kwargs}

#     if num_workers > 0:
#         from torch.multiprocessing import set_start_method

#         with contextlib.suppress(RuntimeError):
#             set_start_method('spawn', force=False)

#         pipeline = ParallelRunner(
#             video_sources=video_source,
#             settings=settings,
#             num_workers=num_workers,
#             **plan_kwargs,
#         )
#     else:
#         pipeline = SingleRunner(
#             video_sources=video_source,
#             settings=settings,
#             interactive=interactive,
#             **plan_kwargs,
#         )

#     pipeline.run(on_frame_results_callback=result_saver_callback)

#     if run_evaluation:
#         from ai_vision.pipelines.body3d.evaluation import run_with_settings as evaluate

#         evaluate(settings)
