from typing import Any, Literal, cast

from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import ProgressProtocol

TaskStructure = Literal["undefined", "tree", "inline", "hidden"]


class ProgressMixin:
    __slots__ = (
        "_task_structure",
        "progress",
        "_parent_task",
        "_is_last_subtask",
        "_task",
    )

    # Public property
    progress: ProgressProtocol | None

    # Internal state
    _task_structure: TaskStructure
    _parent_task: int | None
    _is_last_subtask: bool | None
    _task: int | None

    def __init__(self) -> None:
        self.progress = None
        self._task_structure = "undefined"
        self._parent_task = None
        self._task = None
        self._is_last_subtask = None
        self.progress = NoOpProgress()

    # Task preparation method

    def prepare_task(self, description: str, total: int) -> None:
        # Progress tracker not set, skip task creation
        if self.progress is None:
            return

        match self.task_structure:
            case "hidden", "undefined":
                # Do not create a task
                return
            case "inline":
                # Use shared task from parent
                assert self._task is not None
                self.progress.update(
                    self.task,
                    description=description,
                    total=total,
                    completed=0,
                )
            case "tree":
                # Create a new task under parent
                assert self.parent_task is not None
                assert self.is_last_subtask is not None
                self._task = self.progress.add_task(
                    description,
                    completed=0,
                    total=total,
                    parent_task=self.parent_task,
                    is_last_subtask=self.is_last_subtask,
                )
            case _:
                raise ValueError(f"Unknown task structure: {self.task_structure}")

    # Task structure management methods

    def enable_task_tree_structure(
        self, parent_task: int = None, is_last_subtask: bool = True
    ) -> None:
        self._task_structure = "tree"
        self._parent_task = parent_task
        self._is_last_subtask = is_last_subtask
        self._task = None

    def enable_task_inline_structure(self, shared_task: int) -> None:
        self._task_structure = "inline"
        self._parent_task = None
        self._task = shared_task

    def disable_task_structure(self) -> None:
        self._task_structure = "hidden"
        self._parent_task = None
        self._task = None

    def propagate_task_structure(
        self, instance: Any, is_last_subtask: bool = True
    ) -> None:
        if not isinstance(self, ProgressMixin):
            return
        instance_progress = cast(ProgressMixin, instance)
        instance_progress.progress = self.progress

        # Create task structure based on runner's task structure
        match self.task_structure:
            case "undefined":
                # Do not create task structure
                pass
            case "hidden":
                instance_progress.disable_task_structure()
            case "tree":
                # Setup tree structure under runner task
                instance_progress.enable_task_tree_structure(
                    parent_task=self.task, is_last_subtask=is_last_subtask
                )
            case "inline":
                # Use shared task from runner
                instance_progress.enable_task_inline_structure(self.task)

    # Read-Only Properties

    @property
    def task_structure(self) -> TaskStructure:
        return self._task_structure

    @property
    def has_task_structure(self) -> bool:
        return self._task_structure != "undefined"

    @property
    def parent_task(self) -> int | None:
        return self._parent_task

    @property
    def task(self) -> int | None:
        return self._task
