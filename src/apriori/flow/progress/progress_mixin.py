from typing import final

from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import (
    HasProgress,
    ProgressProtocol,
    SupportsProgressTask,
    TaskStructure,
)


@final
class ProgressMixin(HasProgress, SupportsProgressTask):
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
    _is_last_subtask: bool
    _task: int | None

    def __init__(self) -> None:
        self.progress = NoOpProgress()
        self._task_structure = "undefined"
        self._parent_task = None
        self._task = None
        self._is_last_subtask = True

    # Progress setup method

    def setup_process_task(self) -> None:
        """Setup progress tracker and task structure for this instance.
        Method should be implemented in the subclass."""

        # Set default structure
        if self.task_structure == "undefined":
            self.enable_task_tree_structure()

    # Task preparation method

    def prepare_task(self, description: str, total: int = 1) -> None:
        # Progress tracker not set, skip task creation
        if self.progress is None:
            return

        match self.task_structure:
            case TaskStructure.hidden | TaskStructure.undefined:
                # Do not create a task
                return
            case TaskStructure.inline:
                # Use shared task from parent
                assert self._task is not None
                self.progress.update(
                    self._task,
                    description=description,
                    total=total,
                    completed=0,
                )
            case TaskStructure.tree:
                if self._task is None:
                    # Create a new task under parent
                    self._task = self.progress.add_task(
                        description,
                        completed=0,
                        total=total,
                        parent_task=self._parent_task,
                        is_last_subtask=self._is_last_subtask,
                    )
                else:
                    # Update existing task
                    self.progress.update(
                        self._task,
                        description=description,
                        total=total,
                        completed=0,
                    )
            case _:
                raise ValueError(f"Unknown task structure: {self.task_structure}")

    # Task structure management methods

    def enable_task_tree_structure(
        self, parent_task: int = None, is_last_subtask: bool = True
    ) -> None:
        self._task_structure = TaskStructure.tree
        self._parent_task = parent_task
        self._is_last_subtask = is_last_subtask
        self._task = None

    def enable_task_inline_structure(self, shared_task: int) -> None:
        self._task_structure = TaskStructure.inline
        self._parent_task = None
        self._task = shared_task

    def disable_task_structure(self) -> None:
        self._task_structure = TaskStructure.hidden
        self._parent_task = None
        self._task = None

    # Read-Only Properties

    @property
    def task_structure(self) -> TaskStructure:
        return self._task_structure

    @property
    def has_task_structure(self) -> bool:
        return self._task_structure != TaskStructure.undefined

    @property
    def parent_task(self) -> int | None:
        return self._parent_task

    @property
    def task(self) -> int | None:
        return self._task
