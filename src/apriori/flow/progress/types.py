from typing import Any, Protocol

from typing_extensions import runtime_checkable


class ProgressProtocol(Protocol):
    @property
    def tasks(self) -> list[Any]: ...
    def add_task(
        self,
        description: str,
        total: int,
        **fields: Any,
    ) -> int: ...
    def advance(self, task_id: int, advance: int = 1) -> None: ...
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
    ) -> None: ...
    def remove_task(self, task_id: int) -> None: ...
    def print(self, *objects: Any, **kw_args: Any) -> None: ...
    def log(self, *objects: Any, **kw_args: Any) -> None: ...


class ProgressMixin:
    _progress: ProgressProtocol | None = None
    _shared_task: int | None = None

    @property
    def progress(self) -> ProgressProtocol | None:
        return self._progress

    @progress.setter
    def progress(self, progress: ProgressProtocol | None) -> None:
        self._progress = progress

    @property
    def has_shared_task(self) -> bool:
        return self._shared_task is not None

    @property
    def shared_task(self) -> int | None:
        return self._shared_task

    @shared_task.setter
    def shared_task(self, task: int | None) -> None:
        self._shared_task = task

    def prepare_task(self, description: str, total: int) -> int | None:
        if self.progress is None:
            return None
        if self.has_shared_task:
            self.progress.update(
                self.shared_task,
                description=description,
                total=total,
                completed=0,
            )
            return self.shared_task
        return self.progress.add_task(description, completed=0, total=total)


@runtime_checkable
class HasProgress(Protocol):
    progress: ProgressProtocol | None
    shared_task: int | None
