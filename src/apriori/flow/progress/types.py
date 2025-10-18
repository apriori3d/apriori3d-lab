from typing import Any, Protocol

from typing_extensions import runtime_checkable


class ProgressProtocol(Protocol):
    @property
    def tasks(self) -> list[Any]: ...
    def add_task(self, description: str, total: int, **fields: Any) -> int: ...
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
    def shared_task(self) -> int | None:
        return self._shared_task

    @shared_task.setter
    def shared_task(self, task: int | None) -> None:
        self._shared_task = task


@runtime_checkable
class WithProgress(Protocol):
    progress: ProgressProtocol | None
    shared_task: int | None



@runtime_checkable
class ProgressWithPrefix(Protocol):
    @property
    def prefix(self) -> str: ...

    @prefix.setter
    def prefix(self, prefix: str) -> None: ...


@runtime_checkable
class ProgressWithLevels(Protocol):
    def add_level(self, prefix: str | None = None) -> None: ...
    def remove_level(self) -> None: ...






