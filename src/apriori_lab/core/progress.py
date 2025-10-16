from typing import Any, Generic, Protocol, TypeVar

from rich.live import Live
from rich.progress import (
    Group,
)
from rich.text import Text
from typing_extensions import Self


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


class ConsoleProgress(ProgressProtocol):
    @property
    def tasks(self) -> list[Any]:
        return []

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        return 0

    def advance(self, task_id: int, advance: int = 1) -> None:
        pass

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
        pass

    def remove_task(self, task_id: int) -> None:
        pass

    def print(self, *objects: Any, **kw_args: Any) -> None:
        print(*objects, **kw_args)

    def log(self, *objects: Any, **kw_args: Any) -> None:
        print(*objects, **kw_args)


class NopProgress(ProgressProtocol):
    @property
    def tasks(self) -> list[Any]:
        return []

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        return 0

    def advance(self, task_id: int, advance: int = 1) -> None:
        pass

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
        pass

    def remove_task(self, task_id: int) -> None:
        pass

    def print(self, *objects: Any, **kw_args: Any) -> None:
        pass

    def log(self, *objects: Any, **kw_args: Any) -> None:
        pass


T = TypeVar("T", bound=ProgressProtocol)


class LiveProgress(ProgressProtocol, Generic[T]):
    def __init__(self, progress: T):
        # All protocol methods are forwarded to the internal progress instance
        self.progress: T = progress

        # Status needs to be set before being used in _make_layout
        self._status_task_id: int | None = None

        # Create live display for status updates
        self.live = Live(
            self._make_layout(),
            refresh_per_second=10,
            # Need to pass console to Live, otherwise it creates its own instance
            # which breaks printing from progress
            console=progress.console if hasattr(progress, "console") else None,
        )

    def _make_layout(self):
        # Collect current status of the tracked task
        status = ""
        if self._status_task_id is not None:
            status = next(
                task for task in self.tasks if task.id == self._status_task_id
            ).fields.get("status", status)

        if self.progress.console is not None:
            status_text = self.progress.console.render_str(status)
        else:
            status_text = Text(status)

        # Add status text on the second line below progress bars
        return Group(
            self.progress,
            status_text,
        )

    @property
    def tasks(self) -> list[Any]:
        return self.progress.tasks

    def add_task(
        self,
        description: str,
        total: int,
        show_status: bool = False,
        **fields: Any,
    ) -> int:
        task = self.progress.add_task(description, total=total, **fields)
        # Track this task as status task if requested
        if show_status:
            self._status_task_id = task
        return task

    def advance(self, task_id: int, advance: int = 1) -> None:
        self.progress.advance(task_id, advance=advance)

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
        self.progress.update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )
        # Update live display to reflect changes
        self.live.update(self._make_layout())

    def remove_task(self, task_id: int) -> None:
        self.progress.remove_task(task_id)

        # Clear status task if it was removed
        if self._status_task_id == task_id:
            self._status_task_id = None

        # Update live display to reflect changes
        self.live.update(self._make_layout())

    def print(self, *objects: Any, **kw_args: Any) -> None:
        self.progress.print(*objects, **kw_args)

    def log(self, *objects: Any, **kw_args: Any) -> None:
        self.progress.log(*objects, **kw_args)

    def __enter__(self) -> Self:
        # Enter live display context
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Exit live display context
        self.live.__exit__(exc_type, exc_val, exc_tb)
