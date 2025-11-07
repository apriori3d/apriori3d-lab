from enum import Enum, auto
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


@runtime_checkable
class HasProgress(Protocol):
    progress: ProgressProtocol | None


class TaskStructure(Enum):
    hidden = auto()
    inline = auto()
    tree = auto()
    undefined = auto()


# @runtime_checkable
# class SupportsProgressTask(Protocol, HasProgress):
#     task_structure: TaskStructure

#     def setup_process_task(self) -> None: ...

#     def prepare_task(self, description: str, total: int) -> None: ...

#     def enable_task_tree_structure(
#         self, parent_task: int = None, is_last_subtask: bool = True
#     ) -> None: ...

#     def enable_task_inline_structure(self, shared_task: int) -> None: ...

#     def disable_task_structure(self) -> None: ...
