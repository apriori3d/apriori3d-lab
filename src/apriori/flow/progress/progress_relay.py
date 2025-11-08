from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from multiprocessing.queues import Queue
from typing import TYPE_CHECKING, Any

from apriori.flow.progress.types import ProgressProtocol


class ProgressRelayMethod(Enum):
    print = "print"
    log = "log"
    add_task = "add_task"
    update = "update"
    advance = "advance"
    remove_task = "remove_task"


@dataclass()
class ProgressRelayMessage:
    progress_method: ProgressRelayMethod
    progress_args: tuple[Any, ...]
    progress_kwargs: dict[str, Any]


if TYPE_CHECKING:
    RelayQueue = Queue[Any]
else:
    RelayQueue = Queue  # noqa: F401


class ProgressRelay(ProgressProtocol):
    __slot__ = "queue"
    queue: RelayQueue
    _tasks: list[Any]

    def __init__(self, queue: RelayQueue):
        self.queue = queue
        self._tasks = []

    @property
    def tasks(self) -> list[Any]:
        return self._tasks

    @staticmethod
    def handle_message(progress: ProgressProtocol, message: Any) -> bool:
        """Handle ProgressRelayMessage if applicable. Returns True if consumed."""
        if not isinstance(message, ProgressRelayMessage):
            return False
        handler = getattr(progress, message.progress_method.value)
        handler(*message.progress_args, **message.progress_kwargs)
        return True

    def print(self, *objects: Any, **kw_args: Any) -> None:
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.print,
                progress_args=objects,
                progress_kwargs=kw_args,
            ),
        )

    def log(self, *objects: Any, **kw_args: Any) -> None:
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.log,
                progress_args=objects,
                progress_kwargs=kw_args,
            ),
        )

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        raise RuntimeError("Adding tasks via ProgressRelay is not supported.")

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
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.update,
                progress_args=(task_id,),
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

    def advance(self, task: int, n: float = 1.0) -> None:
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.advance,
                progress_args=(task, n),
                progress_kwargs={},
            ),
        )

    def remove_task(self, task_id: int) -> None:
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.remove_task,
                progress_args=(task_id,),
                progress_kwargs={},
            ),
        )
