import multiprocessing
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
    progress_args: tuple
    progress_kwargs: dict


class ProgressRelay(ProgressProtocol):
    __slot__ = "queue"
    queue: multiprocessing.Queue

    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue

    @staticmethod
    def relay_to(progress: ProgressProtocol, message: Any) -> bool:
        if not isinstance(message, ProgressRelayMessage):
            return False
        method = getattr(progress, message.progress_method.value)
        method(*message.progress_args, **message.progress_kwargs)
        return True

    def print(self, *objects: Any, **kw_args: Any):
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.print,
                progress_args=objects,
                progress_kwargs=kw_args,
            ),
        )

    def log(self, *objects, **kw_args):
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.log,
                progress_args=objects,
                progress_kwargs=kw_args,
            ),
        )

    def add_task(self, description, total, **fields):
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.add_task,
                progress_args=(description, total),
                progress_kwargs=fields,
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

    def advance(self, task: int, n: float = 1.0):
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.advance,
                progress_args=(task, n),
                progress_kwargs={},
            ),
        )

    def remove_task(self, task_id: int):
        self.queue.put(
            ProgressRelayMessage(
                progress_method=ProgressRelayMethod.remove_task,
                progress_args=(task_id,),
                progress_kwargs={},
            ),
        )
