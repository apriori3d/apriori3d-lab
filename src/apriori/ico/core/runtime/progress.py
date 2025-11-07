from typing import Protocol, runtime_checkable

from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import ProgressProtocol


@runtime_checkable
class HasProgress(Protocol):
    progress: ProgressProtocol


class ProgressMixin:
    progress: ProgressProtocol

    def __init__(self) -> None:
        self.progress = NoOpProgress()
