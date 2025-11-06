from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import ProgressProtocol


class ProgressMixin:
    progress: ProgressProtocol

    def __init__(self) -> None:
        self.progress = NoOpProgress()
