# Lifecyycle protocol
from typing import Protocol


class HasLifecycle(Protocol):
    def prepare(self) -> None: ...
    def on_cycle_start(self) -> None: ...
    def on_cycle_end(self) -> None: ...
    def cleanup(self) -> None: ...


class LifecycleMixin:
    __slots__ = ("_prepared",)
    _prepared: bool

    def __init__(self) -> None:
        self._prepared = False

    @property
    def prepared(self) -> bool:
        return self._prepared

    # Lifecycle methods

    def prepare(self) -> None:
        pass

    def on_cycle_start(self) -> None:
        pass

    def on_cycle_end(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
