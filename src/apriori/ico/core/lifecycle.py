# Lifecycle protocol
from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsLifecycle(Protocol):
    def prepare(self) -> None:
        """Prepare the component before execution.

        This may include resource allocation, initialization,
        or any setup required for the component to function correctly.
        """
        ...

    def on_cycle_start(self) -> None:
        """Signal the start of a processing cycle.

        Cycle definition depends on the context:
        - For runners: one full iteration over the input set.
        - For pipeline executors: processing a single input item.
        - For pipeline steps: a single call with a context.
        """
        ...

    def on_cycle_end(self) -> None:
        """Called at the end of a cycle for post-processing or finalization."""
        ...

    def cleanup(self) -> None:
        """Release resources allocated during prepare or execution."""
        ...


class LifecycleMixin:
    """Mixin class providing default no-op implementations of lifecycle methods."""

    _prepared: bool

    def __init__(self) -> None:
        self._prepared = False

    def prepare(self) -> None:
        pass

    def on_cycle_start(self) -> None:
        pass

    def on_cycle_end(self) -> None:
        pass

    def cleanup(self) -> None:
        pass
