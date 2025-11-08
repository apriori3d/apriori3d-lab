from __future__ import annotations

from typing import Any

from typing_extensions import Self

from apriori.flow.progress.noop import NoOpProgress
from apriori.flow.progress.types import ProgressProtocol
from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.meta.ico_form import infer_ico_form
from apriori.ico.core.runtime.progress import SupportsProgress
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import IcoOperatorProtocol
from apriori.ico.core.utils import iterate_children


class IcoRuntimeContour(IcoOperator[None, None]):
    """
    Runtime contour that encapsulates a complete ICO flow.

    ─── ICO Forms ───
        Contour: () → ()
        Flow:    (None → Iterable[O]) → ... → (Iterable[I] → None)

    ─── Description ───
    The contour wraps a full operator flow (called *Flow*) and provides a
    runtime layer responsible for:
        • broadcasting lifecycle events (prepare → reset → cleanup)
        • binding shared progress relays to operators
        • executing the entire flow (() → ())

    The flow itself must begin with a Source (None→Iterable[O])
    and terminate with a Sink (Iterable[O]→None), forming a closed runtime loop.

    ─── Example ───
        >>> from apriori.ico.core import IcoSource, IcoSink, IcoOperator, IcoRuntimeContour

        # Declarative flow: Source → Operator → Sink
        >>> source = IcoSource[int](lambda: range(3), name="dataset")
        >>> double = IcoOperator[int, int](lambda x: x * 2, name="double")
        >>> sink = IcoSink[int](lambda xs: print(list(xs)), name="printer")

        # Compose into a flow and wrap in contour (() → ())
        >>> flow = source | double.map() | sink
        >>> contour = IcoRuntimeContour(flow)

        >>> contour.ready().run().idle()
        [printer] Sink received: [0, 2, 4]

    """

    flow: IcoOperatorProtocol[None, None]
    progress: ProgressProtocol = NoOpProgress()

    def __init__(self, flow: IcoOperatorProtocol[None, None]) -> None:
        # Contour executes the given flow as () → ()
        self._validate_flow(flow)

        super().__init__(
            fn=lambda _: flow(None), name="RuntimeContour", children=[flow]
        )
        self.flow = flow

    # ─── Execution ───

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        self()
        return self

    # ─── Lifecycle ───

    def ready(self) -> Self:
        """Broadcast 'prepare' event through the entire flow."""
        return self.broadcast_event(IcoRuntimeCommand.activate)

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        return self.broadcast_event(IcoRuntimeCommand.reset)

    def idle(self) -> Self:
        """Broadcast 'cleanup' event through the entire flow."""
        return self.broadcast_event(IcoRuntimeCommand.deavtivate)

    def broadcast_event(self, event: IcoRuntimeCommand) -> Self:
        """Propagate lifecycle event recursively."""
        super().broadcast_event(event)
        return self

    # ─── Progress ───

    def bind_progress(self, progress: ProgressProtocol | None = None) -> Self:
        """
        Bind a shared progress relay to all progress-capable nodes.

        Returns:
            Self — allows chaining: contour.bind_progress().ready().run().idle()
        """
        if progress:
            self.progress = progress

        for node in iterate_children(self.flow):
            if isinstance(node, SupportsProgress):
                node.progress = self.progress
        return self

    # ─── Internal utilities ───

    def _validate_flow(self, flow: IcoOperatorProtocol[Any, Any]) -> None:
        """Validate that the flow begins and ends with unit types (() → ())."""
        form = infer_ico_form(flow)
        if not (form.i == "()" and form.o == "()"):
            raise ValueError(
                f"Invalid flow form: expected (() → ()), got ({form.i} → {form.o})"
            )
