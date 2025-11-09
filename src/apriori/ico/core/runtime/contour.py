from typing import Any

from typing_extensions import Self

from apriori.ico.core.dsl.operator import iterate_nodes
from apriori.ico.core.meta.ico_form import infer_ico_form
from apriori.ico.core.runtime.progress.mixin import ProgressMixin
from apriori.ico.core.runtime.progress.types import ProgressProtocol, SupportsProgress
from apriori.ico.core.runtime.runtime_operator import IcoRuntimeOperator
from apriori.ico.core.runtime.types import IcoRuntimeCommand
from apriori.ico.core.types import IcoOperatorProtocol


class IcoRuntimeContour(
    IcoRuntimeOperator[None, None],
    ProgressMixin,
):
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

    _closure: IcoOperatorProtocol[None, None]

    def __init__(
        self,
        closure: IcoOperatorProtocol[None, None],
        name: str | None = None,
    ) -> None:
        # Contour executes the given closure e.g. flow () → ()
        self._validate_closure(closure)

        super().__init__(fn=self._run_fn, name=name)

        self.connect_runtime(closure)
        self._closure = closure

    # ─── Execution ───

    def _run_fn(self, _: None) -> None:
        self._closure(None)

    def run(self) -> Self:
        """Execute the contour by calling itself."""
        self()
        return self

    # ─── Lifecycle ───

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.activate)
        return self

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.reset)
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.deactivate)
        return self

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.pause)
        return self

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.resume)
        return self

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        self.broadcast_command(IcoRuntimeCommand.stop)
        return self

    # ─── Progress ───

    def attach_progress(self, progress: ProgressProtocol) -> Self:
        """
        Bind a shared progress relay to all progress-capable nodes.

        Returns:
            Self — allows chaining: contour.bind_progress().ready().run().idle()
        """
        self.progress = progress

        for node in iterate_nodes(self._closure):
            if isinstance(node, SupportsProgress):
                node.progress = self.progress

        return self

    # ─── Internal utilities ───

    def _validate_closure(self, flow: IcoOperatorProtocol[Any, Any]) -> None:
        """Validate that the flow is a closure: begins and ends with unit types (() → ())."""
        form = infer_ico_form(flow)
        if not (form.i == "()" and form.o == "()"):
            raise ValueError(
                f"Invalid flow form: expected (() → ()), got ({form.i} → {form.o})"
            )
