from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from apriori.ico.core.execution import IcoExecutionState
from apriori.ico.core.flow_meta import IcoFlowMeta
from apriori.ico.core.lifecycle import IcoLifecycleState


class IcoDescriber:
    """Render an IcoFlow structure as a Rich tree."""

    @staticmethod
    def describe(
        flow: IcoFlowMeta,
        *,
        show_states: bool = True,
        show_ico_form: bool = False,
    ) -> Tree:
        """Render the given IcoFlow as a rich tree."""
        return IcoDescriber._build_node(flow, show_states, show_ico_form)

    @staticmethod
    def _build_node(
        flow: IcoFlowMeta,
        show_states: bool,
        show_ico_form: bool,
    ) -> Tree:
        label = IcoDescriber._format_label(flow, show_states, show_ico_form)
        node = Tree(label)
        for child in flow.children:
            node.add(IcoDescriber._build_node(child, show_states, show_ico_form))
        return node

    @staticmethod
    def _format_label(
        flow: IcoFlowMeta, show_states: bool, show_ico_form: bool
    ) -> Text:
        text = Text(flow.name or flow.node_type.name, style="bold cyan")
        text.append(f" ({flow.node_type.name})", style="dim")

        if show_ico_form:
            text.append(f" {flow.signature}", style="magenta")

        # Если flow хранит runtime-данные — показать их
        if show_states:
            lifecycle_state = getattr(flow, "lifecycle_state", None)
            exec_state = getattr(flow, "exec_state", None)

            if lifecycle_state:
                color = {
                    IcoLifecycleState.prepared: "yellow",
                    IcoLifecycleState.ready: "green",
                    IcoLifecycleState.cleaned: "grey70",
                    IcoLifecycleState.unknown: "grey50",
                }.get(lifecycle_state, "white")
                text.append(f" [{lifecycle_state.name}]", style=color)

            if exec_state:
                color = {
                    IcoExecutionState.idle: "grey50",
                    IcoExecutionState.running: "blue",
                    IcoExecutionState.done: "green",
                    IcoExecutionState.faulted: "red",
                }.get(exec_state, "white")
                text.append(f" <{exec_state.name}>", style=color)

        return text


if __name__ == "__main__":
    from apriori.ico.core import IcoPipeline, IcoProcess, IcoSource

    fib = IcoProcess(lambda c: (c[1], c[0] + c[1]), num_iterations=5)
    pipeline = IcoPipeline[int, tuple[int, int], int](
        context=lambda _: (0, 1),
        body=[fib],
        output=lambda c: c[1],
    )

    dataset = IcoSource(lambda: range(3))

    # Преобразуем в IcoFlow
    flow_root = IcoFlowMeta.from_operator(pipeline)

    console = Console()
    console.print(IcoDescriber.describe(flow_root, show_ico_form=True))
