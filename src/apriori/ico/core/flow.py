from __future__ import annotations

from collections.abc import Iterator
from typing import Any, final

from apriori.ico.core.types import IcoOperatorProtocol, NodeType
from apriori.ico.core.utils import infer_ico_signature


@final
class IcoFlow:
    """Hierarchical description of the ICO computational flow."""

    name: str
    node_type: NodeType
    children: list[IcoFlow]
    ico_signature: tuple[str, str | None, str]

    def __init__(
        self,
        name: str | None,
        node_type: NodeType,
        ico_signature: tuple[str, str | None, str],
        children: list[IcoFlow] | None = None,
    ):
        self.node_type = node_type
        self.ico_signature = ico_signature
        self.name = name or self.signature
        self.children = children if children is not None else []

    def __str__(self) -> str:
        return self.name

    def describe(
        self,
        prefix: str = "",
        is_last: bool = True,
        is_root: bool = True,
    ) -> str:
        """Recursively build an ASCII tree representation of the operator structure."""
        # Choose the correct branch symbol for the current node
        # If it's the root node, no branch symbol is needed
        branch = "" if is_root else ("└──" if is_last else "├──")
        s = f"{prefix}{branch}{self.name}"

        # Prepare prefix for children:
        # Root has no prefix, vertical line continues if not last
        if is_root:
            child_prefix = ""
        elif is_last:
            child_prefix = prefix + "   "
        else:
            child_prefix = prefix + "│  "

        count = 1

        # Recurse into children
        for i, child in enumerate(self.children):
            is_child_last = i == len(self.children) - 1
            child_desc = child.describe(child_prefix, is_child_last, is_root=False)
            s += "\n" + child_desc
            count += child_desc.count("\n") + 1

        return s + f"\nTotal nodes: {count}" if is_root else s

    def traverse(self) -> Iterator[IcoFlow]:
        yield self
        for c in self.children:
            yield from c.traverse()

    @property
    def signature(self) -> str:
        i, c, o = self.ico_signature
        if c is None:
            return f"{self.node_type.name}[{i} → {o}]"
        else:
            return f"{self.node_type.name}[{i} → {c} → {o}]"

    # ─── Factory helpers ───

    @staticmethod
    def from_operator(operator: IcoOperatorProtocol[Any, Any]) -> IcoFlow:
        """Recursively build an IcoFlow from an operator tree."""
        return IcoFlow(
            name=operator.name,
            node_type=operator.node_type,
            ico_signature=infer_ico_signature(operator),
            children=[IcoFlow.from_operator(c) for c in operator.children],
        )


# @staticmethod
# def _print_runner_plan(runner: "Runner", prefix="") -> None:
#     # Print runner with input and pipeline info.

#     if prefix == "":
#         # Top-level runner
#         runner_prefix = prefix
#     else:
#         # Nested runner
#         runner_prefix = f"{prefix}└──"
#         prefix = f"{prefix}   "
#     runner.progress.print(f"{runner_prefix}{repr(runner)}")

#     # Print runner input and pipeline as nodes.
#     runner.progress.print(f"{prefix}├──{repr(runner.input)}")
#     runner.progress.print(f"{prefix}└──{repr(runner.pipeline_executor.pipeline)}")

#     # Add indentation for pipeline steps.
#     prefix = f"{prefix}   "
#     # Print pipeline steps.
#     runner.progress.print(
#         f"{prefix}├──{repr(runner.pipeline_executor.pipeline.input_step)}"
#     )

#     for step in runner.pipeline_executor.pipeline.steps:
#         if isinstance(step, HasRunner) and isinstance(step.runner, Runner):
#             runner.progress.print(f"{prefix}├──{repr(step)}")
#             # Print runner plan recursively with increased indentation
#             Runner._print_runner_plan(step.runner, prefix=f"{prefix}│  ")
#         else:
#             runner.progress.print(f"{prefix}├──{repr(step)}")

#     runner.progress.print(
#         f"{prefix}└──{repr(runner.pipeline_executor.pipeline.output_step)}"
# #     )

#  def print_plan(self) -> None:
#         self._prepare(self)
#         self._print_runner_plan(self)
