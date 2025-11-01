from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class StructureElementType(Enum):
    runner = auto()
    executor = auto()
    pool = auto()
    input_step = auto()
    step = auto()
    output_step = auto()


@dataclass
class FlowStructure:
    name: str
    type: StructureElementType
    children: list["FlowStructure"] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


class HasFlowStructure(Protocol):
    def describe_structure(self) -> FlowStructure: ...


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
