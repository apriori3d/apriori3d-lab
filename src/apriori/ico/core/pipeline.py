from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

# ──── Generic type variables for ICO Pipeline ────

I = TypeVar("I")  # noqa: E741
C = TypeVar("C")
O = TypeVar("O")  # noqa: E741

# ──── Pipeline definition ────


@dataclass(slots=True)
class Pipeline(Generic[I, C, O]):
    """
    Immutable description of a transformation pipeline following ICO pattern (Input ─▶ Context ─▶ Output):
      context: I → C
      flow:   [C → C]
      out:     C → O
    Each component can be any callable (function or object implementing __call__).
    """

    context: Callable[[I], C]
    flow: Sequence[Callable[[C], C]]
    output: Callable[[C], O]

    def __len__(self) -> int:
        return len(self.flow)

    def __iter__(self) -> Iterator[Callable[[C], C]]:
        yield from self.flow

    def __call__(self, item: I) -> O:
        """Execute the pipeline on a single input."""
        context = self.context(item)
        for step in self.flow:
            context = step(context)
        return self.output(context)
