from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar, final

from apriori.ico.core.pipeline import I, O

I2 = TypeVar("I2")
O2 = TypeVar("O2")


@final
@dataclass(slots=True)
class Operator(Generic[I, O]):
    """Composable callable I → O."""

    fn: Callable[[I], O]

    def __call__(self, item: I) -> O:
        return self.fn(item)

    # ─── Composition ───

    def compose(self, other: Operator[O, O2]) -> Operator[I, O2]:
        """Function composition: I→O, O→O2 → I→O2."""

        def composed(x: I) -> O2:
            return other(self(x))

        return Operator(composed)

    def then(self, other: Operator[O, O2]) -> Operator[I, O2]:
        """Alias for compose, improves readability."""
        return self.compose(other)

    __or__ = then
    __rshift__ = then

    # ─── Map ───

    def map(self) -> Operator[Iterable[I], Iterable[O]]:
        """Apply this operator elementwise over an iterable (lazy generator):
        Iterable[I] → Iterable[O]
        """
        return Operator(lambda xs: (self(x) for x in xs))


augment = Operator[float, float](lambda x: x * 2)
collate = Operator[Iterable[float], float](lambda x: max(x))
result1 = (augment.map() >> collate)([1.0, 5.0, 3.0])
print(result1)
