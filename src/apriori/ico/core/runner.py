from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, final

from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.types import I, IcoOperatorProtocol, NodeType, O

# ──── Runner  ────


@final
class IcoRunner(
    IcoOperator[Iterable[I], Iterable[O]],
    IcoOperatorProtocol[Iterable[I], Iterable[O]],
    Generic[I, O],
):
    """
    A higher-order operator that maps a body operator over an iterable of inputs,
    following the ICO convention:

        Iterable[I] → Iterable[O]
            body: I → O

    The body itself can be any operator (e.g., a Pipeline, another Runner, etc.),
    enabling nested execution graphs.

    Example:
        runner = IcoRunner(IcoPipeline(...))
        results = list(runner(dataset))
    """

    __slots__ = ("body",)

    body: IcoOperatorProtocol[I, O]

    def __init__(
        self,
        body: IcoOperatorProtocol[I, O],
    ):
        def runner_fn(batch: Iterable[I]) -> Iterable[O]:
            for item in batch:
                yield self.body(item)

        super().__init__(
            fn=runner_fn,
            node_type=NodeType.runner,
            children=[body],
        )
        self.body = body
