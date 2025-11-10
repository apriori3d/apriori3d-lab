from collections.abc import Iterator
from typing import Generic, final

from apriori.ico.core.dsl.operator import IcoOperator
from apriori.ico.core.runtime.agent_link import IcoAgentLinkProtocol
from apriori.ico.core.runtime.types import IcoRuntimeHost, IcoRuntimeProtocol
from apriori.ico.core.types import I, O


@final
class IcoPortal(
    Generic[I, O],
    IcoOperator[Iterator[I], Iterator[O]],
    IcoRuntimeHost,
):
    _agent_link: IcoAgentLinkProtocol[I, O]

    def __init__(
        self,
        agent_link: IcoAgentLinkProtocol[I, O],
        name: str | None = None,
    ) -> None:
        super().__init__(self._portal_fn, name=name)
        self._agent_link = agent_link

    def _portal_fn(self, items: Iterator[I]) -> Iterator[O]:
        for item in items:
            yield self._agent_link.output_channel.receive(
                self._agent_link.input_channel.send(item)
            )

    @property
    def runtime(self) -> IcoRuntimeProtocol:
        return self._agent_link
