from typing import Protocol

from apriori.ico.core.types import I, O


class IcoAgentHostProtocol(Protocol[I, O]):
    async def run_async(self, item: I) -> O: ...
