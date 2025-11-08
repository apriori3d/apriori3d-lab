from __future__ import annotations

import asyncio
from typing import Generic

from apriori.ico.core.runtime.execution import IcoExecutionMixin, IcoExecutionState
from apriori.ico.core.runtime.types import IcoRuntimeMixin
from apriori.ico.core.types import I, IcoOperatorProtocol, O


class IcoAgent(
    IcoOperatorProtocol[I, O],
    Generic[I, O],
    IcoRuntimeMixin,  # Added lifecycle management
    IcoExecutionMixin[I, O],  # Added execution state tracking
):
    async def run_async(self, item: I) -> O:
        raise NotImplementedError("Async run_async method must be implemented.")

    host: IcoOperatorProtocol[I, O]

    def __init__(self, host: IcoOperatorProtocol[I, O]) -> None:
        super().__init__()
        self.host = host

    def __call__(self, item: I) -> O:
        self._set_exec_state(IcoExecutionState.running)
        try:
            output = asyncio.run(self.run_async(item))
        except Exception as e:
            self._set_exec_state(IcoExecutionState.faulted)
            raise e
        else:
            self._set_exec_state(IcoExecutionState.done)
            return output
