from __future__ import annotations

from typing_extensions import Self

from apriori.ico.core.runtime.types import (
    IcoRuntimeCommand,
    IcoRuntimeLifecycleProtocol,
    IcoRuntimeProtocol,
)


class IcoRuntimeLifecycleMixin(IcoRuntimeLifecycleProtocol):
    __as_runtime: IcoRuntimeProtocol

    def __init__(self) -> None:
        super().__init__()
        if not isinstance(self, IcoRuntimeProtocol):
            raise TypeError(
                "IcoRuntimeLifecycleMixin can only be used with IcoRuntimeProtocol instances"
            )
        self.__as_runtime = self

    def activate(self) -> Self:
        """Broadcast 'activate' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.activate)
        return self

    def reset(self) -> Self:
        """Broadcast 'reset' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.reset)
        return self

    def deactivate(self) -> Self:
        """Broadcast 'deactivate' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.deactivate)
        return self

    def pause(self) -> Self:
        """Broadcast 'pause' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.pause)
        return self

    def resume(self) -> Self:
        """Broadcast 'resume' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.resume)
        return self

    def stop(self) -> Self:
        """Broadcast 'stop' event through the entire flow."""
        self.__as_runtime.on_command(IcoRuntimeCommand.stop)
        return self
