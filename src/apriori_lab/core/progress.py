from typing import Protocol

from typing_extensions import Self


class ProgressProtocol(Protocol):
    def add_task(self, description: str, total: int) -> int: ...
    def advance(self, task_id: int, advance: int = 1) -> None: ...
    def print(self, message: str) -> None: ...
    def log(self, message: str) -> None: ...


class HierarchicalProgress(ProgressProtocol):
    def __init__(self, progress: ProgressProtocol):
        self.progress = progress
        self.level = 0

    def add_level(self):
        self.level += 1

    def remove_level(self):
        self.level = max(0, self.level - 1)

    def indent(self):
        return

    def add_task(self, description: str, total: int, **kw_args) -> int:
        indent = f"{'  ' * (self.level - 1)}↳ " if self.level else ""
        return self.progress.add_task(f"{indent}{description}", total=total, **kw_args)

    def advance(self, task_id: int, advance: int = 1) -> None:
        self.progress.advance(task_id, advance=advance)

    def print(self, message: str) -> None:
        indent = "  " * (self.level) if self.level else ""
        message = "\n".join([f"{indent}{s}" for s in message.split("\n")])
        self.progress.print(message)

    def log(self, message: str) -> None:
        self.progress.log(message)

    def __enter__(self) -> Self:
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.progress.__exit__(exc_type, exc_val, exc_tb)


class ConsoleProgress(ProgressProtocol):
    def add_task(self, description: str, total: int) -> int:
        return 0

    def advance(self, task_id: int, advance: int = 1) -> None:
        pass

    def print(self, message: str) -> None:
        print(message)

    def log(self, message: str) -> None:
        print(message)
