from typing import Any

from ai_vision.pipelines.body3d.core.progress.types import LiveProgress
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class EasyProgress(Progress):
    def __init__(self, *args: Any, **kw_args: Any):
        super().__init__(*args, **kw_args)
        self.print = self._print

    def _print(self, *objects: Any, **kw_args: Any) -> None:
        self.console.print(*objects, **kw_args)


class HierarchicalProgress(Progress):
    def __init__(self, *args, **kw_args: Any):
        super().__init__(*args, **kw_args)
        self.level = 0
        # Note: inaccurate design in rich.Progress - print() is not virtual
        self.print = self._print

    def add_level(self):
        self.level += 1

    def remove_level(self):
        self.level = max(0, self.level - 1)

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        indent = f"{'  ' * (self.level - 1)}↳ " if self.level else ""
        return super().add_task(f"{indent}{description}", total=total, **fields)

    def update(
        self,
        task_id: int,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,
    ) -> None:
        if description is not None and self.level:
            indent = f"{'  ' * (self.level - 1)}↳ "
            description = f"{indent}{description}"

        super().update(
            task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

    def _print(self, *objects: Any, **kw_args: Any) -> None:
        indent = "  " * (self.level) if self.level else ""
        ident_objects = []
        for obj in objects:
            if isinstance(obj, str):
                ident_objects.append(
                    "\n".join([f"{indent}{s}" for s in obj.split("\n")]),
                )
            else:
                ident_objects.append(obj)

        self.console.print(*ident_objects, **kw_args)


class PrefixProgress(HierarchicalProgress):
    def __init__(self, *args: Any, **kw_args: Any):
        super().__init__(*args, **kw_args)
        self.prefix = None

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        return super().add_task(
            description if self.prefix is None else f"{self.prefix}{description}",
            total,
            **fields,
        )

    def _print(self, *args: Any, **kw_args: Any) -> None:
        if self.prefix is not None and len(args) == 1 and isinstance(args[0], str):
            # Single string argument - add prefix with frame index
            super()._print(f"{self.prefix}{args[0]}")
        else:
            # Multiple or non-string arguments - print as is
            super()._print(*args, **kw_args)


def get_progress(
    description: str | None = None,
    disable: bool = False,
    add_console: bool = True,
    show_remaining_time: bool = True,
) -> LiveProgress:
    console = Console(width=120) if add_console else None
    description = description or "[progress.description]{task.description}"

    columns = [
        TextColumn(description),
        BarColumn(
            style="white",
            complete_style="bright_blue",
            bar_width=None,
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
    ]
    if show_remaining_time:
        columns.append(TimeRemainingColumn())

    return LiveProgress(
        PrefixProgress(
            *columns,
            console=console,
            disable=disable,
        ),
    )
