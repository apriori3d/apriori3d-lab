from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from apriori.flow.progress.rich.live_progress import LiveProgress
from apriori.flow.progress.rich.tree_progress import TreeProgress


def create_progress(
    description: str | None = None,
    disable: bool = False,
    add_console: bool = True,
    show_remaining_time: bool = True,
) -> TreeProgress:
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

    return TreeProgress(
        *columns,
        console=console,
        disable=disable,
    )


def create_live_progress(
    description: str | None = None,
    disable: bool = False,
    add_console: bool = True,
    show_remaining_time: bool = True,
) -> LiveProgress:
    return LiveProgress(
        create_progress(
            description=description,
            disable=disable,
            add_console=add_console,
            show_remaining_time=show_remaining_time,
        ),
    )
