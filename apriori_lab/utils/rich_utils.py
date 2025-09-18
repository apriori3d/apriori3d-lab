import time
from collections import defaultdict, deque
from functools import partial

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text


class IterationTimeColumn(ProgressColumn):
    def __init__(self, history_len=10):
        super().__init__()
        self.last_update_time = {}
        self.last_format_in_ms = {}
        self.delta_time_history = defaultdict(partial(deque, maxlen=history_len))

    def render(self, task: Task) -> Text:
        now = time.perf_counter()
        last_time = self.last_update_time.get(task.id, now)
        self.last_update_time[task.id] = now
        delta = now - last_time

        task_delta_time = self.delta_time_history[task.id]
        task_delta_time.append(delta)
        avg_delta_time = sum(task_delta_time) / len(task_delta_time)

        # Disable jitter in format type by adding hysteresis threshold
        last_format_in_ms = self.last_format_in_ms.get(task.id, None)
        extent = 0.1 if last_format_in_ms else 0

        if avg_delta_time >= 0.1 + extent:
            time_str = f"{avg_delta_time:.1f}s/iter"
            self.last_format_in_ms[task.id] = False
        else:
            time_str = f"{int(avg_delta_time * 1000)}ms/iter"
            self.last_format_in_ms[task.id] = True

        return Text(time_str, style="progress.remaining")


def get_progress(
    description: str | None = None,
    disable: bool = False,
    add_console: bool = True,
) -> Progress:
    console = Console(width=80) if add_console else None
    description = description or "[progress.description]{task.description}"

    progress = Progress(
        TextColumn(description),
        BarColumn(
            style="white",
            complete_style="bright_blue",
        ),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        IterationTimeColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=disable,
    )
    return progress
