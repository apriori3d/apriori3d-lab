from typing import Any

from rich.progress import Progress


class LevelsProgress(Progress):
    def __init__(self, *args, **kw_args: Any):
        super().__init__(*args, **kw_args)
        self.level = 0
        self._prefixes: list[str] = []
        # Note: inaccurate design in rich.Progress - print() is not virtual
        self.print = self._print
        self._task_prefix = ""
        self._print_prefix = ""

    # Hierarchical levels methods

    def add_level(self, prefix: str | None = None):
        self.level += 1
        self._prefixes.append(prefix)
        self._update_task_prefix()
        self._update_print_prefix()

    def remove_level(self):
        self.level = max(0, self.level - 1)
        self._prefixes.pop()
        self._update_task_prefix()
        self._update_print_prefix()

    def update_current_prefix(self, prefix: str | None):
        if self.level == 0:
            raise RuntimeError("No levels to update prefix for.")
        self._prefixes[-1] = prefix
        self._update_task_prefix()
        self._update_print_prefix()

    # Overridden Progress methods

    def add_task(self, description: str, total: int, **fields: Any) -> int:
        return super().add_task(
            f"{self._task_prefix}{description}", total=total, **fields
        )

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
        if description is not None:
            description = f"{self._task_prefix}{description}"

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
        indent_objects = []
        for obj in objects:
            if isinstance(obj, str):
                indent_objects.append(
                    "\n".join([f"{self._print_prefix}{s}" for s in obj.split("\n")]),
                )
            else:
                indent_objects.append(obj)

        self.console.print(*indent_objects, **kw_args)

    def _update_print_prefix(self):
        self._print_prefix = "".join(p for p in self._prefixes if p is not None)

    def _update_task_prefix(self):
        self._task_prefix = f"{'  ' * (self.level - 1)}↳ " if self.level else ""
