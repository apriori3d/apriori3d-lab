from collections.abc import Sized

from apriori.flow.runner.types import RunnerInputType


def get_input_len_or_zero(input: RunnerInputType) -> int:
    """Utility function to get the length of the runner input."""

    if isinstance(input, Sized):
        return len(input)

    return 0
