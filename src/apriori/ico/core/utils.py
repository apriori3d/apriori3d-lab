# ─── Utility methods ───

from typing import Any, get_args, get_type_hints

from apriori.ico.core.types import IcoOperatorProtocol


def infer_ico_signature(
    operator: IcoOperatorProtocol[Any, Any],
) -> tuple[str, str | None, str]:
    """Try to infer (I, C, O) type names from generic or callable hints."""
    # Try __orig_class__ first
    try:
        args = get_args(getattr(operator, "__orig_class__", None))
        if args:
            if len(args) == 3:
                # (I, C, O)
                return args[0].__name__, args[1].__name__, args[2].__name__
            elif len(args) == 2:
                # (I, O)
                return args[0].__name__, None, args[1].__name__
    except AttributeError:
        pass

    # Fallback to fn type hints
    hints = get_type_hints(operator.fn)
    input_type = next(iter(hints.values()), Any)
    output_type = hints.get("return", Any)

    return (
        getattr(input_type, "__name__", "Any"),
        None,
        getattr(output_type, "__name__", "Any"),
    )
