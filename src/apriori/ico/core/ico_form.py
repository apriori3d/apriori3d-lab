from __future__ import annotations

from dataclasses import dataclass
from typing import Any, final, get_args, get_type_hints

from apriori.ico.core.types import IcoOperatorProtocol, NodeType

# ─── IcoForm dataclass ───


@final
@dataclass(slots=True)
class IcoForm:
    i: str
    c: str | None
    o: str

    @property
    def name(self) -> str:
        if self.i is None and self.c is None:
            # Source node
            return f"() → {self.o}"
        if self.c is None:
            # Operator node
            return f"{self.i} → {self.o}"
        else:
            # Pipeline node
            return f"{self.i} → {self.c} → {self.o}"

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_operator(operator: IcoOperatorProtocol[Any, Any]) -> IcoForm:
        """Infer IcoForm from an operator."""
        return infer_ico_form(operator)


# ──── Factory method ────


def infer_ico_form(operator: IcoOperatorProtocol[Any, Any]) -> IcoForm:
    """
    Infer (I, C, O) form of an ICO operator using:
      1. Generic annotations (`__orig_class__`)
      2. Function type hints
      3. Fallback to ("Any", None, "Any")
    """
    node_type = operator.node_type
    args = get_args(getattr(operator, "__orig_class__", None))

    # ──── Match structural node type ────
    match node_type:
        case NodeType.operator:
            # Example: IcoOperator[int, float] is int → float
            if args and len(args) == 2:
                i_name, o_name = (getattr(a, "__name__", str(a)) for a in args)
                return IcoForm(i_name, None, o_name)

        case NodeType.compose:
            # For a composition, infer form from first and last children
            if len(operator.children) >= 2:
                first = infer_ico_form(operator.children[0])
                last = infer_ico_form(operator.children[-1])
                return IcoForm(first.i, None, last.o)

        case NodeType.map | NodeType.stream:
            # Example: IcoStream[float, float] is Iterable[float] → Iterable[float]
            child = operator.children[0] if len(operator.children) > 0 else None
            inner = infer_ico_form(child) if child else IcoForm("Any", None, "Any")
            return IcoForm(f"Iterable[{inner.i}]", None, f"Iterable[{inner.o}]")

        case NodeType.pipeline:
            # Example: IcoPipeline[int, float, str] is int → float → str
            if args and len(args) == 3:
                i, c, o = (getattr(a, "__name__", str(a)) for a in args)
                return IcoForm(i, c, o)

        case NodeType.process:
            # Example: IcoProcess[float] is float → float
            if args and len(args) == 1:
                c = getattr(args[0], "__name__", str(args[0]))
                return IcoForm(c, None, c)

        case NodeType.source:
            # Example: IcoSource[float] is () → Iterable[float]
            if args:
                o_name = getattr(args[0], "__name__", str(args[0]))
                return IcoForm("()", None, f"Iterable[{o_name}]")

    # ──── Fallback to function type hints ────
    try:
        hints = get_type_hints(operator.fn)
        input_type = next(iter(hints.values()), Any)
        output_type = hints.get("return", Any)
        return IcoForm(
            getattr(input_type, "__name__", "Any"),
            None,
            getattr(output_type, "__name__", "Any"),
        )
    except Exception:
        return IcoForm("Any", None, "Any")
