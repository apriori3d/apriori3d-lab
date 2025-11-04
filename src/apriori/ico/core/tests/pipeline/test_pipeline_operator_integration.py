from collections.abc import Iterable

from apriori.ico.core.dsl import Operator
from apriori.ico.core.pipeline import Pipeline


def test_operator_wraps_pipeline() -> None:
    # Basic pipeline: float → float
    p = Pipeline[float, float, float](
        context=lambda x: x + 1,
        flow=[lambda x: x * 2, lambda x: x + 3],
        output=lambda x: round(x, 2),
    )

    op = Operator(p)

    # Direct call
    assert op(1.0) == (1 + 1) * 2 + 3  # 7

    # Composition with another operator
    normalize = Operator[float, float](lambda x: x / 10)
    composed = op >> normalize
    assert composed(1.0) == 0.7


def test_pipeline_inside_map_operator() -> None:
    # Define a small pipeline that squares a number
    square_pipeline = Pipeline[int, int, int](
        context=lambda x: x,
        flow=[lambda x: x * x],
        output=lambda x: x,
    )

    square_op = Operator(square_pipeline)
    total_op = Operator[Iterable[int], int](sum)

    # Apply map() and reduce-like composition
    pipeline = square_op.map() >> total_op
    result = pipeline([1, 2, 3])
    assert result == 14  # 1² + 2² + 3²


def test_nested_pipeline_composition() -> None:
    # First pipeline: scale and shift
    p1 = Pipeline[int, int, int](
        context=lambda x: x + 1,
        flow=[lambda x: x * 3],
        output=lambda x: x,
    )

    # Second pipeline: convert to string
    p2 = Pipeline[int, str, str](
        context=lambda x: f"[{x}]",
        flow=[lambda s: s + "!"],
        output=lambda s: s,
    )

    composed = Operator(p1) >> Operator(p2)
    assert composed(4) == "[15]!"
