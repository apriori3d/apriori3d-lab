from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline


def test_pipeline_execution_order() -> None:
    p = IcoPipeline[int, int, int](
        context=IcoOperator(lambda x: x + 1),
        flow=[IcoOperator(lambda x: x * 2), IcoOperator(lambda x: x - 3)],
        output=IcoOperator(lambda x: x * 10),
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_len_and_iter() -> None:
    p = IcoPipeline[int, int, int](
        context=IcoOperator(lambda x: x + 1),
        flow=[IcoOperator(lambda x: x + 1), IcoOperator(lambda x: x + 2)],
        output=IcoOperator(lambda x: x),
    )
    assert len(p) == 2
    assert list(p) == p.flow
