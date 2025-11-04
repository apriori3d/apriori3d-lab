from apriori.ico.core.pipeline import Pipeline


def test_pipeline_execution_order() -> None:
    p = Pipeline[int, int, int](
        context=lambda x: x + 1,
        flow=[lambda x: x * 2, lambda x: x - 3],
        output=lambda x: x * 10,
    )
    assert p(2) == ((2 + 1) * 2 - 3) * 10  # 30


def test_pipeline_len_and_iter() -> None:
    p = Pipeline[int, int, int](
        context=lambda x: x,
        flow=[lambda x: x + 1, lambda x: x + 2],
        output=lambda x: x,
    )
    assert len(p) == 2
    assert list(p) == p.flow
