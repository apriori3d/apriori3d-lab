from apriori.ico.core.dsl import Operator


def test_basic_composition() -> None:
    double = Operator(lambda x: x * 2)
    square = Operator(lambda x: x**2)

    composed = double >> square
    assert composed(3) == 36


def test_then_aliases_are_equivalent() -> None:
    inc = Operator(lambda x: x + 1)
    double = Operator(lambda x: x * 2)

    assert (inc >> double)(3) == 8
    assert (inc | double)(3) == 8
    assert inc.then(double)(3) == 8


def test_operator_is_callable() -> None:
    negate = Operator(lambda x: -x)
    assert negate(5) == -5
