from collections.abc import Iterable

from apriori.ico.core.data import IcoData
from apriori.ico.core.flow import IcoFlow
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.pipeline import IcoPipeline
from apriori.ico.core.stream import IcoStream

# ─────────────────────────────
# 1. Define data source
# ─────────────────────────────

data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]


def data_generator() -> Iterable[Iterable[float]]:
    yield from data


dataset = IcoData[Iterable[float]](data_generator, name="dataset")

# ─────────────────────────────
# 2. Define data processing elements
# a. Augmentation pipeline: I → O
# b. Collation pipeline: Iterable[O] → O2
# ─────────────────────────────

augment = IcoPipeline[float, float, float](
    context=IcoOperator(lambda x: x),
    body=[IcoOperator(lambda x: x * 2)],
    output=IcoOperator(lambda x: x),
)
collate = IcoPipeline[Iterable[float], Iterable[float], float](
    context=IcoOperator(list), body=[], output=IcoOperator(max)
)

# ─────────────────────────────
# 3. Compose elements into a complete data processing pipeline
# a. Map augmentation over input data: Iterable[I] → Iterable[O]
# b. Collate augmented results: Iterable[O] → O2
# ─────────────────────────────
pipeline = augment.map() >> collate

# ─────────────────────────────
# 4. Add runner layer
# a. Map the pipeline over batches of input data: Iterable[Iterable[I]] → Iterable[O2]
# ─────────────────────────────

runner = IcoStream[Iterable[float], float](pipeline)

# ─────────────────────────────
# 4. Execute the flow
# ─────────────────────────────
data_flow = dataset >> runner
result = data_flow(None)
result = list(result)
assert result == [6, 12, 18]  # Max of each batch after augmentation

# ─────────────────────────────
# 4. Retrieve and inspect structural description
# ─────────────────────────────
flow = IcoFlow.from_operator(data_flow)
print(flow.describe())
print("Done")
