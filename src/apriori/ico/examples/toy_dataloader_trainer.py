from __future__ import annotations

from collections.abc import Iterable

from rich.console import Console

from apriori.ico.core import (
    IcoDescriber,
    IcoFlowMeta,
    IcoOperator,
    IcoPipeline,
    IcoSource,
    IcoStream,
)


# ──── 1. Define a batched data source ────
def generate_batches() -> Iterable[list[float]]:
    """Simulate dataset batches: () → Iterable[list[float]]"""
    return [
        [0.5, 1.0, 1.5],
        [2.0, 0.8, 0.2],
        [1.0, 1.2, 0.9],
    ]


dataset = IcoSource[list[float]](generate_batches, name="dataset")


# ──── 2. Define augmentation & collation pipelines ────

augment = IcoPipeline[float, float, float](
    context=IcoOperator[float, float](lambda x: x, name="identity_ctx"),
    body=[
        IcoOperator[float, float](lambda x: x * 1.1, name="scale_up"),
        IcoOperator[float, float](lambda x: x + 0.1, name="shift"),
    ],
    output=IcoOperator[float, float](lambda x: x, name="identity_out"),
    name="augment_pipeline",
)

collate = IcoPipeline[Iterable[float], Iterable[float], float](
    context=IcoOperator[Iterable[float], Iterable[float]](list, name="to_list"),
    body=[],
    output=IcoOperator[Iterable[float], float](max, name="max_value"),
    name="collate_pipeline",
)


# ──── 3. Compose into a full data stream ────
# dataset: () → Iterable[Iterable[float]]
# stream: Iterable[Iterable[float]] → Iterable[float]
# augment.map(): Iterable[float] → Iterable[float]
# augment: float → float → float,
# collate: Iterable[float] → Iterable[float] → float
# dataflow: () → Iterable[float]

dataflow = dataset | IcoStream(augment.map() | collate, name="data_stream")


# ──── 4. Define training pipeline ────
# We collate batch into single float via max, so input into training stream is Iterable[float],
# without batch dimension. In real scenarios, item could be tensors with batch dim.
# Each batch item passes through a process that transforms floats ≤ 1 via pow(2)
# train_step: float -> float, ICO Form: I → O
# train_process: float -> float, ICO Form: C → C
# I → C → O = Iterable[float] → Iterable[float] → Iterable[float]


def pow_if_needed(values: float) -> float:
    return values**2 if values <= 1.0 else values


train_step = IcoOperator[float, float](pow_if_needed, name="train_step")

train_pipeline = IcoPipeline[float, float, float](
    context=IcoOperator[float, float](lambda xs: xs, name="identity_ctx"),
    body=[train_step],
    output=IcoOperator[float, float](lambda xs: xs, name="identity_out"),
    name="train_pipeline",
)

train_stream = IcoStream(train_pipeline, name="train_stream")


# ──── 5. Combine all into the full flow ────
full_flow = dataflow | train_stream


# ──── 6. Visualize ────
flow_meta = IcoFlowMeta.from_operator(full_flow)

console = Console()
console.rule("[bold blue]ICO Dataflow: Dataset → Stream → Train")
console.print(IcoDescriber.describe(flow_meta, show_states=False, show_ico_form=True))
