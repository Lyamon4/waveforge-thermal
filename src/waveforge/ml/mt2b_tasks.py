"""Geometry-balanced procedural task stream for NCA-MT2B."""

from __future__ import annotations

from typing import Literal, TypeAlias

from waveforge.ml.multitask_tasks import SourceLayoutTask, sample_primary_task

GeometryStratum: TypeAlias = Literal[
    "compact", "wide_horizontal", "vertically_spread", "mixed"
]

HORIZONTAL_SPAN_THRESHOLD = 0.46
VERTICAL_SPAN_THRESHOLD = 0.21
_STRATA: tuple[GeometryStratum, ...] = (
    "compact",
    "wide_horizontal",
    "vertically_spread",
    "mixed",
)
_CANDIDATES_PER_STRATUM = 100_000


def classify_geometry(
    centers: tuple[tuple[float, float], ...],
) -> GeometryStratum:
    """Classify a three-source layout using only prospective geometry."""
    if len(centers) != 3:
        raise ValueError("geometry classification requires exactly three centers")
    horizontal_span = round(max(x for x, _ in centers) - min(x for x, _ in centers), 12)
    vertical_span = round(max(y for _, y in centers) - min(y for _, y in centers), 12)
    wide = horizontal_span >= HORIZONTAL_SPAN_THRESHOLD
    vertical = vertical_span >= VERTICAL_SPAN_THRESHOLD
    if wide and vertical:
        return "mixed"
    if wide:
        return "wide_horizontal"
    if vertical:
        return "vertically_spread"
    return "compact"


def balanced_task_batch(
    batch_index: int,
    *,
    seed: int,
    excluded_task_ids: frozenset[str],
) -> tuple[SourceLayoutTask, ...]:
    """Return one deterministic, non-leaking task from each locked stratum."""
    if isinstance(batch_index, bool) or not isinstance(batch_index, int):
        raise TypeError("batch_index must be an integer")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    selected: list[SourceLayoutTask] = []
    batch_base = batch_index * len(_STRATA) * _CANDIDATES_PER_STRATUM
    for stratum_index, stratum in enumerate(_STRATA):
        stratum_base = batch_base + stratum_index * _CANDIDATES_PER_STRATUM
        for retry in range(_CANDIDATES_PER_STRATUM):
            task = sample_primary_task(seed, stratum_base + retry)
            if task.task_id in excluded_task_ids:
                continue
            if task.task_id in {item.task_id for item in selected}:
                continue
            if classify_geometry(task.centers) == stratum:
                selected.append(task)
                break
        else:
            raise RuntimeError(f"no feasible task found for stratum {stratum}")
    return tuple(selected)
