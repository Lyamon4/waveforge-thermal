from __future__ import annotations

import pytest

from waveforge.ml.mt2b_tasks import (
    GeometryStratum,
    balanced_task_batch,
    classify_geometry,
)


@pytest.mark.parametrize(
    ("centers", "expected"),
    [
        (((0.20, 0.55), (0.40, 0.60), (0.659, 0.759)), "compact"),
        (((0.20, 0.55), (0.40, 0.60), (0.66, 0.759)), "wide_horizontal"),
        (((0.20, 0.55), (0.40, 0.60), (0.659, 0.76)), "vertically_spread"),
        (((0.20, 0.55), (0.40, 0.60), (0.66, 0.76)), "mixed"),
    ],
)
def test_geometry_strata_use_exact_locked_boundaries(
    centers: tuple[tuple[float, float], ...], expected: GeometryStratum
) -> None:
    assert classify_geometry(centers) == expected


def test_balanced_batch_contains_one_task_from_each_stratum() -> None:
    tasks = balanced_task_batch(0, seed=2026092201, excluded_task_ids=frozenset())

    assert len(tasks) == 4
    assert [classify_geometry(task.centers) for task in tasks] == [
        "compact",
        "wide_horizontal",
        "vertically_spread",
        "mixed",
    ]
    assert len({task.task_id for task in tasks}) == 4


def test_balanced_batches_are_reproducible_and_change_by_batch_index() -> None:
    first = balanced_task_batch(12, seed=2026092201, excluded_task_ids=frozenset())
    repeated = balanced_task_batch(12, seed=2026092201, excluded_task_ids=frozenset())
    next_batch = balanced_task_batch(13, seed=2026092201, excluded_task_ids=frozenset())

    assert [task.task_id for task in first] == [task.task_id for task in repeated]
    assert [task.task_id for task in first] != [task.task_id for task in next_batch]


def test_balanced_batch_rejects_excluded_ids_without_outcome_data() -> None:
    original = balanced_task_batch(4, seed=2026092201, excluded_task_ids=frozenset())
    excluded = frozenset(task.task_id for task in original)
    replacement = balanced_task_batch(
        4,
        seed=2026092201,
        excluded_task_ids=excluded,
    )

    assert excluded.isdisjoint(task.task_id for task in replacement)
    assert [classify_geometry(task.centers) for task in replacement] == [
        "compact",
        "wide_horizontal",
        "vertically_spread",
        "mixed",
    ]


@pytest.mark.parametrize("batch_index", [-1, True])
def test_balanced_batch_rejects_invalid_batch_index(batch_index: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        balanced_task_batch(
            batch_index,  # type: ignore[arg-type]
            seed=2026092201,
            excluded_task_ids=frozenset(),
        )
