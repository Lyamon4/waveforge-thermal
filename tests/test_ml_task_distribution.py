from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from waveforge.ml.task_distribution import (
    CENTER_X_UNITS,
    CENTER_Y_UNITS,
    build_task_registry,
    eligible_layouts,
    write_task_registry,
)


def _squared_distance_units(
    left: tuple[int, int],
    right: tuple[int, int],
) -> int:
    return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2


def test_eligible_layouts_use_exact_center_grid_and_separation() -> None:
    train_pool = eligible_layouts(held_out=False)
    test_pool = eligible_layouts(held_out=True)

    assert CENTER_X_UNITS == (2, 3, 4, 5, 6, 7, 8)
    assert CENTER_Y_UNITS == (6, 7, 8)
    assert train_pool
    assert test_pool
    assert len(set(train_pool)) == len(train_pool)
    assert len(set(test_pool)) == len(test_pool)
    assert set(train_pool).isdisjoint(test_pool)

    for layout in (*train_pool, *test_pool):
        assert layout == tuple(sorted(layout))
        assert all(
            _squared_distance_units(left, right) >= 4
            for left, right in itertools.combinations(layout, 2)
        )
    assert all(all(center[1] in (6, 7) for center in task) for task in train_pool)
    assert all(sum(center[1] == 8 for center in task) == 1 for task in test_pool)


def test_registry_has_locked_disjoint_splits_seeds_and_float_centers(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.md"
    spec.write_bytes(b"locked-stage-c-spec\n")
    registry = build_task_registry(spec)

    assert registry.spec_sha256 == hashlib.sha256(spec.read_bytes()).hexdigest()
    assert len(registry.training) == 16
    assert len(registry.validation) == 4
    assert len(registry.test) == 8

    all_tasks = (*registry.training, *registry.validation, *registry.test)
    assert [task.global_index for task in all_tasks] == list(range(28))
    assert [task.teacher_seed for task in all_tasks] == list(range(41000, 41028))
    assert len({task.task_id for task in all_tasks}) == 28
    assert len({task.center_units for task in all_tasks}) == 28
    for task in all_tasks:
        assert task.centers == tuple(
            (x_unit / 10.0, y_unit / 10.0) for x_unit, y_unit in task.center_units
        )

    assert all(
        all(center[1] != 8 for center in task.center_units)
        for task in (*registry.training, *registry.validation)
    )
    assert all(
        sum(center[1] == 8 for center in task.center_units) == 1
        for task in registry.test
    )


def test_registry_selection_matches_independent_locked_pcg64_shuffle(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text("locked\n", encoding="utf-8")
    registry = build_task_registry(spec)
    train_pool = eligible_layouts(held_out=False)
    test_pool = eligible_layouts(held_out=True)

    train_order = np.random.Generator(np.random.PCG64(202608291)).permutation(
        len(train_pool)
    )
    expected_train = tuple(train_pool[index] for index in train_order[:16])
    remaining = tuple(layout for layout in train_pool if layout not in expected_train)
    validation_order = np.random.Generator(np.random.PCG64(202608292)).permutation(
        len(remaining)
    )
    expected_validation = tuple(remaining[index] for index in validation_order[:4])
    test_order = np.random.Generator(np.random.PCG64(202608293)).permutation(
        len(test_pool)
    )
    expected_test = tuple(test_pool[index] for index in test_order[:8])

    assert tuple(task.center_units for task in registry.training) == expected_train
    assert (
        tuple(task.center_units for task in registry.validation) == expected_validation
    )
    assert tuple(task.center_units for task in registry.test) == expected_test


def test_registry_artifacts_are_atomic_self_consistent_and_non_overwriting(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.md"
    spec.write_text("locked\n", encoding="utf-8")
    output = tmp_path / "artifacts"

    registry = write_task_registry(output, spec)
    registry_path = output / "task_registry.json"
    manifest_path = output / "split_manifest.json"
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert registry_payload["spec_sha256"] == registry.spec_sha256
    assert (
        manifest_payload["task_registry_sha256"]
        == hashlib.sha256(registry_path.read_bytes()).hexdigest()
    )
    assert manifest_payload["splits"]["training"] == [
        task.task_id for task in registry.training
    ]
    assert manifest_payload["splits"]["validation"] == [
        task.task_id for task in registry.validation
    ]
    assert manifest_payload["splits"]["test"] == [
        task.task_id for task in registry.test
    ]
    assert not list(output.glob("*.tmp"))

    try:
        write_task_registry(output, spec)
    except FileExistsError:
        pass
    else:
        raise AssertionError("registry writer must refuse to overwrite artifacts")
