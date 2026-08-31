"""Tests for the prospective multi-task source-layout registry."""

import json

import numpy as np

from waveforge.ml.multitask_tasks import (
    OOD_LEFT_X_RANGE,
    OOD_RIGHT_X_RANGE,
    build_frozen_splits,
    rectangles_overlap,
    sample_primary_task,
    write_split_manifest,
)


def test_primary_task_is_deterministic_and_preserves_equal_power() -> None:
    first = sample_primary_task(seed=17, index=9)
    second = sample_primary_task(seed=17, index=9)

    assert first.task_id == second.task_id
    assert first.centers == second.centers
    assert first.bounds == second.bounds
    assert np.array_equal(first.sources, second.sources)
    assert first.sources.dtype == np.float64
    assert first.sources.shape == (3, 64, 64)
    assert np.allclose(first.sources.sum(axis=(1, 2)) / 4096.0, 1.0)


def test_primary_task_has_sorted_nonoverlapping_rectangles_in_locked_range() -> None:
    task = sample_primary_task(seed=29, index=41)

    assert task.centers == tuple(sorted(task.centers))
    assert all(0.20 <= x <= 0.80 and 0.55 <= y <= 0.82 for x, y in task.centers)
    assert not rectangles_overlap(task.bounds[0], task.bounds[1])
    assert not rectangles_overlap(task.bounds[0], task.bounds[2])
    assert not rectangles_overlap(task.bounds[1], task.bounds[2])


def test_frozen_splits_have_exact_sizes_disjoint_hashes_and_locked_ood() -> None:
    splits = build_frozen_splits()

    assert len(splits.validation) == 32
    assert len(splits.test_id) == 32
    assert len(splits.test_ood) == 16
    hashes = [task.task_id for task in splits.all_tasks]
    assert len(hashes) == 80
    assert len(set(hashes)) == 80
    for task in splits.test_ood:
        assert any(
            OOD_LEFT_X_RANGE[0] <= x <= OOD_LEFT_X_RANGE[1]
            or OOD_RIGHT_X_RANGE[0] <= x <= OOD_RIGHT_X_RANGE[1]
            for x, _ in task.centers
        )


def test_split_manifest_is_canonical_and_excludes_source_arrays(tmp_path) -> None:
    path = tmp_path / "split_manifest.json"

    write_split_manifest(path, build_frozen_splits())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["split_seeds"] == {
        "test_id": 2026083142,
        "test_ood": 2026083143,
        "validation": 2026083141,
    }
    assert len(payload["validation"]) == 32
    assert "sources" not in payload["validation"][0]
    assert path.read_bytes().endswith(b"\n")
