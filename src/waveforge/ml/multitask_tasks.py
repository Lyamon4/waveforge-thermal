"""Deterministic procedural tasks and immutable split manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D

SOURCE_SIZE = 0.20
PRIMARY_X_RANGE = (0.20, 0.80)
PRIMARY_Y_RANGE = (0.55, 0.82)
OOD_LEFT_X_RANGE = (0.10, 0.18)
OOD_RIGHT_X_RANGE = (0.82, 0.90)
VALIDATION_SEED = 2026083141
TEST_ID_SEED = 2026083142
TEST_OOD_SEED = 2026083143
MAXIMUM_PROPOSALS = 10_000
TRAINING_INDEX_STRIDE = 1_000_000
COLLISION_RETRY_STRIDE = 1_000_000_000_000

Center = tuple[float, float]
Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SourceLayoutTask:
    """One three-scenario thermal design requirement."""

    task_id: str
    centers: tuple[Center, Center, Center]
    bounds: tuple[Bounds, Bounds, Bounds]
    sources: NDArray[np.float64]


@dataclass(frozen=True)
class FrozenTaskSplits:
    """Prospectively frozen validation and untouched test tasks."""

    validation: tuple[SourceLayoutTask, ...]
    test_id: tuple[SourceLayoutTask, ...]
    test_ood: tuple[SourceLayoutTask, ...]

    @property
    def all_tasks(self) -> tuple[SourceLayoutTask, ...]:
        return self.validation + self.test_id + self.test_ood


def rectangles_overlap(first: Bounds, second: Bounds) -> bool:
    """Return whether two half-open rectangles overlap with positive area."""
    return bool(
        max(first[0], second[0]) < min(first[1], second[1])
        and max(first[2], second[2]) < min(first[3], second[3])
    )


def _stream_seed(seed: int, index: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("index must be a non-negative integer")
    digest = hashlib.sha256(f"{seed}:{index}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _bounds(center: Center) -> Bounds:
    half = 0.5 * SOURCE_SIZE
    return (
        center[0] - half,
        center[0] + half,
        center[1] - half,
        center[1] + half,
    )


def _draw_nonoverlapping(
    rng: np.random.Generator,
    x_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[Center, Center, Center]:
    for _ in range(MAXIMUM_PROPOSALS):
        centers = tuple(
            (float(rng.uniform(*x_range)), float(rng.uniform(*PRIMARY_Y_RANGE)))
            for x_range in x_ranges
        )
        ordered = tuple(sorted(centers))
        candidate_bounds = tuple(_bounds(center) for center in ordered)
        if all(
            not rectangles_overlap(candidate_bounds[left], candidate_bounds[right])
            for left in range(3)
            for right in range(left + 1, 3)
        ):
            return ordered  # type: ignore[return-value]
    raise RuntimeError("source-layout rejection sampling exceeded 10000 proposals")


def _task_id(
    centers: tuple[Center, Center, Center],
    sources: NDArray[np.float64],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(centers, separators=(",", ":")).encode("ascii"))
    digest.update(str(sources.dtype).encode("ascii"))
    digest.update(str(sources.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(sources).tobytes())
    return digest.hexdigest()


def _sample_task(seed: int, index: int, *, ood: bool) -> SourceLayoutTask:
    rng = np.random.Generator(np.random.PCG64(_stream_seed(seed, index)))
    if ood:
        edge_range = OOD_LEFT_X_RANGE if rng.integers(0, 2) == 0 else OOD_RIGHT_X_RANGE
        x_ranges = (edge_range, PRIMARY_X_RANGE, PRIMARY_X_RANGE)
    else:
        x_ranges = (PRIMARY_X_RANGE, PRIMARY_X_RANGE, PRIMARY_X_RANGE)
    centers = _draw_nonoverlapping(rng, x_ranges)
    bounds = tuple(_bounds(center) for center in centers)
    grid = Grid2D(nx=64, ny=64)
    sources = np.stack(
        [area_overlap_rectangular_source(grid, rectangle, 1.0) for rectangle in bounds]
    ).astype(np.float64, copy=False)
    return SourceLayoutTask(
        task_id=_task_id(centers, sources),
        centers=centers,
        bounds=bounds,  # type: ignore[arg-type]
        sources=sources,
    )


def sample_primary_task(seed: int, index: int) -> SourceLayoutTask:
    """Sample one reproducible in-distribution training task."""
    return _sample_task(seed, index, ood=False)


def sample_training_task(
    seed: int,
    update: int,
    microbatch_index: int,
    *,
    blocked_task_ids: frozenset[str],
) -> SourceLayoutTask:
    """Derive one task from its tuple and deterministically reject split leakage."""
    if update < 0 or microbatch_index < 0:
        raise ValueError("training task indices must be non-negative")
    base_index = update * TRAINING_INDEX_STRIDE + microbatch_index
    for retry in range(MAXIMUM_PROPOSALS):
        task = sample_primary_task(
            seed,
            base_index + retry * COLLISION_RETRY_STRIDE,
        )
        if task.task_id not in blocked_task_ids:
            return task
    raise RuntimeError("training sampler could not avoid frozen task leakage")


def build_frozen_splits() -> FrozenTaskSplits:
    """Construct the exact preregistered validation and test manifests."""
    return FrozenTaskSplits(
        validation=tuple(sample_primary_task(VALIDATION_SEED, i) for i in range(32)),
        test_id=tuple(sample_primary_task(TEST_ID_SEED, i) for i in range(32)),
        test_ood=tuple(_sample_task(TEST_OOD_SEED, i, ood=True) for i in range(16)),
    )


def _manifest_task(task: SourceLayoutTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "centers": [list(center) for center in task.centers],
        "bounds": [list(bounds) for bounds in task.bounds],
    }


def write_split_manifest(path: Path, splits: FrozenTaskSplits) -> None:
    """Atomically write the compact immutable split manifest."""
    payload = {
        "schema_version": 1,
        "split_seeds": {
            "validation": VALIDATION_SEED,
            "test_id": TEST_ID_SEED,
            "test_ood": TEST_OOD_SEED,
        },
        "validation": [_manifest_task(task) for task in splits.validation],
        "test_id": [_manifest_task(task) for task in splits.test_id],
        "test_ood": [_manifest_task(task) for task in splits.test_ood],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
