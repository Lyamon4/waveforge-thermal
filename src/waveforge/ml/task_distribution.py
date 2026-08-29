"""Deterministic prospective source-layout registry for the ML spike."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

CENTER_X_UNITS = (2, 3, 4, 5, 6, 7, 8)
CENTER_Y_UNITS = (6, 7, 8)
TRAIN_SELECTION_SEED = 202608291
VALIDATION_SELECTION_SEED = 202608292
TEST_SELECTION_SEED = 202608293
CenterUnits = tuple[int, int]
LayoutUnits = tuple[CenterUnits, CenterUnits, CenterUnits]
Split = Literal["training", "validation", "test"]


@dataclass(frozen=True)
class SourceLayoutTask:
    """One immutable source-layout task selected before teacher generation."""

    task_id: str
    split: Split
    split_index: int
    global_index: int
    teacher_seed: int
    center_units: LayoutUnits

    @property
    def centers(self) -> tuple[tuple[float, float], ...]:
        """Return physical centers converted from exact tenths units."""
        return tuple((x / 10.0, y / 10.0) for x, y in self.center_units)

    def to_payload(self) -> dict[str, object]:
        """Serialize without losing the exact integer-grid representation."""
        return {
            "task_id": self.task_id,
            "split": self.split,
            "split_index": self.split_index,
            "global_index": self.global_index,
            "teacher_seed": self.teacher_seed,
            "center_units": [list(center) for center in self.center_units],
            "centers": [list(center) for center in self.centers],
        }


@dataclass(frozen=True)
class TaskRegistry:
    """Complete immutable `16/4/8` split registry."""

    spec_sha256: str
    training: tuple[SourceLayoutTask, ...]
    validation: tuple[SourceLayoutTask, ...]
    test: tuple[SourceLayoutTask, ...]
    training_pool_size: int
    test_pool_size: int

    @property
    def tasks(self) -> tuple[SourceLayoutTask, ...]:
        """Return tasks in the locked global manifest order."""
        return (*self.training, *self.validation, *self.test)

    def to_payload(self) -> dict[str, object]:
        """Return the machine-readable normative registry payload."""
        return {
            "schema_version": 1,
            "status": "LOCKED_BEFORE_TEACHER_GENERATION",
            "spec_sha256": self.spec_sha256,
            "source_size": [0.2, 0.2],
            "integrated_power_per_source": 1.0,
            "minimum_center_separation": 0.2,
            "center_grid_units_tenths": {
                "x": list(CENTER_X_UNITS),
                "y": list(CENTER_Y_UNITS),
            },
            "selection_seeds": {
                "training": TRAIN_SELECTION_SEED,
                "validation": VALIDATION_SELECTION_SEED,
                "test": TEST_SELECTION_SEED,
            },
            "pool_sizes": {
                "train_validation": self.training_pool_size,
                "held_out_test": self.test_pool_size,
            },
            "tasks": [task.to_payload() for task in self.tasks],
        }


def _separated(layout: tuple[CenterUnits, ...]) -> bool:
    return all(
        (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 >= 4
        for left, right in itertools.combinations(layout, 2)
    )


def eligible_layouts(*, held_out: bool) -> tuple[LayoutUnits, ...]:
    """Enumerate one locked pool using exact integer-grid predicates."""
    allowed_y = CENTER_Y_UNITS if held_out else CENTER_Y_UNITS[:2]
    centers = tuple(itertools.product(CENTER_X_UNITS, allowed_y))
    layouts: list[LayoutUnits] = []
    for combination in itertools.combinations(centers, 3):
        layout = tuple(sorted(combination))
        if not _separated(layout):
            continue
        if held_out and sum(center[1] == 8 for center in layout) != 1:
            continue
        layouts.append(layout)  # type: ignore[arg-type]
    return tuple(layouts)


def _shuffle_indices(length: int, seed: int) -> np.ndarray:
    return np.random.Generator(np.random.PCG64(seed)).permutation(length)


def _task(
    split: Split,
    split_index: int,
    global_index: int,
    layout: LayoutUnits,
) -> SourceLayoutTask:
    return SourceLayoutTask(
        task_id=f"ml_{split}_{split_index:03d}",
        split=split,
        split_index=split_index,
        global_index=global_index,
        teacher_seed=41000 + global_index,
        center_units=layout,
    )


def build_task_registry(spec_path: Path) -> TaskRegistry:
    """Select the complete registry by the locked PCG64 procedure."""
    if not spec_path.is_file():
        raise FileNotFoundError(f"locked ML specification is missing: {spec_path}")
    training_pool = eligible_layouts(held_out=False)
    test_pool = eligible_layouts(held_out=True)

    training_order = _shuffle_indices(len(training_pool), TRAIN_SELECTION_SEED)
    selected_training = tuple(training_pool[index] for index in training_order[:16])
    selected_training_set = set(selected_training)
    validation_pool = tuple(
        layout for layout in training_pool if layout not in selected_training_set
    )
    validation_order = _shuffle_indices(len(validation_pool), VALIDATION_SELECTION_SEED)
    selected_validation = tuple(
        validation_pool[index] for index in validation_order[:4]
    )
    test_order = _shuffle_indices(len(test_pool), TEST_SELECTION_SEED)
    selected_test = tuple(test_pool[index] for index in test_order[:8])

    training = tuple(
        _task("training", index, index, layout)
        for index, layout in enumerate(selected_training)
    )
    validation = tuple(
        _task("validation", index, 16 + index, layout)
        for index, layout in enumerate(selected_validation)
    )
    test = tuple(
        _task("test", index, 20 + index, layout)
        for index, layout in enumerate(selected_test)
    )
    return TaskRegistry(
        spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        training=training,
        validation=validation,
        test=test,
        training_pool_size=len(training_pool),
        test_pool_size=len(test_pool),
    )


def _encoded(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def write_task_registry(output_dir: Path, spec_path: Path) -> TaskRegistry:
    """Atomically write the immutable registry and split manifest once."""
    registry_path = output_dir / "task_registry.json"
    manifest_path = output_dir / "split_manifest.json"
    if registry_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing ML task registry")
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = build_task_registry(spec_path)
    registry_bytes = _encoded(registry.to_payload())
    manifest_payload = {
        "schema_version": 1,
        "spec_sha256": registry.spec_sha256,
        "task_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "splits": {
            "training": [task.task_id for task in registry.training],
            "validation": [task.task_id for task in registry.validation],
            "test": [task.task_id for task in registry.test],
        },
        "held_out_rule": "exactly_one_center_with_y_units_8",
        "training_access_forbidden": [task.task_id for task in registry.test],
    }
    registry_temporary = registry_path.with_suffix(".json.tmp")
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    registry_temporary.write_bytes(registry_bytes)
    manifest_temporary.write_bytes(_encoded(manifest_payload))
    registry_temporary.replace(registry_path)
    manifest_temporary.replace(manifest_path)
    return registry
