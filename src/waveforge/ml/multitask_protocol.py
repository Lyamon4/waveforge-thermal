"""Prospectively locked schedules and registries for multi-task NCA."""

from __future__ import annotations

from dataclasses import dataclass

DEVELOPMENT_SEED = 2026083101
PRODUCTION_SEEDS = (2026083102, 2026083103, 2026083104)
VALIDATION_INTERVAL = 250
PILOT_UPDATES = 1500
TV_WEIGHT = 1.0e-3
TARGET_MATERIAL_FRACTION = 0.25
PRIMARY_BINARY_CELL_COUNT = 1024


@dataclass(frozen=True)
class MultitaskStage:
    """Objective and optimizer settings at one relative training stage."""

    stage_id: int
    beta: float
    alpha: float
    binary_weight: float
    tv_weight: float
    learning_rate: float


def settings_at(update: int, total_updates: int) -> MultitaskStage:
    """Return the exact 20%/20%/60% continuation settings."""
    if (
        isinstance(total_updates, bool)
        or not isinstance(total_updates, int)
        or total_updates < 1
    ):
        raise ValueError("total update count must be a positive integer")
    if (
        isinstance(update, bool)
        or not isinstance(update, int)
        or not 0 <= update < total_updates
    ):
        raise ValueError("update must lie in [0,total_updates)")

    first_boundary = total_updates // 5
    second_boundary = (2 * total_updates) // 5
    if update < first_boundary:
        return MultitaskStage(1, 2.0, 100.0, 0.0, TV_WEIGHT, 1.0e-3)
    if update < second_boundary:
        return MultitaskStage(2, 4.0, 250.0, 0.01, TV_WEIGHT, 3.0e-4)
    return MultitaskStage(3, 8.0, 500.0, 0.02, TV_WEIGHT, 1.0e-4)
