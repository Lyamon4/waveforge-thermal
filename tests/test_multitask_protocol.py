"""Tests for the locked relative multi-task training protocol."""

import pytest

from waveforge.ml.multitask_protocol import (
    DEVELOPMENT_SEED,
    PRODUCTION_SEEDS,
    settings_at,
)


@pytest.mark.parametrize(
    ("update", "beta", "alpha", "binary_weight", "learning_rate"),
    [
        (0, 2.0, 100.0, 0.0, 1.0e-3),
        (1999, 2.0, 100.0, 0.0, 1.0e-3),
        (2000, 4.0, 250.0, 0.01, 3.0e-4),
        (3999, 4.0, 250.0, 0.01, 3.0e-4),
        (4000, 8.0, 500.0, 0.02, 1.0e-4),
        (9999, 8.0, 500.0, 0.02, 1.0e-4),
    ],
)
def test_relative_schedule_for_10000_updates(
    update: int,
    beta: float,
    alpha: float,
    binary_weight: float,
    learning_rate: float,
) -> None:
    stage = settings_at(update, 10_000)

    assert stage.beta == beta
    assert stage.alpha == alpha
    assert stage.binary_weight == binary_weight
    assert stage.learning_rate == learning_rate
    assert stage.tv_weight == 0.001


def test_pilot_schedule_uses_300_and_600_update_boundaries() -> None:
    assert settings_at(299, 1500).stage_id == 1
    assert settings_at(300, 1500).stage_id == 2
    assert settings_at(599, 1500).stage_id == 2
    assert settings_at(600, 1500).stage_id == 3


def test_protocol_locks_new_model_seeds() -> None:
    assert DEVELOPMENT_SEED == 2026083101
    assert PRODUCTION_SEEDS == (2026083102, 2026083103, 2026083104)


@pytest.mark.parametrize(
    ("update", "total_updates"),
    [(-1, 100), (100, 100), (0, 0)],
)
def test_schedule_rejects_out_of_range_updates(update: int, total_updates: int) -> None:
    with pytest.raises(ValueError, match="update"):
        settings_at(update, total_updates)
