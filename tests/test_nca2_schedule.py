from __future__ import annotations

import pytest

from waveforge.ml.nca2_schedule import (
    ObjectiveSettings,
    learning_rate_at,
    objective_settings_at,
)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, ObjectiveSettings(2.0, 100.0, 0.001, 0.0)),
        (249, ObjectiveSettings(2.0, 100.0, 0.001, 0.0)),
        (250, ObjectiveSettings(4.0, 250.0, 0.001, 0.01)),
        (499, ObjectiveSettings(4.0, 250.0, 0.001, 0.01)),
        (500, ObjectiveSettings(8.0, 500.0, 0.001, 0.02)),
        (1499, ObjectiveSettings(8.0, 500.0, 0.001, 0.02)),
    ],
)
def test_objective_schedule_has_exact_half_open_boundaries(
    iteration: int, expected: ObjectiveSettings
) -> None:
    assert objective_settings_at(iteration) == expected


@pytest.mark.parametrize(
    ("protocol_id", "iteration", "expected"),
    [
        ("A", 0, 1.0e-3),
        ("A", 699, 1.0e-3),
        ("A", 1499, 1.0e-3),
        ("B", 0, 1.0e-3),
        ("B", 249, 1.0e-3),
        ("B", 250, 3.0e-4),
        ("B", 499, 3.0e-4),
        ("B", 500, 1.0e-4),
        ("B", 1499, 1.0e-4),
    ],
)
def test_learning_rate_schedule_has_exact_half_open_boundaries(
    protocol_id: str, iteration: int, expected: float
) -> None:
    assert learning_rate_at(protocol_id, iteration) == expected


@pytest.mark.parametrize("iteration", [-1, 1500])
def test_schedules_reject_out_of_range_iterations(iteration: int) -> None:
    with pytest.raises(ValueError, match=r"\[0,1500\)"):
        objective_settings_at(iteration)
    with pytest.raises(ValueError, match=r"\[0,1500\)"):
        learning_rate_at("A", iteration)


def test_learning_rate_schedule_rejects_unregistered_protocol() -> None:
    with pytest.raises(ValueError, match="protocol"):
        learning_rate_at("C", 0)
