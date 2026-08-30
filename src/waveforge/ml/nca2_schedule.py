"""Pure half-open objective and learning-rate schedules for NCA-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NCA2ProtocolId = Literal["A", "B"]


@dataclass(frozen=True)
class ObjectiveSettings:
    projection_beta: float
    smooth_max_alpha: float
    tv_weight: float
    binarization_weight: float


def _validate_iteration(iteration: int) -> None:
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise TypeError("iteration must be an integer")
    if not 0 <= iteration < 1500:
        raise ValueError("iteration must lie in [0,1500)")


def objective_settings_at(iteration: int) -> ObjectiveSettings:
    """Return the preregistered objective settings for one zero-based update."""
    _validate_iteration(iteration)
    if iteration < 250:
        return ObjectiveSettings(2.0, 100.0, 0.001, 0.0)
    if iteration < 500:
        return ObjectiveSettings(4.0, 250.0, 0.001, 0.01)
    return ObjectiveSettings(8.0, 500.0, 0.001, 0.02)


def learning_rate_at(protocol_id: str, iteration: int) -> float:
    """Return the preregistered Protocol A/B rate for one update."""
    _validate_iteration(iteration)
    if protocol_id == "A":
        return 1.0e-3
    if protocol_id != "B":
        raise ValueError(f"unregistered NCA-2 protocol: {protocol_id}")
    if iteration < 250:
        return 1.0e-3
    if iteration < 500:
        return 3.0e-4
    return 1.0e-4
