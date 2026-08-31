from __future__ import annotations

import numpy as np
import pytest
import torch

from waveforge.ml.mt2b_conditioning import (
    FIXED_TEMPERATURE_SCALE,
    build_mt2b_conditioning,
    canonical_temperature_scale,
)


def _sources(batch: int = 1) -> torch.Tensor:
    sources = torch.zeros((batch, 3, 64, 64), dtype=torch.float64)
    sources[:, 0, 40:44, 10:14] = 25.0
    sources[:, 1, 44:48, 28:32] = 50.0
    sources[:, 2, 48:52, 48:52] = 75.0
    return sources


def _linear_solver(sources: np.ndarray) -> np.ndarray:
    return 2.0 * sources


def test_canonical_temperature_scale_is_fixed_not_sample_derived() -> None:
    assert FIXED_TEMPERATURE_SCALE == 0.900613256638055
    assert canonical_temperature_scale() == FIXED_TEMPERATURE_SCALE


def test_raw_condition_has_matched_four_channel_shape_and_zero_fields() -> None:
    condition = build_mt2b_conditioning(_sources(), variant="RAW")

    assert condition.shape == (1, 4, 64, 64)
    assert condition.dtype is torch.float32
    torch.testing.assert_close(condition[:, 0], _sources().sum(dim=1).float() / 25.0)
    assert torch.count_nonzero(condition[:, 1:3]) == 0
    assert torch.all(condition[:, 3, 0, :] == 1.0)
    assert torch.count_nonzero(condition[:, 3, 1:, :]) == 0


def test_physics_condition_uses_fixed_mean_and_max_without_clamping() -> None:
    sources = _sources()
    condition = build_mt2b_conditioning(
        sources,
        variant="PHYSICS",
        temperature_solver=_linear_solver,
    )
    fields = 2.0 * sources.numpy()
    expected_mean = torch.from_numpy(fields.mean(axis=1) / FIXED_TEMPERATURE_SCALE)
    expected_max = torch.from_numpy(fields.max(axis=1) / FIXED_TEMPERATURE_SCALE)

    torch.testing.assert_close(condition[:, 1].double(), expected_mean)
    torch.testing.assert_close(condition[:, 2].double(), expected_max)
    assert float(condition[:, 2].max()) > 1.0


def test_physics_condition_is_scenario_permutation_invariant() -> None:
    sources = _sources()
    first = build_mt2b_conditioning(
        sources,
        variant="PHYSICS",
        temperature_solver=_linear_solver,
    )
    permuted = build_mt2b_conditioning(
        sources[:, [2, 0, 1]],
        variant="PHYSICS",
        temperature_solver=_linear_solver,
    )

    torch.testing.assert_close(first, permuted)


def test_physics_condition_is_detached_from_source_autograd() -> None:
    sources = _sources().requires_grad_(True)
    condition = build_mt2b_conditioning(
        sources,
        variant="PHYSICS",
        temperature_solver=_linear_solver,
    )

    assert condition.requires_grad is False


@pytest.mark.parametrize(
    "sources",
    [
        torch.zeros((3, 64, 64), dtype=torch.float32),
        torch.zeros((1, 2, 64, 64), dtype=torch.float64),
        torch.full((1, 3, 64, 64), float("nan"), dtype=torch.float64),
    ],
)
def test_conditioning_rejects_invalid_physical_sources(sources: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        build_mt2b_conditioning(sources, variant="RAW")


def test_physics_variant_requires_temperature_solver() -> None:
    with pytest.raises(ValueError, match="temperature_solver"):
        build_mt2b_conditioning(_sources(), variant="PHYSICS")
