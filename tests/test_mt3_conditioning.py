from __future__ import annotations

import numpy as np
import pytest
import torch

from waveforge.ml.mt3_conditioning import (
    FIXED_TEMPERATURE_SCALE,
    build_mt3_conditioning,
    compute_initial_probe,
    evaluate_probe_objective,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def _sources(batch: int = 1) -> torch.Tensor:
    tasks = [sample_primary_task(2026092312, index) for index in range(batch)]
    return torch.from_numpy(np.stack([task.sources for task in tasks])).to(
        torch.float64
    )


def test_initial_probe_is_feasible_and_conditioning_is_matched() -> None:
    sources = _sources()
    probe = compute_initial_probe(sources, allow_cpu_unit_test=True)

    assert probe.design.shape == (1, 64, 64)
    torch.testing.assert_close(
        probe.design.mean(dim=(-2, -1)),
        torch.tensor([0.25]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert probe.temperatures.shape == (1, 3, 64, 64)
    assert probe.benefit_raw.shape == (1, 64, 64)
    assert torch.isfinite(probe.benefit_raw).all()
    assert torch.count_nonzero(probe.benefit_raw) > 0
    assert float(probe.benefit_normalized.abs().max()) <= 8.0

    field = build_mt3_conditioning(probe, sources, variant="FIELD_UNET")
    sens = build_mt3_conditioning(probe, sources, variant="SENS_UNET")
    assert field.shape == sens.shape == (1, 5, 64, 64)
    assert field.dtype is sens.dtype is torch.float32
    assert torch.count_nonzero(field[:, 3]) == 0
    torch.testing.assert_close(sens[:, 3], probe.benefit_normalized.float())
    torch.testing.assert_close(
        field[:, (0, 1, 2, 4)],
        sens[:, (0, 1, 2, 4)],
    )
    torch.testing.assert_close(
        sens[:, 1].double(),
        probe.temperature_mean / FIXED_TEMPERATURE_SCALE,
    )
    torch.testing.assert_close(
        sens[:, 2].double(),
        probe.temperature_max / FIXED_TEMPERATURE_SCALE,
    )


def test_probe_is_source_permutation_invariant() -> None:
    sources = _sources()
    original = compute_initial_probe(sources, allow_cpu_unit_test=True)
    permuted = compute_initial_probe(
        sources[:, [2, 0, 1]],
        allow_cpu_unit_test=True,
    )

    torch.testing.assert_close(original.temperature_mean, permuted.temperature_mean)
    torch.testing.assert_close(original.temperature_max, permuted.temperature_max)
    torch.testing.assert_close(
        original.benefit_normalized,
        permuted.benefit_normalized,
        atol=1.0e-10,
        rtol=1.0e-8,
    )


@pytest.mark.parametrize("pixel", [(48, 16), (49, 30), (41, 50)])
def test_probe_sensitivity_matches_central_difference(
    pixel: tuple[int, int],
) -> None:
    sources = _sources()
    probe = compute_initial_probe(sources, allow_cpu_unit_test=True)
    epsilon = 1.0e-2
    plus_logits = torch.zeros((1, 64, 64), dtype=torch.float32)
    minus_logits = torch.zeros((1, 64, 64), dtype=torch.float32)
    plus_logits[(0, *pixel)] = epsilon
    minus_logits[(0, *pixel)] = -epsilon

    plus = evaluate_probe_objective(
        sources,
        plus_logits,
        allow_cpu_unit_test=True,
    )
    minus = evaluate_probe_objective(
        sources,
        minus_logits,
        allow_cpu_unit_test=True,
    )
    finite_difference = float(((plus - minus) / (2.0 * epsilon)).item())
    analytic = float((-probe.benefit_raw[(0, *pixel)]).item())

    assert analytic == pytest.approx(finite_difference, rel=2.0e-2, abs=2.0e-5)


def test_mt3_conditioning_is_detached() -> None:
    sources = _sources().requires_grad_(True)
    probe = compute_initial_probe(sources, allow_cpu_unit_test=True)
    condition = build_mt3_conditioning(probe, sources, variant="SENS_UNET")

    assert condition.requires_grad is False
    assert probe.benefit_normalized.requires_grad is False


@pytest.mark.parametrize(
    "sources",
    [
        torch.zeros((3, 64, 64), dtype=torch.float64),
        torch.zeros((1, 2, 64, 64), dtype=torch.float64),
        torch.zeros((1, 3, 64, 64), dtype=torch.float32),
    ],
)
def test_initial_probe_rejects_invalid_source_contract(sources: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="sources"):
        compute_initial_probe(sources, allow_cpu_unit_test=True)
