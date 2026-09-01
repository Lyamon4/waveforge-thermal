from __future__ import annotations

import pytest
import torch

from waveforge.ml.mt3_unet import (
    MT3UNet,
    count_mt3_parameters,
    project_mt3_candidates,
)


def test_mt3_unet_emits_four_deterministic_logits() -> None:
    torch.manual_seed(2026092311)
    model = MT3UNet().eval()
    condition = torch.randn(2, 5, 64, 64, dtype=torch.float32)

    first = model(condition)
    second = model(condition)

    assert first.shape == (2, 4, 64, 64)
    assert first.dtype is torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_mt3_unet_has_compact_registered_parameter_count() -> None:
    model = MT3UNet()
    count = count_mt3_parameters(model)

    assert 1_000_000 < count < 5_000_000
    assert count == sum(parameter.numel() for parameter in model.parameters())


def test_every_candidate_has_exact_continuous_and_binary_budget() -> None:
    torch.manual_seed(7)
    logits = torch.randn(2, 4, 64, 64, dtype=torch.float32, requires_grad=True)
    projected = project_mt3_candidates(logits, beta=8.0)

    assert projected.designs.shape == projected.binary.shape == (2, 4, 64, 64)
    torch.testing.assert_close(
        projected.designs.mean(dim=(-2, -1)),
        torch.full((2, 4), 0.25),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        projected.binary.sum(dim=(-2, -1)),
        torch.full((2, 4), 1024.0),
        atol=0.0,
        rtol=0.0,
    )
    assert len(projected.projection_errors) == 8
    assert max(projected.projection_errors) <= 1.0e-6

    projected.designs.square().mean().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


@pytest.mark.parametrize(
    "condition",
    [
        torch.zeros((1, 4, 64, 64), dtype=torch.float32),
        torch.zeros((1, 5, 32, 32), dtype=torch.float32),
        torch.zeros((1, 5, 64, 64), dtype=torch.float64),
    ],
)
def test_mt3_unet_rejects_invalid_condition(condition: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="condition"):
        MT3UNet()(condition)


def test_candidate_projection_rejects_wrong_contract() -> None:
    with pytest.raises(ValueError, match="logits"):
        project_mt3_candidates(
            torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            beta=8.0,
        )
