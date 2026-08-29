"""Проверки differentiable physics path для pure NCA."""

from __future__ import annotations

import pytest
import torch

from waveforge.design.parameterization import binary_design
from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import PureNCA, project_nca_material
from waveforge.ml.nca_training import evaluate_nca


def test_zero_material_logit_projects_to_exact_quarter_volume() -> None:
    logits = torch.zeros((1, 1, 64, 64), dtype=torch.float32, requires_grad=True)

    result = project_nca_material(logits)

    assert result.design.dtype is torch.float32
    assert result.design.mean().item() == pytest.approx(0.25, abs=1.0e-6)
    assert torch.count_nonzero(binary_design(result.design)) == 0
    assert result.projection.converged is True
    assert result.projection.iterations == 80


def test_only_material_channel_enters_readout() -> None:
    state = torch.zeros((1, 16, 64, 64), dtype=torch.float32)
    state[:, 1:] = torch.randn_like(state[:, 1:])

    first = project_nca_material(state[:, 0:1]).design
    state[:, 1:] *= 100.0
    second = project_nca_material(state[:, 0:1]).design

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_projection_gradient_is_finite_nonzero_and_volume_tangent() -> None:
    coordinates = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32)
    logits = (coordinates[:, None] + coordinates[None, :])[None, None].clone()
    logits.requires_grad_(True)
    weights = torch.linspace(0.2, 1.0, 4096, dtype=torch.float32).reshape(64, 64)

    design = project_nca_material(logits).design
    torch.sum(design * weights).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    gradient_norm = torch.linalg.vector_norm(logits.grad).item()
    assert gradient_norm > 1.0e-8
    assert abs(logits.grad.sum().item()) <= 1.0e-5 * gradient_norm


def test_training_forward_does_not_threshold_design(monkeypatch) -> None:
    def forbidden_threshold(*args, **kwargs):
        raise AssertionError("strict threshold entered differentiable training path")

    monkeypatch.setattr("waveforge.ml.nca_training.binary_design", forbidden_threshold)
    model = PureNCA()
    sources = gate2_source_batch(device=torch.device("cpu"))

    result = evaluate_nca(model, sources, allow_cpu_unit_test=True)

    assert result.binary_design is None
    assert result.continuous_design.requires_grad is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_nca_forward_uses_float32_neural_and_float64_physics() -> None:
    model = PureNCA().cuda()
    sources = gate2_source_batch(device=torch.device("cuda"))

    result = evaluate_nca(model, sources)

    assert result.rollout.final_state.dtype is torch.float32
    assert result.continuous_design.dtype is torch.float32
    assert result.temperatures.dtype is torch.float64
    assert result.temperatures.device.type == "cuda"
    assert result.objective.total.dtype is torch.float64
    expected = (
        result.objective.thermal_smooth
        + 0.001 * result.objective.total_variation.double()
        + 0.02 * result.objective.binarization_penalty.double()
    )
    torch.testing.assert_close(result.objective.total, expected)
    assert result.projection.absolute_error <= 1.0e-6
    assert len(result.solve_trace.records) == 3
    assert all(record.role == "forward" for record in result.solve_trace.records)
    assert all(record.converged for record in result.solve_trace.records)


def test_evaluation_rejects_cpu_without_explicit_unit_permission() -> None:
    model = PureNCA()
    sources = gate2_source_batch(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="CUDA"):
        evaluate_nca(model, sources)


def test_material_readout_rejects_wrong_shape_or_dtype() -> None:
    with pytest.raises(ValueError, match="shape"):
        project_nca_material(torch.zeros((1, 64, 64), dtype=torch.float32))
    with pytest.raises(ValueError, match="float32"):
        project_nca_material(torch.zeros((1, 1, 64, 64), dtype=torch.float64))
