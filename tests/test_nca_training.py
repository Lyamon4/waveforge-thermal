"""Проверки differentiable physics path для pure NCA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from waveforge.design.parameterization import binary_design
from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import PureNCA, project_nca_material
from waveforge.ml.nca_training import (
    NCAForwardResult,
    NCARunStatus,
    evaluate_nca,
    initialize_nca,
    run_nca_training,
)
from waveforge.physics.cg import CGConvergenceError, CGDiagnostics


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


def test_same_model_seed_produces_exact_initial_weights() -> None:
    first = initialize_nca(20260831, torch.device("cpu"))
    second = initialize_nca(20260831, torch.device("cpu"))

    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_first_two_updates_open_expected_gradient_path() -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))

    result = run_nca_training(
        sources,
        seed=20260831,
        learning_rate=1.0e-3,
        iterations=2,
        mode="unit",
        output_dir=None,
        allow_cpu_unit_test=True,
    )

    assert result.status is NCARunStatus.PASS
    assert result.completed_iterations == 2
    assert [record.iteration for record in result.records] == [0, 1]
    assert result.initial_objective == result.records[0].total_objective
    assert result.records[0].conv1x1_weight_gradient_norm > 1.0e-12
    assert result.records[0].conv3x3_weight_gradient_norm == 0.0
    assert result.records[1].conv3x3_weight_gradient_norm > 1.0e-12
    assert len(result.solve_records) == 12
    assert all(record.converged for record in result.solve_records)
    first_record = result.records[0]
    assert first_record.continuous_material_fraction == pytest.approx(0.25, abs=1.0e-6)
    assert first_record.binary_material_fraction == 0.0
    assert first_record.material_std == 0.0
    assert first_record.hidden_state_rms == 0.0
    assert first_record.delta_state_rms == 0.0
    assert first_record.maximum_absolute_delta == 0.0
    assert first_record.maximum_absolute_state == 0.0
    assert first_record.maximum_explicit_relative_residual <= 1.0e-6
    assert first_record.all_cg_converged is True
    assert first_record.all_parameter_gradients_finite is True
    assert first_record.finite is True
    assert first_record.gradient_norm_before_clipping > 0.0
    assert first_record.gradient_norm_after_clipping <= 1.0 + 1.0e-6
    assert math.isfinite(first_record.wall_seconds)
    assert result.final_continuous_design is not None
    assert result.final_binary_design is not None
    assert result.final_continuous_design.shape == (64, 64)
    assert len(result.initial_model_hash) == 64
    assert len(result.final_model_hash) == 64


def test_training_writes_checkpoint_metrics_and_pickle_free_designs(
    tmp_path: Path,
) -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))

    result = run_nca_training(
        sources,
        seed=7,
        learning_rate=1.0e-3,
        iterations=1,
        mode="unit",
        output_dir=tmp_path,
        allow_cpu_unit_test=True,
        checkpoint_interval=1,
    )

    assert result.status is NCARunStatus.PASS
    checkpoint = torch.load(
        tmp_path / "checkpoint_000001.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["completed_updates"] == 1
    assert checkpoint["last_iteration"] == 0
    assert checkpoint["model_state_sha256"] == result.final_model_hash
    metrics = (tmp_path / "optimization_metrics.csv").read_text(encoding="utf-8")
    assert "conv3x3_weight_gradient_norm" in metrics
    assert "maximum_explicit_relative_residual" in metrics
    payload = json.loads((tmp_path / "nca_run_result.json").read_text())
    assert payload["completed_iterations"] == 1
    assert payload["status"] == "PASS"
    continuous = np.load(tmp_path / "design_continuous_64.npy", allow_pickle=False)
    binary = np.load(tmp_path / "design_binary_64.npy", allow_pickle=False)
    np.testing.assert_array_equal(binary, continuous >= 0.5)


@pytest.mark.parametrize(
    ("failure", "reason_code"),
    [
        (FloatingPointError("non-finite objective"), "NONFINITE_TRAINING_STATE"),
        (
            CGConvergenceError(
                CGDiagnostics(2000, 2.0e-6, False, "MAXIMUM_ITERATIONS")
            ),
            "CG_NONCONVERGENCE",
        ),
        (torch.OutOfMemoryError("CUDA out of memory"), "CUDA_OOM"),
    ],
)
def test_numerical_failure_is_fail_closed_without_final_design(
    failure: Exception,
    reason_code: str,
) -> None:
    def failing_evaluator(*args, **kwargs) -> NCAForwardResult:
        raise failure

    sources = gate2_source_batch(device=torch.device("cpu"))

    result = run_nca_training(
        sources,
        seed=7,
        learning_rate=1.0e-3,
        iterations=2,
        mode="unit",
        output_dir=None,
        evaluator=failing_evaluator,
        allow_cpu_unit_test=True,
    )

    assert result.status is NCARunStatus.INVALID_RUN
    assert result.reason_codes == (reason_code,)
    assert result.completed_iterations == 0
    assert result.final_continuous_design is None
    assert result.final_binary_design is None
