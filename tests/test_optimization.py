"""Tests for the deterministic fail-closed Gate 2A optimizer."""

from pathlib import Path

import numpy as np
import torch

from waveforge.design.optimize import (
    OptimizationConfig,
    alpha_for_iteration,
    array_sha256,
    beta_for_iteration,
    binarization_weight_for_iteration,
    initialize_logits,
    optimize_design,
)
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.experiments.run_inverse_design import benchmark_full_optimization_step
from waveforge.physics.cg import CGConfig
from waveforge.physics.grid import Grid2D
from waveforge.verification.compare import Gate2Status


def _source_batch(grid: Grid2D, device: torch.device) -> torch.Tensor:
    source = area_overlap_rectangular_source(
        grid,
        bounds=(0.40, 0.60, 0.62, 0.82),
        power=1.0,
    )
    return torch.as_tensor(source[None], dtype=torch.float64, device=device)


def test_schedules_match_every_locked_stage_boundary() -> None:
    """An off-by-one or post-result schedule change must fail."""
    expected = {
        0: (1.0, 50.0, 0.0),
        199: (1.0, 50.0, 0.0),
        200: (2.0, 200.0, 0.005),
        349: (2.0, 200.0, 0.005),
        350: (4.0, 500.0, 0.01),
        499: (4.0, 500.0, 0.01),
        500: (8.0, 500.0, 0.02),
        599: (8.0, 500.0, 0.02),
    }
    for iteration, values in expected.items():
        assert beta_for_iteration(iteration) == values[0]
        assert alpha_for_iteration(iteration) == values[1]
        assert binarization_weight_for_iteration(iteration) == values[2]


def test_initial_logits_use_isolated_reproducible_seed_and_content_hash() -> None:
    """Using global RNG state or omitting content identity must fail."""
    first = initialize_logits(20260828, device=torch.device("cpu"))
    np.random.seed(1)
    _ = np.random.normal(size=100)
    second = initialize_logits(20260828, device=torch.device("cpu"))
    different = initialize_logits(20260829, device=torch.device("cpu"))

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert first.dtype is torch.float32
    assert first.shape == (16, 16)
    assert array_sha256(first) == array_sha256(second)
    assert array_sha256(first) != array_sha256(different)


def test_real_cg_nonconvergence_invalidates_before_optimizer_step() -> None:
    """Returning an unconverged field or taking an Adam step must fail."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    grid = Grid2D(nx=64, ny=64)
    config = OptimizationConfig(
        iterations=2,
        mode="unit",
        cg_config=CGConfig(maximum_iterations=1),
    )

    result = optimize_design(
        _source_batch(grid, torch.device("cuda")),
        seed=20260828,
        config=config,
        output_dir=None,
    )

    assert result.status is Gate2Status.INVALID_RUN
    assert result.completed_iterations == 0
    assert result.records == ()
    assert "CG_NONCONVERGENCE" in result.reason_codes
    assert result.initial_logits_hash == result.final_logits_hash


def test_valid_run_with_binary_budget_failure_is_no_go_effect(tmp_path: Path) -> None:
    """A numerically valid but non-binary map must not be reported as invalid."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    grid = Grid2D(nx=64, ny=64)
    config = OptimizationConfig(iterations=1, mode="unit")

    result = optimize_design(
        _source_batch(grid, torch.device("cuda")),
        seed=20260828,
        config=config,
        output_dir=tmp_path,
    )

    assert result.status is Gate2Status.NO_GO_EFFECT
    assert result.completed_iterations == 1
    assert result.binary_material_fraction < 0.24
    assert result.reason_codes == ("BINARY_BUDGET_FAILURE",)
    assert len(result.records) == 1
    assert np.isfinite(result.records[0].total_objective)
    assert (tmp_path / "initial_logits.pt").is_file()
    assert (tmp_path / "checkpoint_final.pt").is_file()
    assert (tmp_path / "optimization_metrics.csv").is_file()
    assert (tmp_path / "optimization_result.json").is_file()


def test_full_step_benchmark_includes_three_forward_and_adjoint_solves(
    tmp_path: Path,
) -> None:
    """Timing only a forward pass or omitting dtype evidence must fail."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    output_path = tmp_path / "full_iteration_benchmark.json"

    payload = benchmark_full_optimization_step(output_path)

    assert payload["status"] == "PASS"
    assert payload["schema_version"] == 2
    assert payload["forward_solve_count"] == 3
    assert payload["adjoint_solve_count"] == 3
    assert payload["design_dtype"] == "float32"
    assert payload["physics_solve_dtype"] == "float64"
    assert payload["gradient_dtype"] == "float32"
    assert payload["maximum_explicit_relative_residual"] <= 1e-6
    assert payload["step_wall_seconds"] > 0.0
    assert payload["peak_cuda_memory_bytes"] > 0
    assert output_path.is_file() and output_path.stat().st_size > 0
