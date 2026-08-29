from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from waveforge.ml.teacher import (
    TeacherConfig,
    TeacherStatus,
    optimize_teacher,
    teacher_schedule,
    teacher_source_batch,
    verify_teacher_at_64,
)

PILOT_CENTERS = ((0.30, 0.70), (0.50, 0.70), (0.70, 0.70))


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [
        (0, (1.0, 50.0, 0.0)),
        (66, (1.0, 50.0, 0.0)),
        (67, (2.0, 200.0, 0.005)),
        (116, (2.0, 200.0, 0.005)),
        (117, (4.0, 500.0, 0.01)),
        (166, (4.0, 500.0, 0.01)),
        (167, (8.0, 500.0, 0.02)),
        (199, (8.0, 500.0, 0.02)),
    ],
)
def test_reduced_teacher_schedule_is_locked(
    iteration: int,
    expected: tuple[float, float, float],
) -> None:
    assert teacher_schedule(iteration, resolution=32) == expected


def test_teacher_config_rejects_unregistered_production_protocols() -> None:
    TeacherConfig(resolution=32, iterations=200)
    TeacherConfig(resolution=64, iterations=600)
    with pytest.raises(ValueError, match="locked"):
        TeacherConfig(resolution=32, iterations=199)
    with pytest.raises(ValueError, match="locked"):
        TeacherConfig(resolution=64, iterations=200)


@pytest.mark.parametrize("resolution", [32, 64])
def test_teacher_sources_have_exact_shape_power_and_bounds(resolution: int) -> None:
    sources = teacher_source_batch(
        PILOT_CENTERS,
        resolution=resolution,
        device=torch.device("cpu"),
    )

    assert sources.shape == (3, resolution, resolution)
    assert sources.dtype is torch.float64
    assert torch.isfinite(sources).all()
    cell_area = 1.0 / resolution**2
    integrated = sources.sum(dim=(-2, -1)) * cell_area
    torch.testing.assert_close(
        integrated,
        torch.ones(3, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_teacher_sources_reject_noncanonical_or_overlapping_layouts() -> None:
    with pytest.raises(ValueError, match="canonical"):
        teacher_source_batch(
            tuple(reversed(PILOT_CENTERS)),
            resolution=32,
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="separation"):
        teacher_source_batch(
            ((0.30, 0.70), (0.40, 0.70), (0.70, 0.70)),
            resolution=32,
            device=torch.device("cpu"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_one_step_teacher_uses_locked_mixed_precision_and_finite_physics() -> None:
    result = optimize_teacher(
        PILOT_CENTERS,
        seed=31001,
        config=TeacherConfig(
            resolution=32,
            iterations=1,
            mode="unit",
            enforce_final_binary_budget=False,
        ),
        output_dir=None,
    )

    assert result.status is TeacherStatus.PASS
    assert result.completed_iterations == 1
    assert result.initial_logits.dtype is torch.float32
    assert result.final_logits.dtype is torch.float32
    assert result.continuous_design is not None
    assert result.continuous_design.dtype is torch.float32
    assert result.continuous_design.shape == (32, 32)
    assert result.continuous_material_fraction == pytest.approx(0.25, abs=1.0e-6)
    assert result.records[0].maximum_residual <= 1.0e-6
    assert math.isfinite(result.records[0].total_objective)


def test_independent_64_verifier_uses_exact_transfer_and_strict_budget() -> None:
    design = np.zeros((32, 32), dtype=np.float64)
    design[:, 12:20] = 1.0
    verification = verify_teacher_at_64(
        PILOT_CENTERS,
        design,
        source_resolution=32,
    )

    assert verification.transferred_design.shape == (64, 64)
    np.testing.assert_array_equal(
        verification.transferred_design,
        np.repeat(np.repeat(design, 2, axis=0), 2, axis=1),
    )
    assert verification.material_fraction == 0.25
    assert len(verification.scenario_peaks) == 3
    assert verification.worst_peak == max(verification.scenario_peaks)
    assert verification.maximum_residual <= 1.0e-10


def test_teacher_artifacts_record_arrays_hashes_and_no_pickles(tmp_path: Path) -> None:
    result = optimize_teacher(
        PILOT_CENTERS,
        seed=7,
        config=TeacherConfig(
            resolution=32,
            iterations=1,
            mode="unit",
            enforce_final_binary_budget=False,
            device="cpu",
        ),
        output_dir=tmp_path,
    )

    assert result.status is TeacherStatus.PASS
    assert (tmp_path / "teacher_result.json").is_file()
    assert (tmp_path / "optimization_metrics.csv").is_file()
    continuous = np.load(tmp_path / "design_continuous_32.npy", allow_pickle=False)
    binary = np.load(tmp_path / "design_binary_32.npy", allow_pickle=False)
    assert continuous.shape == (32, 32)
    assert binary.shape == (32, 32)
    assert not list(tmp_path.rglob("*.pt"))
