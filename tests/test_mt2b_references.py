from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from waveforge.experiments.run_mt2b_references import (
    MT2BReferenceError,
    build_reference_job,
    load_reference_artifact,
    validate_acceleration_qualification,
)
from waveforge.ml.mt2b_reference_optimization import (
    optimize_reference_scenario_batched,
)
from waveforge.ml.multitask_tasks import VALIDATION_SEED, sample_primary_task
from waveforge.reproducibility import artifact_sha256


def _binary_design() -> np.ndarray:
    design = np.zeros((64, 64), dtype=np.float64)
    design.reshape(-1)[:1024] = 1.0
    return design


def test_reference_job_uses_only_locked_validation_layout_and_seed_formula() -> None:
    job = build_reference_job(7)
    expected = sample_primary_task(VALIDATION_SEED, 7)

    assert job.task_index == 7
    assert job.task_id == expected.task_id
    assert job.optimizer_seed == 2026083207
    assert job.split_name == "validation"


@pytest.mark.parametrize("index", [-1, 32])
def test_reference_job_rejects_out_of_registry_index(index: int) -> None:
    with pytest.raises(ValueError, match=r"\[0,32\)"):
        build_reference_job(index)


def test_reference_artifact_requires_exact_binary_budget_and_identity(
    tmp_path: Path,
) -> None:
    job = build_reference_job(0)
    design_path = tmp_path / "binary_design_64.npy"
    result_path = tmp_path / "reference_result.json"
    np.save(design_path, _binary_design(), allow_pickle=False)
    result_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "PASS",
                "split": "validation",
                "task_index": 0,
                "task_id": job.task_id,
                "optimizer_seed": 2026083200,
                "completed_iterations": 600,
                "binary_cell_count": 1024,
                "binary_material_fraction": 0.25,
                "binary_design_sha256": artifact_sha256(design_path),
            }
        ),
        encoding="utf-8",
    )

    artifact = load_reference_artifact(result_path, design_path, expected=job)

    assert artifact["task_id"] == job.task_id
    assert np.array_equal(artifact["binary_design"], _binary_design())

    broken = _binary_design()
    broken.reshape(-1)[1024] = 1.0
    np.save(design_path, broken, allow_pickle=False)
    with pytest.raises(MT2BReferenceError, match="1024"):
        load_reference_artifact(result_path, design_path, expected=job)


def test_acceleration_qualification_requires_exact_binary_equivalence() -> None:
    sequential = _binary_design()
    scenario_batched = sequential.copy()

    accepted = validate_acceleration_qualification(
        sequential,
        scenario_batched,
        sequential_tmax=0.2,
        scenario_batched_tmax=0.2,
    )

    assert accepted["accepted"] is True
    assert accepted["binary_design_exact_match"] is True

    scenario_batched.reshape(-1)[0] = 0.0
    scenario_batched.reshape(-1)[1024] = 1.0
    rejected = validate_acceleration_qualification(
        sequential,
        scenario_batched,
        sequential_tmax=0.2,
        scenario_batched_tmax=0.2,
    )
    assert rejected["accepted"] is False


def test_scenario_batched_reference_optimizer_runs_fail_closed_unit_step() -> None:
    task = sample_primary_task(VALIDATION_SEED, 0)
    sources = torch.as_tensor(task.sources, dtype=torch.float64)

    result = optimize_reference_scenario_batched(
        sources,
        seed=2026083200,
        iterations=1,
        allow_cpu_unit_test=True,
    )

    assert result.completed_iterations == 1
    assert result.binary_design.shape == (64, 64)
    assert int(torch.count_nonzero(result.binary_design).item()) == 1024
    assert result.binary_material_fraction == 0.25
    assert len(result.records) == 1
    assert result.records[0].maximum_relative_residual <= 1.0e-6


def test_reference_optimizer_production_contract_requires_cuda_and_600_steps() -> None:
    task = sample_primary_task(VALIDATION_SEED, 0)
    sources = torch.as_tensor(task.sources, dtype=torch.float64)

    with pytest.raises(ValueError, match="600"):
        optimize_reference_scenario_batched(
            sources,
            seed=2026083200,
            iterations=1,
        )
