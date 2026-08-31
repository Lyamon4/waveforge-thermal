from __future__ import annotations

import numpy as np
import pytest
import torch

from waveforge.ml.mt2b_evaluation import BootstrapResult, MT2BCheckpointSummary
from waveforge.ml.mt2b_final_evaluation import (
    classify_mt2b_result,
    generate_mt2b_designs,
)
from waveforge.ml.mt2b_nca import MT2BNCA
from waveforge.ml.multitask_tasks import VALIDATION_SEED, sample_primary_task
from waveforge.ml.nca_training import model_state_sha256


def _summary(median: float, p90: float) -> MT2BCheckpointSummary:
    return MT2BCheckpointSummary(
        completed_updates=1750,
        split_name="validation",
        task_count=32,
        invalid_count=0,
        median_relative_gap=median,
        p90_relative_gap=p90,
        worst_relative_gap=0.25,
        median_absolute_tmax=0.18,
    )


def _bootstrap(*, lower: float) -> BootstrapResult:
    return BootstrapResult(
        statistic="median",
        resamples=10_000,
        seed=2026092203,
        median_paired_delta=0.04,
        lower_bound=lower,
        upper_bound=0.06,
        conditioning_ci_pass=lower > 0.0,
    )


def test_result_requires_paired_conditioning_evidence_before_quality_tier() -> None:
    raw = np.full(32, 0.12)
    physics = np.full(32, 0.08)

    verdict = classify_mt2b_result(
        physics_summary=_summary(0.08, 0.15),
        raw_gaps=raw,
        physics_gaps=physics,
        bootstrap=_bootstrap(lower=0.02),
    )

    assert verdict.status == "PHYSICS_GO"
    assert verdict.paired_win_count == 32
    assert verdict.median_paired_gap_reduction == pytest.approx(0.04)

    no_ci = classify_mt2b_result(
        physics_summary=_summary(0.08, 0.15),
        raw_gaps=raw,
        physics_gaps=physics,
        bootstrap=_bootstrap(lower=0.0),
    )
    assert no_ci.status == "PHYSICS_NO_GO"


def test_result_uses_locked_quality_thresholds_and_invalid_run() -> None:
    raw = np.full(32, 0.22)
    physics = raw - 0.04
    bootstrap = _bootstrap(lower=0.01)

    conditional = classify_mt2b_result(
        physics_summary=_summary(0.14, 0.29),
        raw_gaps=raw,
        physics_gaps=physics,
        bootstrap=bootstrap,
    )
    assert conditional.status == "PHYSICS_CONDITIONAL_GO"

    invalid_summary = _summary(0.04, 0.08)
    invalid_summary = MT2BCheckpointSummary(
        **{**invalid_summary.__dict__, "invalid_count": 1}
    )
    invalid = classify_mt2b_result(
        physics_summary=invalid_summary,
        raw_gaps=raw,
        physics_gaps=physics,
        bootstrap=bootstrap,
    )
    assert invalid.status == "MT2B_INVALID_RUN"


@pytest.mark.parametrize("variant", ["RAW", "PHYSICS"])
def test_frozen_design_generation_has_exact_budget_and_no_optimizer(
    tmp_path, variant: str
) -> None:
    model = MT2BNCA()
    checkpoint = tmp_path / "checkpoint_000250.pt"
    torch.save(
        {
            "schema_version": 1,
            "completed_updates": 250,
            "model_state_sha256": model_state_sha256(model),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )
    task = sample_primary_task(VALIDATION_SEED, 0)

    result = generate_mt2b_designs(
        checkpoint,
        (task,),
        variant=variant,
        device=torch.device("cpu"),
    )

    assert result.completed_updates == 250
    assert result.variant == variant
    assert len(result.designs) == 1
    assert result.designs[0].task_id == task.task_id
    assert np.count_nonzero(result.designs[0].binary_design) == 1024
    assert result.designs[0].binary_material_fraction == 0.25
