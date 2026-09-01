from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from waveforge.ml.mt3_protocol import training_settings_at
from waveforge.ml.mt3_training import (
    MT3BatchForward,
    MT3QualificationRun,
    MT3RunConfig,
    MT3RunStatus,
    evaluate_mt3_batch,
    initialize_mt3_model,
    mt3_model_state_sha256,
    run_mt3_training,
    select_mt3_learning_rate,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def test_field_and_sens_initial_models_have_identical_bytes() -> None:
    field = initialize_mt3_model(2026092311, torch.device("cpu"))
    sens = initialize_mt3_model(2026092311, torch.device("cpu"))

    assert mt3_model_state_sha256(field) == mt3_model_state_sha256(sens)
    assert sum(parameter.numel() for parameter in field.parameters()) == 2_918_724


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        (0, (2.0, 100.0, 0.0, 1.0)),
        (799, (2.0, 100.0, 0.0, 1.0)),
        (800, (4.0, 250.0, 0.01, 0.3)),
        (1599, (4.0, 250.0, 0.01, 0.3)),
        (1600, (8.0, 500.0, 0.02, 0.1)),
        (3999, (8.0, 500.0, 0.02, 0.1)),
    ],
)
def test_locked_mt3_stage_boundaries(
    update: int,
    expected: tuple[float, float, float, float],
) -> None:
    stage = training_settings_at(update)
    assert (
        stage.projection_beta,
        stage.smooth_max_alpha,
        stage.binarization_weight,
        stage.learning_rate_multiplier,
    ) == expected


def test_mt3_batch_forward_reaches_every_unet_parameter_on_cpu() -> None:
    model = initialize_mt3_model(2026092311, torch.device("cpu"))
    task = sample_primary_task(2026092312, 0)
    sources = torch.from_numpy(task.sources).unsqueeze(0)

    forward = evaluate_mt3_batch(
        model,
        sources,
        training_settings_at(0),
        variant="SENS_UNET",
        allow_cpu_unit_test=True,
    )
    forward.loss.backward()

    assert forward.candidate_thermal_smooth.shape == (1, 4)
    assert forward.candidate_exact_tmax.shape == (1, 4)
    assert forward.maximum_projection_absolute_error <= 1.0e-6
    assert [record.role for record in forward.candidate_trace.records] == [
        *(["forward"] * 12),
        *(["adjoint"] * 12),
    ]
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.bias = nn.Parameter(torch.tensor(0.10))


def _tiny_factory(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return _TinyModel().to(device)


def _tiny_tasks(seed: int, update: int) -> tuple[object, ...]:
    del seed, update
    return (sample_primary_task(17, 0),)


def _tiny_evaluator(
    model: nn.Module,
    sources: torch.Tensor,
    stage: object,
    *,
    variant: str,
    allow_cpu_unit_test: bool,
) -> MT3BatchForward:
    del sources, stage, variant, allow_cpu_unit_test
    loss = sum(parameter.square() for parameter in model.parameters())
    values = torch.full((1, 4), 0.2)
    return MT3BatchForward(
        loss=loss,
        candidate_thermal_smooth=values,
        candidate_exact_tmax=values,
        mean_total_variation=0.0,
        mean_binarization_penalty=0.0,
        softmin=0.2,
        diversity_penalty=0.0,
        maximum_projection_absolute_error=0.0,
        candidate_trace=None,
        probe_trace=None,
    )


def test_mt3_training_resumes_without_duplicate_updates(tmp_path: Path) -> None:
    config = MT3RunConfig(
        variant="FIELD_UNET",
        model_seed=11,
        task_seed=12,
        base_learning_rate=1.0e-4,
        total_updates=4,
        batch_size=1,
        checkpoint_interval=1,
        mode="unit",
        device="cpu",
    )
    first = run_mt3_training(
        config=config,
        output_dir=tmp_path,
        evaluator=_tiny_evaluator,
        model_factory=_tiny_factory,
        task_batch_provider=_tiny_tasks,
        maximum_updates_this_call=2,
    )
    assert first.status is MT3RunStatus.INCOMPLETE
    assert first.completed_updates == 2
    assert first.last_checkpoint is not None

    second = run_mt3_training(
        config=config,
        output_dir=tmp_path,
        evaluator=_tiny_evaluator,
        model_factory=_tiny_factory,
        task_batch_provider=_tiny_tasks,
        resume_checkpoint=first.last_checkpoint,
    )

    assert second.status is MT3RunStatus.PASS
    assert second.completed_updates == 4
    assert [record.update for record in second.records] == [0, 1, 2, 3]
    assert len(list(tmp_path.glob("checkpoint_*.pt"))) == 5


def test_lr_qualification_uses_locked_aggregate_order() -> None:
    rows = (
        MT3QualificationRun(1.0e-4, 1, True, 0.06, 0.12),
        MT3QualificationRun(1.0e-4, 2, True, 0.05, 0.11),
        MT3QualificationRun(3.0e-4, 1, True, 0.04, 0.10),
        MT3QualificationRun(3.0e-4, 2, False, float("inf"), float("inf")),
    )

    verdict = select_mt3_learning_rate(rows)

    assert verdict.production_authorized is True
    assert verdict.selected_learning_rate == pytest.approx(1.0e-4)
    assert verdict.reason == "more_valid_runs"


def test_lr_qualification_uses_smaller_lr_as_final_tie_break() -> None:
    rows = tuple(
        MT3QualificationRun(lr, seed, True, 0.05, 0.10)
        for lr in (1.0e-4, 3.0e-4)
        for seed in (1, 2)
    )

    verdict = select_mt3_learning_rate(rows)

    assert verdict.selected_learning_rate == pytest.approx(1.0e-4)
    assert verdict.reason == "smaller_learning_rate"
