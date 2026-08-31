"""Tests for shared-NCA sequential microbatch training."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from waveforge.ml.multitask_protocol import MultitaskStage
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.ml.multitask_training import (
    MultitaskForward,
    MultitaskRunConfig,
    MultitaskRunStatus,
    run_multitask_training,
)
from waveforge.physics.batched_cg import (
    BatchedCGConvergenceError,
    BatchedCGDiagnostics,
)
from waveforge.physics.cg import CGConvergenceError, CGDiagnostics


class OneParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def unit_config(*, updates: int, microbatch_size: int = 4) -> MultitaskRunConfig:
    return MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=updates,
        microbatch_size=microbatch_size,
        checkpoint_interval=1,
        mode="unit",
        device="cpu",
    )


def fake_task(value: float, index: int) -> SourceLayoutTask:
    return SourceLayoutTask(
        task_id=f"task-{index}",
        centers=((0.2, 0.6), (0.5, 0.7), (0.8, 0.6)),
        bounds=(
            (0.1, 0.3, 0.5, 0.7),
            (0.4, 0.6, 0.6, 0.8),
            (0.7, 0.9, 0.5, 0.7),
        ),
        sources=np.full((3, 64, 64), value, dtype=np.float64),
    )


def fake_evaluator(
    model: nn.Module,
    sources: Tensor,
    stage: MultitaskStage,
    *,
    allow_cpu_unit_test: bool,
) -> MultitaskForward:
    assert allow_cpu_unit_test
    assert stage.beta in {2.0, 4.0, 8.0}
    value = sources.mean()
    loss = next(model.parameters()) * value
    return MultitaskForward(
        loss=loss,
        thermal_smooth=float(loss.detach()),
        exact_tmax=float(loss.detach()),
        continuous_material_fraction=0.25,
        projection_absolute_error=0.0,
        solve_trace=None,
    )


def recovery_evaluator(
    model: nn.Module,
    sources: Tensor,
    stage: MultitaskStage,
    *,
    allow_cpu_unit_test: bool,
) -> MultitaskForward:
    assert not allow_cpu_unit_test
    assert stage.beta == 8.0
    value = sources.mean()
    loss = next(model.parameters()) * value
    return MultitaskForward(
        loss=loss,
        thermal_smooth=float(loss.detach()),
        exact_tmax=float(loss.detach()),
        continuous_material_fraction=0.25,
        projection_absolute_error=0.0,
        solve_trace=None,
    )


def model_factory(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return OneParameterModel().to(device)


def test_microbatch_averages_four_task_losses_before_one_adam_step(
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, int]] = []

    def task_provider(
        seed: int, update: int, microbatch_index: int
    ) -> SourceLayoutTask:
        assert seed == 5678
        calls.append((update, microbatch_index))
        return fake_task(float(microbatch_index + 1), microbatch_index)

    result = run_multitask_training(
        config=unit_config(updates=2),
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=tmp_path,
        model_factory=model_factory,
    )

    assert result.status is MultitaskRunStatus.PASS
    assert result.completed_updates == 2
    assert calls == [(0, i) for i in range(4)] + [(1, i) for i in range(4)]
    assert result.records[0].task_exposures == 4
    assert result.records[0].gradient_norm_before_clipping == pytest.approx(2.5)
    assert result.records[0].mean_total_objective == pytest.approx(2.5)
    assert (tmp_path / "checkpoint_000000.pt").is_file()


def test_resume_restores_model_optimizer_and_rng_exactly(tmp_path: Path) -> None:
    def task_provider(
        seed: int, update: int, microbatch_index: int
    ) -> SourceLayoutTask:
        value = float(1 + update + microbatch_index)
        return fake_task(value, update * 10 + microbatch_index)

    interrupted_dir = tmp_path / "interrupted"
    first = run_multitask_training(
        config=unit_config(updates=4, microbatch_size=2),
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=interrupted_dir,
        model_factory=model_factory,
        maximum_updates_this_call=2,
    )
    assert first.status is MultitaskRunStatus.INCOMPLETE
    assert first.last_checkpoint is not None

    resumed = run_multitask_training(
        config=unit_config(updates=4, microbatch_size=2),
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=interrupted_dir,
        model_factory=model_factory,
        resume_checkpoint=first.last_checkpoint,
    )
    uninterrupted = run_multitask_training(
        config=unit_config(updates=4, microbatch_size=2),
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=None,
        model_factory=model_factory,
    )

    assert resumed.status is MultitaskRunStatus.PASS
    assert resumed.completed_updates == 4
    assert resumed.final_model_hash == uninterrupted.final_model_hash
    assert [record.mean_total_objective for record in resumed.records] == pytest.approx(
        [record.mean_total_objective for record in uninterrupted.records]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_resume_keeps_cpu_rng_state_on_cpu(tmp_path: Path) -> None:
    config = MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=2,
        microbatch_size=1,
        checkpoint_interval=1,
        mode="unit",
        device="cuda",
    )

    def task_provider(
        seed: int, update: int, microbatch_index: int
    ) -> SourceLayoutTask:
        return fake_task(float(1 + update), update)

    first = run_multitask_training(
        config=config,
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=tmp_path,
        model_factory=model_factory,
        maximum_updates_this_call=1,
    )
    assert first.status is MultitaskRunStatus.INCOMPLETE
    assert first.last_checkpoint is not None

    resumed = run_multitask_training(
        config=config,
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=tmp_path,
        model_factory=model_factory,
        resume_checkpoint=first.last_checkpoint,
    )
    uninterrupted = run_multitask_training(
        config=config,
        task_provider=task_provider,
        evaluator=fake_evaluator,
        output_dir=None,
        model_factory=model_factory,
    )

    assert resumed.status is MultitaskRunStatus.PASS
    assert resumed.completed_updates == 2
    assert resumed.final_model_hash == uninterrupted.final_model_hash
    assert [record.mean_total_objective for record in resumed.records] == pytest.approx(
        [record.mean_total_objective for record in uninterrupted.records]
    )


def _synthetic_completed_pilot_checkpoint(tmp_path: Path) -> Path:
    source_dir = tmp_path / "source"
    source = run_multitask_training(
        config=unit_config(updates=1, microbatch_size=1),
        task_provider=lambda seed, update, microbatch: fake_task(1.0, update),
        evaluator=fake_evaluator,
        output_dir=source_dir,
        model_factory=model_factory,
    )
    assert source.last_checkpoint is not None
    payload = torch.load(source.last_checkpoint, map_location="cpu", weights_only=False)
    record = payload["records"][0]
    payload["config"]["total_updates"] = 1500
    payload["config"]["checkpoint_interval"] = 250
    payload["config"]["mode"] = "pilot"
    payload["config"]["device"] = "cuda"
    payload["completed_updates"] = 1500
    payload["records"] = [dict(record, update=update) for update in range(1500)]
    checkpoint = source_dir / "checkpoint_001500.pt"
    torch.save(payload, checkpoint)
    return checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_recovery_extends_only_a_completed_1500_update_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = _synthetic_completed_pilot_checkpoint(tmp_path)
    config = MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=3000,
        microbatch_size=1,
        checkpoint_interval=250,
        mode="recovery",
        device="cuda",
        schedule_id="recovery_final_decay",
        resume_from_total_updates=1500,
    )

    result = run_multitask_training(
        config=config,
        task_provider=lambda seed, update, microbatch: fake_task(1.0, update),
        evaluator=recovery_evaluator,
        output_dir=tmp_path / "recovery",
        model_factory=model_factory,
        resume_checkpoint=checkpoint,
        maximum_updates_this_call=1,
    )

    assert result.status is MultitaskRunStatus.INCOMPLETE
    assert result.completed_updates == 1501
    assert result.records[-1].update == 1500
    assert result.records[-1].stage_id == 4
    assert result.records[-1].learning_rate == 1.0e-4

    resumed = run_multitask_training(
        config=config,
        task_provider=lambda seed, update, microbatch: fake_task(1.0, update),
        evaluator=recovery_evaluator,
        output_dir=tmp_path / "recovery",
        model_factory=model_factory,
        resume_checkpoint=result.last_checkpoint,
        maximum_updates_this_call=1,
    )
    assert resumed.completed_updates == 1502
    assert resumed.records[-1].update == 1501


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_recovery_rejects_an_incomplete_source_checkpoint(tmp_path: Path) -> None:
    checkpoint = _synthetic_completed_pilot_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["completed_updates"] = 1499
    payload["records"] = payload["records"][:1499]
    torch.save(payload, checkpoint)
    config = MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=3000,
        microbatch_size=1,
        checkpoint_interval=250,
        mode="recovery",
        device="cuda",
        schedule_id="recovery_final_decay",
        resume_from_total_updates=1500,
    )

    with pytest.raises(ValueError, match="complete source checkpoint"):
        run_multitask_training(
            config=config,
            output_dir=tmp_path / "recovery",
            model_factory=model_factory,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_recovery_rejects_a_mismatched_declared_source_total(tmp_path: Path) -> None:
    checkpoint = _synthetic_completed_pilot_checkpoint(tmp_path)
    config = MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=3000,
        microbatch_size=1,
        checkpoint_interval=250,
        mode="recovery",
        device="cuda",
        schedule_id="recovery_final_decay",
        resume_from_total_updates=1499,
    )

    with pytest.raises(ValueError, match="locked recovery source total"):
        run_multitask_training(
            config=config,
            output_dir=tmp_path / "recovery",
            model_factory=model_factory,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=1,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ordinary_resume_rejects_a_total_update_extension(tmp_path: Path) -> None:
    checkpoint = _synthetic_completed_pilot_checkpoint(tmp_path)
    config = MultitaskRunConfig(
        model_seed=1234,
        task_seed=5678,
        total_updates=3000,
        microbatch_size=1,
        checkpoint_interval=250,
        mode="unit",
        device="cuda",
    )

    with pytest.raises(ValueError, match="locked run configuration"):
        run_multitask_training(
            config=config,
            task_provider=lambda seed, update, microbatch: fake_task(1.0, update),
            evaluator=fake_evaluator,
            output_dir=tmp_path / "rejected",
            model_factory=model_factory,
            resume_checkpoint=checkpoint,
            maximum_updates_this_call=1,
        )


def test_any_cg_failure_marks_run_invalid_without_optimizer_step() -> None:
    evaluator_calls = 0

    def task_provider(
        seed: int, update: int, microbatch_index: int
    ) -> SourceLayoutTask:
        return fake_task(float(microbatch_index + 1), microbatch_index)

    def failing_evaluator(
        model: nn.Module,
        sources: Tensor,
        stage: MultitaskStage,
        *,
        allow_cpu_unit_test: bool,
    ) -> MultitaskForward:
        nonlocal evaluator_calls
        evaluator_calls += 1
        if evaluator_calls == 2:
            raise CGConvergenceError(
                CGDiagnostics(
                    iterations=2000,
                    relative_residual=2.0e-5,
                    converged=False,
                    reason="maximum_iterations",
                )
            )
        return fake_evaluator(
            model,
            sources,
            stage,
            allow_cpu_unit_test=allow_cpu_unit_test,
        )

    result = run_multitask_training(
        config=unit_config(updates=1),
        task_provider=task_provider,
        evaluator=failing_evaluator,
        output_dir=None,
        model_factory=model_factory,
    )

    assert result.status is MultitaskRunStatus.INVALID_RUN
    assert result.reason_codes == ("CG_NONCONVERGENCE",)
    assert result.completed_updates == 0
    assert result.initial_model_hash == result.final_model_hash


def test_any_batched_cg_failure_marks_run_invalid_without_optimizer_step() -> None:
    def failing_evaluator(
        model: nn.Module,
        sources: Tensor,
        stage: MultitaskStage,
        *,
        allow_cpu_unit_test: bool,
    ) -> MultitaskForward:
        del model, sources, stage, allow_cpu_unit_test
        raise BatchedCGConvergenceError(
            BatchedCGDiagnostics(
                iterations=torch.tensor([[2000, 2000, 2000]]),
                relative_residuals=torch.tensor([[2.0e-5, 3.0e-5, 4.0e-5]]),
                converged=torch.tensor([[False, False, False]]),
            )
        )

    result = run_multitask_training(
        config=unit_config(updates=1, microbatch_size=1),
        task_provider=lambda seed, update, microbatch: fake_task(1.0, update),
        evaluator=failing_evaluator,
        output_dir=None,
        model_factory=model_factory,
    )

    assert result.status is MultitaskRunStatus.INVALID_RUN
    assert result.reason_codes == ("CG_NONCONVERGENCE",)
    assert result.completed_updates == 0
    assert result.initial_model_hash == result.final_model_hash


@pytest.mark.parametrize(
    ("field", "value"),
    [("total_updates", 0), ("microbatch_size", 0), ("checkpoint_interval", 0)],
)
def test_invalid_training_config_is_rejected(field: str, value: int) -> None:
    values = {
        "model_seed": 1,
        "task_seed": 2,
        "total_updates": 2,
        "microbatch_size": 2,
        "checkpoint_interval": 1,
        "mode": "unit",
        "device": "cpu",
    }
    values[field] = value

    with pytest.raises(ValueError):
        MultitaskRunConfig(**values)


def test_task_provider_type_is_callable() -> None:
    def provider(seed: int, update: int, microbatch: int) -> SourceLayoutTask:
        return fake_task(1.0, microbatch)

    typed_provider: Callable[[int, int, int], SourceLayoutTask] = provider
    assert typed_provider(1, 0, 0).task_id == "task-0"
