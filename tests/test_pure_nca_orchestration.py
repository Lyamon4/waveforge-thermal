"""Проверки фазовых gates и CUDA benchmark pure-NCA spike."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from waveforge.ml.nca_training import NCAIterationRecord, NCARunResult, NCARunStatus
from waveforge.reproducibility import artifact_sha256


def _record(iteration: int, wall_seconds: float) -> NCAIterationRecord:
    return NCAIterationRecord(
        iteration=iteration,
        total_objective=1.0,
        thermal_smooth=1.0,
        exact_continuous_tmax=1.0,
        total_variation=0.0,
        binarization_penalty=0.1875,
        continuous_material_fraction=0.25,
        binary_material_fraction=0.0,
        projection_absolute_error=0.0,
        material_logit_mean=0.0,
        material_logit_std=0.0,
        material_logit_minimum=0.0,
        material_logit_maximum=0.0,
        material_std=0.0,
        hidden_state_rms=0.0,
        delta_state_rms=0.0,
        maximum_absolute_delta=0.0,
        maximum_absolute_state=0.0,
        gradient_norm_before_clipping=1.0,
        gradient_norm_after_clipping=1.0,
        conv3x3_weight_gradient_norm=0.0,
        conv1x1_weight_gradient_norm=1.0,
        all_parameter_gradients_finite=True,
        maximum_cg_iterations=1,
        maximum_explicit_relative_residual=1.0e-8,
        all_cg_converged=True,
        finite=True,
        wall_seconds=wall_seconds,
    )


def _run_result(samples: list[float]) -> NCARunResult:
    return NCARunResult(
        status=NCARunStatus.PASS,
        reason_codes=(),
        seed=20260830,
        mode="benchmark",
        learning_rate=1.0e-3,
        requested_iterations=len(samples),
        completed_iterations=len(samples),
        initial_objective=1.0,
        records=tuple(_record(index, value) for index, value in enumerate(samples)),
        solve_records=(),
        initial_model_hash="a" * 64,
        final_model_hash="b" * 64,
        final_continuous_design=None,
        final_binary_design=None,
    )


def test_qualification_cannot_start_before_all_preflight_artifacts_pass(
    tmp_path: Path,
) -> None:
    from waveforge.experiments.run_pure_nca_spike import (
        PreflightGateError,
        run_qualification_phase,
    )

    with pytest.raises(PreflightGateError, match="preflight"):
        run_qualification_phase(tmp_path)


def test_production_cannot_start_without_selected_lr(tmp_path: Path) -> None:
    from waveforge.experiments.run_pure_nca_spike import (
        QualificationGateError,
        run_production_phase,
    )

    config = Path("configs/pure_nca_spike.yaml")
    spec = Path(
        "docs/superpowers/specs/2026-08-29-pure-nca-physics-trained-spike-design.md"
    )
    (tmp_path / "protocol_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "config_sha256": artifact_sha256(config),
                "spec_sha256": artifact_sha256(spec),
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "environment.json",
        "initial_state_sanity.json",
        "determinism_preflight.json",
        "preflight_report.json",
        "complete_step_benchmark.json",
    ):
        (tmp_path / name).write_text('{"status":"PASS"}', encoding="utf-8")

    with pytest.raises(QualificationGateError, match="selected"):
        run_production_phase(tmp_path, seed=20260901)


def test_complete_step_benchmark_excludes_warmups_and_projects_costs() -> None:
    from waveforge.experiments.run_pure_nca_spike import benchmark_complete_steps

    sync_calls: list[str] = []
    reset_calls: list[str] = []
    runner_kwargs: dict[str, object] = {}
    samples = [90.0, 91.0, 92.0] + [float(value) for value in range(1, 11)]

    def fake_runner(**kwargs) -> NCARunResult:
        runner_kwargs.update(kwargs)
        hook = kwargs["iteration_start_hook"]
        for iteration in range(13):
            hook(iteration)
        return _run_result(samples)

    report = benchmark_complete_steps(
        sources=object(),
        training_runner=fake_runner,
        synchronizer=lambda: sync_calls.append("sync"),
        reset_peak_memory=lambda: reset_calls.append("reset"),
        peak_allocated_memory=lambda: 123,
        peak_reserved_memory=lambda: 456,
        timer=lambda: 10.0,
    )

    assert runner_kwargs["iterations"] == 13
    assert runner_kwargs["synchronize"] is not None
    assert runner_kwargs["clock"] is not None
    assert reset_calls == ["reset"]
    assert report["warmup_runs"] == 3
    assert report["measured_runs"] == 10
    assert report["samples_seconds"] == [float(value) for value in range(1, 11)]
    assert report["median_seconds"] == pytest.approx(5.5)
    assert report["p90_seconds"] == pytest.approx(9.1)
    assert report["mean_seconds"] == pytest.approx(5.5)
    assert report["standard_deviation_seconds"] == pytest.approx(2.8722813232690143)
    assert report["peak_allocated_bytes"] == 123
    assert report["peak_reserved_bytes"] == 456
    assert report["projected_qualification_seconds"] == pytest.approx(5.5 * 600)
    assert report["projected_production_seconds"] == pytest.approx(5.5 * 6000)


def test_benchmark_rejects_invalid_or_incomplete_training_result() -> None:
    from waveforge.experiments.run_pure_nca_spike import benchmark_complete_steps

    incomplete = replace(_run_result([1.0] * 13), completed_iterations=12)

    with pytest.raises(RuntimeError, match="13"):
        benchmark_complete_steps(
            sources=object(),
            training_runner=lambda **kwargs: incomplete,
            synchronizer=lambda: None,
            reset_peak_memory=lambda: None,
            peak_allocated_memory=lambda: 0,
            peak_reserved_memory=lambda: 0,
            timer=lambda: 0.0,
        )
