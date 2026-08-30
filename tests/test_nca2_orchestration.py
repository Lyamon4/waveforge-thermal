from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from waveforge.experiments.run_nca2_stabilization import (
    benchmark_revised_loop,
    build_protocol_manifest,
    validate_runtime_gate,
)
from waveforge.ml.nca_training import NCARunStatus
from waveforge.reproducibility import artifact_sha256


def _benchmark_result(samples: list[float]):
    return SimpleNamespace(
        status=NCARunStatus.PASS,
        completed_iterations=len(samples),
        records=tuple(SimpleNamespace(wall_seconds=value) for value in samples),
    )


def test_revised_benchmark_projects_locked_campaign_from_mean() -> None:
    samples = [90.0, 91.0, 92.0, *map(float, range(1, 11))]
    result = _benchmark_result(samples)
    reset_calls: list[str] = []

    def fake_runner(**kwargs):
        hook = kwargs["iteration_start_hook"]
        for iteration in range(13):
            hook(iteration)
        return result

    report = benchmark_revised_loop(
        sources=object(),
        training_runner=fake_runner,
        synchronizer=lambda: None,
        reset_peak_memory=lambda: reset_calls.append("reset"),
        peak_allocated_memory=lambda: 123,
        peak_reserved_memory=lambda: 456,
    )

    assert report["warmup_steps"] == 3
    assert report["measured_steps"] == 10
    assert report["samples_seconds"] == [float(value) for value in range(1, 11)]
    assert report["mean_step_seconds"] == pytest.approx(5.5)
    assert report["qualification_updates"] == 4200
    assert report["production_updates"] == 4500
    assert report["total_updates"] == 8700
    assert report["projected_gpu_hours"] == pytest.approx(5.5 * 8700 / 3600)
    assert report["peak_allocated_bytes"] == 123
    assert report["peak_reserved_bytes"] == 456
    assert reset_calls == ["reset"]


@pytest.mark.parametrize(
    ("hours", "expected_status", "authorized"),
    [
        (6.6, "PASS", True),
        (6.6000001, "NCA2_RUNTIME_REVIEW_REQUIRED", False),
    ],
)
def test_runtime_gate_is_inclusive_only_at_locked_cap(
    hours: float,
    expected_status: str,
    authorized: bool,
) -> None:
    report = {"projected_gpu_hours": hours}

    gated = validate_runtime_gate(report)

    assert gated["status"] == expected_status
    assert gated["qualification_authorized"] is authorized
    assert gated["maximum_projected_gpu_hours"] == 6.6


def test_revised_benchmark_rejects_invalid_or_incomplete_result() -> None:
    invalid = _benchmark_result([1.0] * 13)
    invalid.status = NCARunStatus.INVALID_RUN
    with pytest.raises(RuntimeError, match="13 PASS"):
        benchmark_revised_loop(
            sources=object(),
            training_runner=lambda **kwargs: invalid,
            synchronizer=lambda: None,
            reset_peak_memory=lambda: None,
            peak_allocated_memory=lambda: 0,
            peak_reserved_memory=lambda: 0,
        )


def test_protocol_manifest_keeps_old_result_and_exact_provenance(
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.yaml"
    spec = tmp_path / "spec.md"
    old_verdict = tmp_path / "old_verdict.json"
    config.write_text("scope: locked\n", encoding="utf-8", newline="\n")
    spec.write_text("# locked\n", encoding="utf-8", newline="\n")
    old_verdict.write_text(
        json.dumps({"status": "NCA_NO_GO_EFFECT"}),
        encoding="utf-8",
        newline="\n",
    )

    manifest = build_protocol_manifest(
        benchmark={"status": "PASS", "qualification_authorized": True},
        config_path=config,
        spec_path=spec,
        old_verdict_path=old_verdict,
        implementation_git_sha="a" * 40,
        determinism={"mode": "strict", "seed": 20260910},
    )

    assert manifest["status"] == "PASS"
    assert manifest["config_sha256"] == artifact_sha256(config)
    assert manifest["spec_sha256"] == artifact_sha256(spec)
    assert manifest["old_experiment_status"] == "NCA_NO_GO_EFFECT"
    assert manifest["old_verdict_sha256"] == artifact_sha256(old_verdict)
    assert manifest["implementation_git_sha"] == "a" * 40
    assert manifest["determinism"] == {"mode": "strict", "seed": 20260910}

    incomplete = _benchmark_result([1.0] * 12)
    with pytest.raises(RuntimeError, match="13 PASS"):
        benchmark_revised_loop(
            sources=object(),
            training_runner=lambda **kwargs: incomplete,
            synchronizer=lambda: None,
            reset_peak_memory=lambda: None,
            peak_allocated_memory=lambda: 0,
            peak_reserved_memory=lambda: 0,
        )
