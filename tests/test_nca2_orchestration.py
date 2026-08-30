from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.experiments.run_nca2_stabilization import (
    benchmark_revised_loop,
    build_protocol_manifest,
    evaluate_qualification_checkpoints,
    execute_qualification_runs,
    freeze_nca2_checkpoint,
    validate_production_checkpoint_registry,
    validate_production_seed,
    validate_runtime_gate,
)
from waveforge.ml.nca_training import NCARunStatus, initialize_nca, model_state_sha256
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


def test_qualification_executes_exact_cartesian_registry(tmp_path: Path) -> None:
    calls: list[dict] = []
    configured_seeds: list[int] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status=NCARunStatus.PASS,
            completed_iterations=700,
            initial_model_hash=f"initial-{kwargs['seed']}",
            records=tuple(SimpleNamespace(iteration=index) for index in range(700)),
        )

    runs = execute_qualification_runs(
        output_dir=tmp_path,
        sources=object(),
        training_runner=fake_runner,
        synchronizer=lambda: None,
        seed_configurator=lambda seed: configured_seeds.append(seed),
    )

    assert [(run.protocol_id, run.seed) for run in runs] == [
        ("A", 20260901),
        ("A", 20260902),
        ("A", 20260903),
        ("B", 20260901),
        ("B", 20260902),
        ("B", 20260903),
    ]
    assert all(call["iterations"] == 700 for call in calls)
    assert all(call["checkpoint_interval"] == 50 for call in calls)
    assert all(call["mode"] == "qualification" for call in calls)
    assert configured_seeds == [
        20260901,
        20260902,
        20260903,
        20260901,
        20260902,
        20260903,
    ]
    for seed in (20260901, 20260902, 20260903):
        hashes = [run.initial_model_hash for run in runs if run.seed == seed]
        assert hashes == [f"initial-{seed}", f"initial-{seed}"]


def test_qualification_checkpoint_registry_is_exact_and_independent(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for completed in range(50, 701, 50):
        (run_dir / f"checkpoint_{completed:06d}.pt").write_bytes(b"checkpoint")

    calls: list[tuple[int, str]] = []

    def fake_finalizer(**kwargs):
        del kwargs
        design = np.zeros((64, 64), dtype=np.float64)
        design[:, :16] = 1.0
        return SimpleNamespace(continuous_design=design, binary_design=design)

    def fake_verifier(candidate_id, design, *, fidelity):
        del design
        calls.append((int(candidate_id.rsplit("_", 1)[1]), fidelity))
        return SimpleNamespace(worst_peak=0.2 - len(calls) * 0.001)

    diagnostics = evaluate_qualification_checkpoints(
        protocol_id="B",
        seed=20260901,
        run_dir=run_dir,
        sources=object(),
        finalizer=fake_finalizer,
        verifier=fake_verifier,
    )

    assert [row.completed_updates for row in diagnostics] == [
        500,
        550,
        600,
        650,
        700,
    ]
    assert calls == [
        (500, "low_64"),
        (550, "low_64"),
        (600, "low_64"),
        (650, "low_64"),
        (700, "low_64"),
    ]
    assert diagnostics[-1].binary_fraction == 0.25

    (run_dir / "checkpoint_000650.pt").unlink()
    with pytest.raises(RuntimeError, match="checkpoint registry"):
        evaluate_qualification_checkpoints(
            protocol_id="B",
            seed=20260901,
            run_dir=run_dir,
            sources=object(),
            finalizer=fake_finalizer,
            verifier=fake_verifier,
        )


def test_production_seed_registry_rejects_replacement_seed() -> None:
    for seed in (20260911, 20260912, 20260913):
        assert validate_production_seed(seed) == seed
    with pytest.raises(ValueError, match="unregistered production seed"):
        validate_production_seed(20260914)


def test_production_checkpoint_registry_requires_exact_post_update_series(
    tmp_path: Path,
) -> None:
    for completed in range(50, 1501, 50):
        (tmp_path / f"checkpoint_{completed:06d}.pt").write_bytes(b"checkpoint")

    final = validate_production_checkpoint_registry(tmp_path)

    assert final.name == "checkpoint_001500.pt"
    (tmp_path / "checkpoint_001450.pt").unlink()
    with pytest.raises(RuntimeError, match="checkpoint registry"):
        validate_production_checkpoint_registry(tmp_path)


def test_freeze_nca2_checkpoint_uses_post_update_metadata_and_strict_threshold(
    tmp_path: Path,
) -> None:
    model = initialize_nca(17, torch.device("cpu"))
    checkpoint = {
        "completed_updates": 1500,
        "last_iteration": 1499,
        "model_state_sha256": model_state_sha256(model),
        "model_state": model.state_dict(),
    }
    path = tmp_path / "checkpoint_001500.pt"
    torch.save(checkpoint, path)

    frozen = freeze_nca2_checkpoint(
        checkpoint_path=path,
        sources=gate2_source_batch(device=torch.device("cpu")),
        completed_updates=1500,
        protocol_id="B",
    )

    assert frozen.continuous_design.shape == (64, 64)
    assert frozen.binary_design.shape == (64, 64)
    assert np.array_equal(
        frozen.binary_design,
        (frozen.continuous_design >= 0.5).astype(np.float64),
    )
