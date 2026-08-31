"""Tests for paid-A100 campaign gates and CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import waveforge.experiments.run_multitask_nca as campaign
from waveforge.experiments.run_multitask_nca import (
    BenchmarkCandidate,
    MultitaskGateError,
    PilotStatus,
    RecoveryStatus,
    assemble_production_payload,
    build_parser,
    build_recovery_validation_tasks,
    calculate_runtime_gate,
    classify_pilot,
    classify_recovery,
    lock_runtime_budget_amendment,
    prepare_recovery_source,
    registered_test_baseline_jobs,
    run_recovery,
    select_microbatch,
    validate_production_gate,
)


def test_runtime_gate_requires_at_least_5000_updates_per_seed() -> None:
    verdict = calculate_runtime_gate(
        seconds_per_update=2.0,
        remaining_hours=6.0,
    )
    assert verdict.production_authorized is False
    assert verdict.updates_per_seed == 3600


def test_runtime_gate_caps_each_seed_at_15000_updates() -> None:
    verdict = calculate_runtime_gate(
        seconds_per_update=0.1,
        remaining_hours=6.0,
    )
    assert verdict.production_authorized is True
    assert verdict.updates_per_seed == 15_000


def test_eight_hour_budget_fits_minimum_updates_at_measured_a100_rate() -> None:
    verdict = calculate_runtime_gate(
        seconds_per_update=1.62,
        remaining_hours=8.0,
    )
    assert verdict.production_authorized is True
    assert verdict.updates_per_seed == 5925


def test_budget_amendment_preserves_original_benchmark_and_fails_after_pilot(
    tmp_path: Path,
) -> None:
    original = {
        "schema_version": 1,
        "status": "PASS",
        "selected_microbatch_size": 1,
        "selected_median_seconds_per_update": 1.62,
        "production_updates_per_seed": 4444,
        "production_runtime_authorized": False,
        "remaining_training_hours": 6.0,
        "candidates": [],
    }
    (tmp_path / "benchmark_verdict.json").write_text(
        json.dumps(original), encoding="utf-8"
    )

    amended = lock_runtime_budget_amendment(
        tmp_path,
        production_training_hours=8.0,
        maximum_campaign_cost_usd=7.0,
        hourly_cost_usd=0.633,
    )

    assert amended["production_updates_per_seed"] == 5925
    assert amended["production_runtime_authorized"] is True
    preserved = json.loads(
        (tmp_path / "benchmark_verdict_original.json").read_text(encoding="utf-8")
    )
    assert preserved == original
    amendment = json.loads(
        (tmp_path / "runtime_budget_amendment.json").read_text(encoding="utf-8")
    )
    assert amendment["prospective_before_pilot"] is True
    assert amendment["maximum_campaign_cost_usd"] == 7.0

    (tmp_path / "pilot_verdict.json").write_text("{}", encoding="utf-8")
    with pytest.raises(MultitaskGateError, match="before pilot"):
        lock_runtime_budget_amendment(
            tmp_path,
            production_training_hours=8.0,
            maximum_campaign_cost_usd=7.0,
            hourly_cost_usd=0.633,
        )


def test_test_baseline_registry_uses_locked_tasks_and_multistarts() -> None:
    jobs = registered_test_baseline_jobs()
    single = [job for job in jobs if job["family"] == "single_start"]
    multi = [job for job in jobs if job["family"] == "multistart_challenge"]

    assert len(single) == 32
    assert {job["start_index"] for job in single} == {0}
    assert len(multi) == 16 * 4
    assert {job["split"] for job in multi} == {"test_id", "test_ood"}
    assert {job["start_index"] for job in multi} == {0, 1, 2, 3}


def test_parallel_production_payload_requires_all_registered_seeds() -> None:
    shards = [
        {
            "status": "PASS",
            "seed": seed,
            "completed_updates": 5925,
            "training_wall_seconds": 100.0,
            "selected_checkpoint": f"checkpoint_{seed}.pt",
            "frozen_checkpoint": f"frozen_seed_{seed}.pt",
            "frozen_sha256": "a" * 64,
        }
        for seed in (2026083102, 2026083103, 2026083104)
    ]
    payload = assemble_production_payload(
        shards,
        updates_per_seed=5925,
        microbatch_size=1,
        training_hours_cap=8.0,
        worker_count=3,
    )
    assert payload["status"] == "PASS"
    assert payload["production_seeds"] == [2026083102, 2026083103, 2026083104]
    assert payload["worker_count"] == 3
    assert payload["test_sets_accessed"] is False

    with pytest.raises(MultitaskGateError, match="registered seeds"):
        assemble_production_payload(
            shards[:2],
            updates_per_seed=5925,
            microbatch_size=1,
            training_hours_cap=8.0,
            worker_count=2,
        )


def test_microbatch_selection_uses_throughput_and_two_percent_tie_break() -> None:
    selected = select_microbatch(
        [
            BenchmarkCandidate(1, 1.0, 1.1, 1.00, True, 100),
            BenchmarkCandidate(2, 1.9, 2.0, 1.04, True, 200),
            BenchmarkCandidate(4, 3.8, 4.0, 1.052, True, 400),
        ]
    )
    assert selected.microbatch_size == 2


def test_invalid_benchmark_candidate_cannot_be_selected() -> None:
    selected = select_microbatch(
        [
            BenchmarkCandidate(1, 1.0, 1.1, 1.0, True, 100),
            BenchmarkCandidate(4, 1.0, 1.1, 4.0, False, 400),
        ]
    )
    assert selected.microbatch_size == 1


def test_pilot_gate_distinguishes_go_conditional_kill_and_invalid() -> None:
    common = {
        "numerically_valid": True,
        "projection_valid": True,
        "binary_budget_valid": True,
        "validation_improved": True,
        "matched_condition_wins": 24,
        "source_independent": False,
    }
    assert classify_pilot(median_gradient_gap=0.14, **common) is PilotStatus.PILOT_GO
    assert (
        classify_pilot(median_gradient_gap=0.18, **common)
        is PilotStatus.PILOT_CONDITIONAL
    )
    assert classify_pilot(median_gradient_gap=0.21, **common) is PilotStatus.PILOT_KILL
    assert (
        classify_pilot(
            median_gradient_gap=0.1, **(common | {"numerically_valid": False})
        )
        is PilotStatus.INVALID_RUN
    )


def test_recovery_gate_requires_every_locked_effect_condition() -> None:
    common = {
        "numerically_valid": True,
        "completed_updates": 3000,
        "binary_budget_valid": True,
        "matched_condition_wins": 23,
        "source_independent": False,
        "selected_median_tmax": 0.19,
        "original_median_tmax": 0.2035900680052531,
        "median_gradient_gap": 0.15,
    }
    assert classify_recovery(**common) is RecoveryStatus.RECOVERY_GO
    assert (
        classify_recovery(**(common | {"matched_condition_wins": 22}))
        is RecoveryStatus.RECOVERY_NO_GO
    )
    assert (
        classify_recovery(**(common | {"selected_median_tmax": 0.21}))
        is RecoveryStatus.RECOVERY_NO_GO
    )
    assert (
        classify_recovery(**(common | {"median_gradient_gap": 0.1500000001}))
        is RecoveryStatus.RECOVERY_NO_GO
    )
    assert (
        classify_recovery(**(common | {"completed_updates": 2999}))
        is RecoveryStatus.INVALID_RUN
    )


def test_recovery_source_hash_is_enforced_before_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source_checkpoint = source / "pilot" / "checkpoint_001500.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"immutable pilot checkpoint")
    (source / "pilot_verdict.json").write_text(
        json.dumps(
            {
                "status": "PILOT_KILL",
                "selected_checkpoint": "pilot/checkpoint_001500.pt",
                "completed_updates": 1500,
            }
        ),
        encoding="utf-8",
    )
    recovery = tmp_path / "recovery"

    with pytest.raises(MultitaskGateError, match="SHA-256"):
        prepare_recovery_source(source, recovery)

    original_hash = campaign.artifact_sha256

    def accepted_hash(path: Path) -> str:
        if path == source_checkpoint or path == recovery / "checkpoint_001500.pt":
            return campaign.RECOVERY_SOURCE_CHECKPOINT_SHA256
        return original_hash(path)

    monkeypatch.setattr(campaign, "artifact_sha256", accepted_hash)
    copied, provenance = prepare_recovery_source(source, recovery)

    assert copied.read_bytes() == source_checkpoint.read_bytes()
    assert provenance["source_checkpoint_sha256"] == (
        campaign.RECOVERY_SOURCE_CHECKPOINT_SHA256
    )
    assert provenance["source_pilot_status"] == "PILOT_KILL"
    assert provenance["test_splits_accessed"] is False


def test_recovery_validation_registry_never_constructs_test_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def fake_sample(seed: int, index: int) -> object:
        calls.append((seed, index))
        return object()

    monkeypatch.setattr(campaign, "sample_primary_task", fake_sample)
    tasks = build_recovery_validation_tasks()

    assert len(tasks) == 32
    assert calls == [(campaign.VALIDATION_SEED, index) for index in range(32)]


def test_recovery_orchestration_uses_validation_only_and_never_starts_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    recovery = tmp_path / "recovery"
    source.mkdir()
    (source / "benchmark_verdict.json").write_text(
        json.dumps({"selected_median_seconds_per_update": 1.0}), encoding="utf-8"
    )
    (source / "pilot_verdict.json").write_text(
        json.dumps({"microbatch_size": 1}), encoding="utf-8"
    )

    def fake_prepare(source_dir: Path, recovery_dir: Path) -> tuple[Path, dict]:
        assert source_dir == source
        recovery_dir.mkdir()
        checkpoint = recovery_dir / "checkpoint_001500.pt"
        checkpoint.write_bytes(b"source")
        return checkpoint, {"source_checkpoint_sha256": "a" * 64}

    final_checkpoint = recovery / "checkpoint_003000.pt"

    def fake_training(**kwargs: object) -> SimpleNamespace:
        final_checkpoint.write_bytes(b"final")
        return SimpleNamespace(
            status=campaign.MultitaskRunStatus.PASS,
            completed_updates=3000,
            last_checkpoint=final_checkpoint,
            reason_codes=(),
        )

    validation_tasks = tuple(object() for _ in range(32))

    def fake_checkpoint_evaluation(*args: object, **kwargs: object) -> SimpleNamespace:
        call_index = fake_checkpoint_evaluation.calls
        fake_checkpoint_evaluation.calls += 1
        peak = 0.19 if call_index == 0 else 0.20
        tasks = tuple(
            SimpleNamespace(
                peak_temperature=peak,
                binary_material_fraction=0.25,
                binary_design=np.asarray([1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]),
            )
            for index in range(32)
        )
        return SimpleNamespace(tasks=tasks)

    fake_checkpoint_evaluation.calls = 0

    monkeypatch.setattr(campaign, "prepare_recovery_source", fake_prepare)
    monkeypatch.setattr(campaign, "configure_cuda_reproducibility", lambda seed: None)
    monkeypatch.setattr(campaign, "run_multitask_training", fake_training)
    monkeypatch.setattr(
        campaign, "build_recovery_validation_tasks", lambda: validation_tasks
    )
    monkeypatch.setattr(
        campaign,
        "_read_recovery_gradient_references",
        lambda source_dir, tasks: {"task": 0.1},
    )
    monkeypatch.setattr(
        campaign,
        "_evaluate_recovery_checkpoints",
        lambda recovery_dir, tasks, references: (
            final_checkpoint,
            [
                {
                    "checkpoint": final_checkpoint.name,
                    "peak_summary": {"median_peak": 0.19},
                    "gradient_gap_summary": {"median_relative_gap": 0.10},
                }
            ],
        ),
    )
    monkeypatch.setattr(
        campaign, "evaluate_frozen_checkpoint", fake_checkpoint_evaluation
    )
    monkeypatch.setattr(campaign, "artifact_sha256", lambda path: "b" * 64)
    monkeypatch.setattr(
        campaign,
        "build_frozen_splits",
        lambda: pytest.fail("recovery accessed frozen ID/OOD task construction"),
    )
    monkeypatch.setattr(
        campaign,
        "run_production",
        lambda path: pytest.fail("recovery started production"),
    )

    verdict = run_recovery(recovery, source)

    assert verdict["status"] == "RECOVERY_GO"
    assert verdict["completed_updates"] == 3000
    assert verdict["new_global_update_range"] == [1500, 2999]
    assert verdict["test_splits_accessed"] is False
    assert verdict["production_authorized"] is True
    assert verdict["production_started"] is False


def test_production_requires_benchmark_and_pilot_go(tmp_path: Path) -> None:
    with pytest.raises(MultitaskGateError, match="benchmark"):
        validate_production_gate(tmp_path)

    (tmp_path / "benchmark_verdict.json").write_text(
        json.dumps({"production_runtime_authorized": True}),
        encoding="utf-8",
    )
    with pytest.raises(MultitaskGateError, match="pilot"):
        validate_production_gate(tmp_path)

    (tmp_path / "pilot_verdict.json").write_text(
        json.dumps({"status": "PILOT_CONDITIONAL"}),
        encoding="utf-8",
    )
    with pytest.raises(MultitaskGateError, match="PILOT_GO"):
        validate_production_gate(tmp_path)

    (tmp_path / "pilot_verdict.json").write_text(
        json.dumps({"status": "PILOT_GO"}),
        encoding="utf-8",
    )
    validate_production_gate(tmp_path)


@pytest.mark.parametrize(
    "phase",
    [
        "preflight",
        "benchmark",
        "budget",
        "pilot",
        "recovery",
        "production-lock",
        "production-seed",
        "production-finalize",
        "production",
        "test",
        "hashes",
    ],
)
def test_parser_accepts_every_locked_phase(phase: str, tmp_path: Path) -> None:
    parser = build_parser()
    parsed = parser.parse_args(["--phase", phase, "--output", str(tmp_path)])
    assert parsed.phase == phase


def test_parser_accepts_parallel_seed_and_worker_count(tmp_path: Path) -> None:
    parsed = build_parser().parse_args(
        [
            "--phase",
            "production-seed",
            "--output",
            str(tmp_path),
            "--seed",
            "2026083103",
            "--worker-count",
            "3",
        ]
    )
    assert parsed.seed == 2026083103
    assert parsed.worker_count == 3


def test_parser_accepts_a_separate_recovery_source_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    parsed = build_parser().parse_args(
        [
            "--phase",
            "recovery",
            "--output",
            str(tmp_path / "recovery"),
            "--source-output",
            str(source),
        ]
    )
    assert parsed.source_output == source
