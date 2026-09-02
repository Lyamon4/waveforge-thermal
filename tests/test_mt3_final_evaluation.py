from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from waveforge.ml.mt3_evaluation import (
    MT3DevelopmentVerdict,
    build_test_authorization_bundle,
)
from waveforge.ml.mt3_final_evaluation import (
    classify_mt3_test_split,
    load_authorized_task_splits,
    median_gap_bootstrap,
    registered_baseline_jobs,
)


def _authorized_bundle(tmp_path: Path) -> dict[str, object]:
    frozen = tmp_path / "frozen.pt"
    frozen.write_bytes(b"frozen-model")
    verdict = MT3DevelopmentVerdict(
        status="MT3_DEVELOPMENT_GO",
        test_authorized=True,
        median_gap=-0.04,
        p90_gap=-0.01,
        worst_gap=0.0,
        win_count=32,
        valid_count=32,
        exact_budget_count=32,
        exact_reason="passed",
    )
    return build_test_authorization_bundle(
        verdict=verdict,
        implementation_commit="a" * 40,
        artifacts={"frozen.pt": frozen},
    )


def test_sealed_split_factory_is_not_called_before_authorization_validation(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps({"test_authorized": True, "bundle_sha256": "wrong"}),
        encoding="utf-8",
    )
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    with pytest.raises(PermissionError, match="sealed"):
        load_authorized_task_splits(
            authorization,
            expected_bundle_sha256="0" * 64,
            split_factory=factory,
        )

    assert calls == 0


def test_valid_authorization_opens_split_factory_exactly_once(tmp_path: Path) -> None:
    payload = _authorized_bundle(tmp_path)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(json.dumps(payload), encoding="utf-8")
    sentinel = object()
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return sentinel

    result = load_authorized_task_splits(
        authorization,
        expected_bundle_sha256=str(payload["bundle_sha256"]),
        split_factory=factory,
    )

    assert result is sentinel
    assert calls == 1


def test_registered_jobs_cover_strong_single_and_adam_multistart() -> None:
    jobs = registered_baseline_jobs(
        id_task_ids=tuple(f"id-{index}" for index in range(32)),
        ood_task_ids=tuple(f"ood-{index}" for index in range(16)),
    )

    single = [job for job in jobs if job.start_index == 0]
    extra = [job for job in jobs if job.start_index > 0]
    assert len(single) == 96
    assert {(job.method, job.split) for job in single} == {
        ("ADAM", "test_id"),
        ("ADAM", "test_ood"),
        ("MMA", "test_id"),
        ("MMA", "test_ood"),
    }
    assert len(extra) == 48
    assert {job.method for job in extra} == {"ADAM"}
    assert {job.start_index for job in extra} == {1, 2, 3}
    assert all(job.task_index < 8 for job in extra)
    assert len({job.job_id for job in jobs}) == len(jobs)


def test_bootstrap_and_locked_id_verdict_use_full_distribution() -> None:
    gaps = np.linspace(-0.08, -0.01, 32, dtype=np.float64)
    bootstrap = median_gap_bootstrap(gaps, split="test_id")
    verdict = classify_mt3_test_split(
        gaps=gaps,
        invalid_count=0,
        exact_budget_count=32,
        equivalent_evaluation_speedup=20.0,
        split="test_id",
        bootstrap=bootstrap,
    )

    assert bootstrap.seed == 2026092401
    assert bootstrap.resamples == 10_000
    assert bootstrap.upper_bound < 0.0
    assert verdict.status == "MT3_BEATS_SINGLE_START_ID"
    assert verdict.win_count == 32


def test_invalid_or_weak_test_distribution_cannot_receive_go() -> None:
    weak = np.full(32, 0.08, dtype=np.float64)
    bootstrap = median_gap_bootstrap(weak, split="test_id")

    invalid = classify_mt3_test_split(
        gaps=weak,
        invalid_count=1,
        exact_budget_count=31,
        equivalent_evaluation_speedup=100.0,
        split="test_id",
        bootstrap=bootstrap,
    )
    no_go = classify_mt3_test_split(
        gaps=weak,
        invalid_count=0,
        exact_budget_count=32,
        equivalent_evaluation_speedup=100.0,
        split="test_id",
        bootstrap=bootstrap,
    )

    assert invalid.status == "MT3_INVALID_RUN_ID"
    assert no_go.status == "MT3_NO_GO_ID"
