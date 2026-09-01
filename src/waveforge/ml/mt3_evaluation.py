"""Solver-consistent checkpoint selection and sealed MT3 development gate."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from waveforge.reproducibility import artifact_sha256

MT3DevelopmentStatus = Literal["MT3_DEVELOPMENT_GO", "MT3_DEVELOPMENT_NO_GO"]


@dataclass(frozen=True)
class MT3CheckpointSummary:
    completed_updates: int
    variant: Literal["FIELD_UNET", "SENS_UNET"]
    split_name: str
    task_count: int
    invalid_count: int
    exact_budget_count: int
    median_r25_relative_gap: float
    p90_r25_relative_gap: float
    worst_r25_relative_gap: float
    r25_win_count: int
    median_best4_relative_gap: float


@dataclass(frozen=True)
class MT3DevelopmentVerdict:
    status: MT3DevelopmentStatus
    test_authorized: bool
    median_gap: float
    p90_gap: float
    worst_gap: float
    win_count: int
    valid_count: int
    exact_budget_count: int
    exact_reason: str


def solver_consistent_gap(
    *,
    candidate_tmax: np.float64,
    reference_tmax: np.float64,
    candidate_solver_id: str,
    reference_solver_id: str,
) -> float:
    """Compute a relative gap only when both values share one solver path."""
    if candidate_solver_id != reference_solver_id:
        raise ValueError("candidate and reference must use the same solver")
    candidate = float(candidate_tmax)
    reference = float(reference_tmax)
    if (
        not math.isfinite(candidate)
        or not math.isfinite(reference)
        or candidate <= 0.0
        or reference <= 0.0
    ):
        raise ValueError("solver-consistent Tmax values must be finite and positive")
    return (candidate - reference) / reference


def select_mt3_checkpoint(
    summaries: list[MT3CheckpointSummary],
) -> MT3CheckpointSummary:
    """Select by R25 quality before any one-shot or display metric."""
    if not summaries:
        raise ValueError("checkpoint selection requires at least one summary")
    if any(summary.split_name != "validation" for summary in summaries):
        raise ValueError("checkpoint selection may use validation rows only")
    eligible: list[MT3CheckpointSummary] = []
    for summary in summaries:
        values = (
            summary.median_r25_relative_gap,
            summary.p90_r25_relative_gap,
            summary.worst_r25_relative_gap,
            summary.median_best4_relative_gap,
        )
        if (
            summary.variant == "SENS_UNET"
            and summary.task_count == 32
            and summary.invalid_count == 0
            and summary.exact_budget_count == 32
            and all(math.isfinite(value) for value in values)
        ):
            eligible.append(summary)
    if not eligible:
        raise ValueError("no eligible SENS_UNET validation checkpoint")
    return min(
        eligible,
        key=lambda summary: (
            summary.median_r25_relative_gap,
            summary.p90_r25_relative_gap,
            summary.median_best4_relative_gap,
            summary.invalid_count,
            summary.completed_updates,
        ),
    )


def classify_mt3_development(
    *,
    median_gap: float,
    p90_gap: float,
    worst_gap: float,
    wins: int,
    valid_count: int,
    exact_budget_count: int,
) -> MT3DevelopmentVerdict:
    """Apply all preregistered SENS_UNET_BEST4_R25 development thresholds."""
    values = (median_gap, p90_gap, worst_gap)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("development gaps must be finite")
    if not 0 <= wins <= 32:
        raise ValueError("development win count must lie in [0,32]")
    counts_valid = valid_count == 32 and exact_budget_count == 32
    quality_valid = (
        median_gap <= 0.02 and p90_gap <= 0.07 and worst_gap <= 0.20 and wins >= 8
    )
    passed = counts_valid and quality_valid
    if passed:
        status: MT3DevelopmentStatus = "MT3_DEVELOPMENT_GO"
        reason = "all numerical, material-budget, quality, and win gates passed"
    else:
        status = "MT3_DEVELOPMENT_NO_GO"
        reason = "one or more locked development gates failed"
    return MT3DevelopmentVerdict(
        status=status,
        test_authorized=passed,
        median_gap=median_gap,
        p90_gap=p90_gap,
        worst_gap=worst_gap,
        win_count=wins,
        valid_count=valid_count,
        exact_budget_count=exact_budget_count,
        exact_reason=reason,
    )


def _bundle_sha256(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def build_test_authorization_bundle(
    *,
    verdict: MT3DevelopmentVerdict,
    implementation_commit: str,
    artifacts: dict[str, Path],
) -> dict[str, object]:
    """Hash every frozen input before any sealed test registry can be opened."""
    if len(implementation_commit) != 40 or any(
        character not in "0123456789abcdef" for character in implementation_commit
    ):
        raise ValueError("implementation commit must be a lowercase Git SHA")
    if not artifacts:
        raise ValueError("authorization bundle requires frozen artifacts")
    payload: dict[str, object] = {
        "schema_version": 1,
        "development_status": verdict.status,
        "test_authorized": verdict.test_authorized,
        "implementation_commit": implementation_commit,
        "artifacts": {
            name: {"path": str(path), "sha256": artifact_sha256(path)}
            for name, path in sorted(artifacts.items())
        },
        "test_id_accessed": False,
        "test_ood_accessed": False,
    }
    payload["bundle_sha256"] = _bundle_sha256(payload)
    return payload


def load_sealed_registry(
    registry_path: Path,
    *,
    authorization_path: Path,
    expected_bundle_sha256: str,
) -> object:
    """Read authorization and verify its hash before touching sealed rows."""
    try:
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("test registry remains sealed") from error
    observed_hash = authorization.pop("bundle_sha256", None)
    if (
        observed_hash != expected_bundle_sha256
        or _bundle_sha256(authorization) != observed_hash
        or authorization.get("test_authorized") is not True
        or authorization.get("development_status") != "MT3_DEVELOPMENT_GO"
    ):
        raise PermissionError("test registry remains sealed")
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("authorized test registry is unreadable") from error
