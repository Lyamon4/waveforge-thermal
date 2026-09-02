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


def summarize_mt3_checkpoint_rows(
    rows: list[dict[str, object]],
    *,
    completed_updates: int,
    variant: Literal["FIELD_UNET", "SENS_UNET"],
) -> MT3CheckpointSummary:
    """Summarize one frozen checkpoint from 32 solver-matched layouts."""
    if len(rows) != 32:
        raise ValueError("MT3 checkpoint summary requires exactly 32 rows")
    if completed_updates <= 0 or completed_updates % 500 != 0:
        raise ValueError("MT3 checkpoint updates must be a positive multiple of 500")
    indices = [int(row["task_index"]) for row in rows]
    if indices != list(range(32)):
        raise ValueError("MT3 checkpoint rows must follow task indices 0..31")

    r25_gaps: list[float] = []
    best4_gaps: list[float] = []
    exact_budget_count = 0
    invalid_count = 0
    for row in rows:
        candidate_solver = str(row["candidate_solver"])
        reference_solver = str(row["reference_solver"])
        if candidate_solver != reference_solver:
            raise ValueError("candidate and reference must use the same solver")
        if candidate_solver != "independent_scipy_64":
            raise ValueError("MT3 development rows require independent_scipy_64")
        reference = float(row["reference_tmax_scipy64"])
        r25 = float(row["r25_tmax_scipy64"])
        best4 = float(row["best4_tmax_scipy64"])
        valid = (
            math.isfinite(reference)
            and math.isfinite(r25)
            and math.isfinite(best4)
            and reference > 0.0
            and r25 > 0.0
            and best4 > 0.0
            and int(row["refinement_updates"]) == 25
        )
        if int(row["binary_cell_count"]) == 1024:
            exact_budget_count += 1
        if not valid:
            invalid_count += 1
            continue
        r25_gaps.append((r25 - reference) / reference)
        best4_gaps.append((best4 - reference) / reference)

    if len(r25_gaps) != 32 or len(best4_gaps) != 32:
        median_r25 = math.inf
        p90_r25 = math.inf
        worst_r25 = math.inf
        wins = 0
        median_best4 = math.inf
    else:
        r25_array = np.asarray(r25_gaps, dtype=np.float64)
        best4_array = np.asarray(best4_gaps, dtype=np.float64)
        median_r25 = float(np.median(r25_array))
        p90_r25 = float(np.quantile(r25_array, 0.9))
        worst_r25 = float(np.max(r25_array))
        wins = int(np.count_nonzero(r25_array < 0.0))
        median_best4 = float(np.median(best4_array))

    return MT3CheckpointSummary(
        completed_updates=completed_updates,
        variant=variant,
        split_name="validation",
        task_count=32,
        invalid_count=invalid_count,
        exact_budget_count=exact_budget_count,
        median_r25_relative_gap=median_r25,
        p90_r25_relative_gap=p90_r25,
        worst_r25_relative_gap=worst_r25,
        r25_win_count=wins,
        median_best4_relative_gap=median_best4,
    )


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
