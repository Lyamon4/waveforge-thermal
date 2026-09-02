"""Fail-closed split access, registered baselines, and frozen MT3 verdicts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

from waveforge.reproducibility import artifact_sha256

SplitName = Literal["test_id", "test_ood"]
T = TypeVar("T")


@dataclass(frozen=True)
class BaselineJob:
    job_id: str
    method: Literal["ADAM", "MMA"]
    split: SplitName
    task_index: int
    task_id: str
    start_index: int
    seed: int
    evaluations: int = 600


@dataclass(frozen=True)
class MedianGapBootstrap:
    split: SplitName
    seed: int
    resamples: int
    median: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class MT3TestVerdict:
    status: str
    split: SplitName
    task_count: int
    invalid_count: int
    exact_budget_count: int
    median_gap: float
    p90_gap: float
    worst_gap: float
    win_count: int
    win_rate: float
    equivalent_evaluation_speedup: float
    bootstrap_lower: float
    bootstrap_upper: float


def _bundle_sha256(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def load_authorized_task_splits(
    authorization_path: Path,
    *,
    expected_bundle_sha256: str,
    split_factory: Callable[[], T],
) -> T:
    """Validate the frozen authorization before invoking the sealed factory."""
    try:
        payload = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError("test splits remain sealed") from error
    observed = payload.pop("bundle_sha256", None)
    authorized = (
        observed == expected_bundle_sha256
        and observed == _bundle_sha256(payload)
        and payload.get("test_authorized") is True
        and payload.get("development_status") == "MT3_DEVELOPMENT_GO"
        and payload.get("test_id_accessed") is False
        and payload.get("test_ood_accessed") is False
    )
    if not authorized:
        raise PermissionError("test splits remain sealed")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise PermissionError("test splits remain sealed")
    for record in artifacts.values():
        if not isinstance(record, dict):
            raise PermissionError("test splits remain sealed")
        path = Path(str(record.get("path", "")))
        expected = str(record.get("sha256", ""))
        if not path.is_file() or artifact_sha256(path) != expected:
            raise PermissionError("test splits remain sealed")
    return split_factory()


def _job_seed(method: str, split: SplitName, task_index: int, start_index: int) -> int:
    method_offset = 0 if method == "ADAM" else 100_000
    split_offset = 0 if split == "test_id" else 10_000
    return 2026092500 + method_offset + split_offset + 10 * task_index + start_index


def registered_baseline_jobs(
    *, id_task_ids: tuple[str, ...], ood_task_ids: tuple[str, ...]
) -> tuple[BaselineJob, ...]:
    """Return the immutable single-start and Adam-only multistart registry."""
    if len(id_task_ids) != 32 or len(ood_task_ids) != 16:
        raise ValueError("baseline registry requires exactly 32 ID and 16 OOD tasks")
    jobs: list[BaselineJob] = []
    for split, task_ids in (("test_id", id_task_ids), ("test_ood", ood_task_ids)):
        for method in ("ADAM", "MMA"):
            for task_index, task_id in enumerate(task_ids):
                seed = _job_seed(method, split, task_index, 0)
                jobs.append(
                    BaselineJob(
                        job_id=f"{split}_{task_index:02d}_{method.lower()}_start0",
                        method=method,
                        split=split,
                        task_index=task_index,
                        task_id=task_id,
                        start_index=0,
                        seed=seed,
                    )
                )
        for task_index, task_id in enumerate(task_ids[:8]):
            for start_index in (1, 2, 3):
                seed = _job_seed("ADAM", split, task_index, start_index)
                jobs.append(
                    BaselineJob(
                        job_id=(f"{split}_{task_index:02d}_adam_start{start_index}"),
                        method="ADAM",
                        split=split,
                        task_index=task_index,
                        task_id=task_id,
                        start_index=start_index,
                        seed=seed,
                    )
                )
    return tuple(jobs)


def _split_seed(split: SplitName) -> int:
    return 2026092401 if split == "test_id" else 2026092402


def median_gap_bootstrap(
    gaps: NDArray[np.float64], *, split: SplitName
) -> MedianGapBootstrap:
    """Bootstrap the full-layout median with the preregistered split seed."""
    values = np.asarray(gaps, dtype=np.float64)
    expected = 32 if split == "test_id" else 16
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"{split} bootstrap requires {expected} finite gaps")
    seed = _split_seed(split)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, expected, size=(10_000, expected))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, (2.5, 97.5))
    return MedianGapBootstrap(
        split=split,
        seed=seed,
        resamples=10_000,
        median=float(np.median(values)),
        lower_bound=float(lower),
        upper_bound=float(upper),
    )


def classify_mt3_test_split(
    *,
    gaps: NDArray[np.float64],
    invalid_count: int,
    exact_budget_count: int,
    equivalent_evaluation_speedup: float,
    split: SplitName,
    bootstrap: MedianGapBootstrap,
) -> MT3TestVerdict:
    """Apply the frozen distribution-level MT3 verdict without cherry-picking."""
    values = np.asarray(gaps, dtype=np.float64)
    expected = 32 if split == "test_id" else 16
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ValueError(f"{split} verdict requires {expected} finite gaps")
    if bootstrap.split != split or bootstrap.resamples != 10_000:
        raise ValueError("bootstrap does not match the requested split")
    if (
        not math.isfinite(equivalent_evaluation_speedup)
        or equivalent_evaluation_speedup <= 0
    ):
        raise ValueError("evaluation speedup must be finite and positive")
    median = float(np.median(values))
    p90 = float(np.quantile(values, 0.9))
    worst = float(np.max(values))
    wins = int(np.count_nonzero(values < 0.0))
    valid = invalid_count == 0 and exact_budget_count == expected
    suffix = "ID" if split == "test_id" else "OOD"
    if not valid:
        status = f"MT3_INVALID_RUN_{suffix}"
    elif median < 0.0 and bootstrap.upper_bound < 0.0 and wins / expected >= 0.60:
        status = f"MT3_BEATS_SINGLE_START_{suffix}"
    elif median <= 0.02 and p90 <= 0.07 and wins / expected >= 0.40:
        status = f"MT3_COMPETITIVE_{suffix}"
    elif 0.02 < median <= 0.05 and equivalent_evaluation_speedup >= 10.0:
        status = f"MT3_SPEED_ONLY_{suffix}"
    else:
        status = f"MT3_NO_GO_{suffix}"
    return MT3TestVerdict(
        status=status,
        split=split,
        task_count=expected,
        invalid_count=invalid_count,
        exact_budget_count=exact_budget_count,
        median_gap=median,
        p90_gap=p90,
        worst_gap=worst,
        win_count=wins,
        win_rate=wins / expected,
        equivalent_evaluation_speedup=equivalent_evaluation_speedup,
        bootstrap_lower=bootstrap.lower_bound,
        bootstrap_upper=bootstrap.upper_bound,
    )
