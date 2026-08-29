"""Independent SciPy evaluation and verdict rules for the tree challenge."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import (
    assemble_steady_system,
    factorize_system,
    solve_factorized,
)

SOURCE_BOUNDS = (
    (0.40, 0.60, 0.62, 0.82),
    (0.18, 0.38, 0.62, 0.82),
    (0.62, 0.82, 0.62, 0.82),
)


class ChallengeStatus(StrEnum):
    """Machine-readable outcomes for the prospective challenge."""

    STRONG_CHALLENGE_PASS = "STRONG_CHALLENGE_PASS"
    CHALLENGE_COMPARABLE = "CHALLENGE_COMPARABLE"
    CHALLENGE_FAIL = "CHALLENGE_FAIL"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class ChallengeEvaluation:
    """One frozen binary design evaluated for all three nominal sources."""

    candidate_id: str
    resolution: int
    design_hash_64: str
    transferred_design_hash: str
    material_fraction: float
    scenario_peaks: tuple[float, float, float]
    scenario_residuals: tuple[float, float, float]
    worst_peak: float
    average_peak: float
    maximum_residual: float
    wall_seconds: float
    temperature_fields: tuple[NDArray[np.float64], ...] | None = None


@dataclass(frozen=True)
class ChallengeSeedComparison:
    """Nominal and robustness effect for one frozen WaveForge seed."""

    seed: int
    nominal_improvement: float
    robustness_passing_cases: int

    @property
    def nominal_pass_5pct(self) -> bool:
        """Return whether the unrounded nominal effect reaches five percent."""
        return self.nominal_improvement >= 0.05

    @property
    def seed_strong_pass(self) -> bool:
        """Require nominal and robustness criteria for the same seed."""
        return self.nominal_pass_5pct and self.robustness_passing_cases >= 23


@dataclass(frozen=True)
class ChallengeVerdict:
    """Campaign-level status with explicit evidence and reason codes."""

    status: ChallengeStatus
    reason_codes: tuple[str, ...]
    metrics: dict[str, object]


def _array_sha256(array: NDArray[np.float64]) -> str:
    values = np.asarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _transfer_design(design_64: NDArray[np.float64], resolution: int) -> NDArray:
    if resolution not in (64, 128, 256):
        raise ValueError("challenge resolution must be 64, 128 or 256")
    factor = resolution // 64
    return np.repeat(np.repeat(design_64, factor, axis=0), factor, axis=1)


def evaluate_frozen_binary_design(
    candidate_id: str,
    frozen_design_64: NDArray[np.float64],
    *,
    resolution: int,
    include_temperature_fields: bool = False,
) -> ChallengeEvaluation:
    """Evaluate three source RHS with one independent SciPy factorization."""
    design_64 = np.asarray(frozen_design_64, dtype=np.float64)
    if design_64.shape != (64, 64):
        raise ValueError("challenge design must have shape 64x64")
    if not np.all((design_64 == 0.0) | (design_64 == 1.0)):
        raise ValueError("challenge design must be strict binary")
    if int(np.sum(design_64)) != 1024:
        raise ValueError("challenge design must contain exactly 1024 high cells")

    transferred = _transfer_design(design_64, resolution)
    grid = Grid2D(nx=resolution, ny=resolution)
    sources = tuple(
        area_overlap_rectangular_source(grid, bounds, 1.0) for bounds in SOURCE_BOUNDS
    )
    conductivity = 1.0 + 19.0 * transferred**3
    started = time.perf_counter()
    first_system = assemble_steady_system(
        grid,
        conductivity,
        sources[0],
        BoundaryConditions.production(),
    )
    factorization = factorize_system(first_system)
    temperatures: list[NDArray[np.float64]] = []
    residuals: list[float] = []
    for source in sources:
        source_rhs = source.ravel().copy()
        system = replace(
            first_system,
            source_rhs=source_rhs,
            rhs=source_rhs + first_system.dirichlet_rhs,
        )
        result = solve_factorized(system, factorization)
        temperatures.append(result.temperature.copy())
        residuals.append(result.normalized_residual)
    elapsed = time.perf_counter() - started

    peaks = tuple(float(np.max(field)) for field in temperatures)
    residual_tuple = tuple(float(value) for value in residuals)
    if (
        len(peaks) != 3
        or not all(math.isfinite(value) and value > 0.0 for value in peaks)
        or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0e-10 for value in residual_tuple
        )
    ):
        raise FloatingPointError("challenge SciPy evaluation failed integrity checks")
    return ChallengeEvaluation(
        candidate_id=candidate_id,
        resolution=resolution,
        design_hash_64=_array_sha256(design_64),
        transferred_design_hash=_array_sha256(transferred),
        material_fraction=float(np.mean(transferred)),
        scenario_peaks=peaks,  # type: ignore[arg-type]
        scenario_residuals=residual_tuple,  # type: ignore[arg-type]
        worst_peak=max(peaks),
        average_peak=float(np.mean(peaks)),
        maximum_residual=max(residual_tuple),
        wall_seconds=elapsed,
        temperature_fields=(
            tuple(temperatures) if include_temperature_fields else None
        ),
    )


def classify_challenge(
    comparisons: tuple[ChallengeSeedComparison, ...],
    *,
    valid: bool,
) -> ChallengeVerdict:
    """Apply locked invalid/fail/strong/comparable precedence literally."""
    numerical_valid = (
        valid
        and len(comparisons) == 3
        and len({comparison.seed for comparison in comparisons}) == 3
        and all(
            math.isfinite(comparison.nominal_improvement)
            and isinstance(comparison.robustness_passing_cases, int)
            and not isinstance(comparison.robustness_passing_cases, bool)
            and 0 <= comparison.robustness_passing_cases <= 28
            for comparison in comparisons
        )
    )
    if not numerical_valid:
        return ChallengeVerdict(
            status=ChallengeStatus.INVALID_RUN,
            reason_codes=("NUMERICAL_OR_REGISTRY_FAILURE",),
            metrics={"comparison_count": len(comparisons)},
        )

    negative_count = sum(
        comparison.nominal_improvement < 0.0 for comparison in comparisons
    )
    strong_count = sum(comparison.seed_strong_pass for comparison in comparisons)
    metrics: dict[str, object] = {
        "negative_seed_count": negative_count,
        "strong_seed_count": strong_count,
        "required_strong_seeds": 2,
        "seed_metrics": {
            str(comparison.seed): {
                "nominal_improvement": comparison.nominal_improvement,
                "nominal_pass_5pct": comparison.nominal_pass_5pct,
                "robustness_passing_cases": comparison.robustness_passing_cases,
                "seed_strong_pass": comparison.seed_strong_pass,
            }
            for comparison in comparisons
        },
    }
    if negative_count >= 2:
        return ChallengeVerdict(
            status=ChallengeStatus.CHALLENGE_FAIL,
            reason_codes=("TREE_BETTER_FOR_AT_LEAST_TWO_SEEDS",),
            metrics=metrics,
        )
    if strong_count >= 2:
        return ChallengeVerdict(
            status=ChallengeStatus.STRONG_CHALLENGE_PASS,
            reason_codes=(),
            metrics=metrics,
        )
    return ChallengeVerdict(
        status=ChallengeStatus.CHALLENGE_COMPARABLE,
        reason_codes=("STRONG_ADVANTAGE_NOT_ESTABLISHED",),
        metrics=metrics,
    )
