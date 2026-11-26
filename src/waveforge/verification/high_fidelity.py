"""Frozen-map independent SciPy verification for Gate 2A."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady

Fidelity = Literal["low_64", "reference_128", "reference_256"]


class VerificationIntegrityError(RuntimeError):
    """Raised when the provided map is not the frozen candidate."""


@dataclass(frozen=True)
class VerificationRecord:
    """One independently rasterized and solved heat-source scenario."""

    candidate_id: str
    fidelity: Fidelity
    scenario_id: str
    peak_temperature: float
    protected_zone_peak: float
    normalized_residual: float
    wall_seconds: float
    source_hash: str
    integrated_power: float


@dataclass(frozen=True)
class CandidateVerification:
    """Aggregate exact metrics for one frozen candidate at one fidelity."""

    candidate_id: str
    fidelity: Fidelity
    grid_shape: tuple[int, int]
    design_hash_64: str
    transferred_design_hash: str
    is_binary: bool
    material_fraction: float
    total_variation: float
    worst_peak: float
    average_peak: float
    protected_zone_peak: float
    total_wall_seconds: float
    scenario_records: tuple[VerificationRecord, ...]
    claimed_worst_peak: float | None
    claim_matches: bool | None


def array_sha256(array: NDArray[np.float64]) -> str:
    """Hash dtype, shape, and stable little-endian row-major bytes."""
    values = np.asarray(array, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(json.dumps(list(values.shape)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def replicate_design(
    design: NDArray[np.float64],
    *,
    factor: int,
) -> NDArray[np.float64]:
    """Transfer without interpolation by exact parent-cell replication."""
    values = np.asarray(design, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("design must be a finite two-dimensional array")
    if factor < 1 or isinstance(factor, bool):
        raise ValueError("replication factor must be a positive integer")
    return np.repeat(np.repeat(values, factor, axis=0), factor, axis=1)


def _total_variation(design: NDArray[np.float64]) -> float:
    horizontal = np.mean(np.abs(design[:, 1:] - design[:, :-1]))
    vertical = np.mean(np.abs(design[1:, :] - design[:-1, :]))
    return float(horizontal + vertical)


def _protected_zone_peak(
    temperature: NDArray[np.float64],
    grid: Grid2D,
) -> float:
    x_mask = (grid.x_centers >= 0.40) & (grid.x_centers <= 0.60)
    y_mask = (grid.y_centers >= 0.85) & (grid.y_centers <= 1.00)
    return float(np.max(temperature[np.ix_(y_mask, x_mask)]))


def verify_candidate(
    candidate_id: str,
    frozen_design_64: NDArray[np.float64],
    *,
    fidelity: Fidelity,
    expected_design_hash: str | None = None,
    claimed_worst_peak: float | None = None,
    k_low: float = 1.0,
    k_high: float = 20.0,
    interpolation_power: float = 3.0,
) -> CandidateVerification:
    """Independently verify one unchanged design under all three scenarios."""
    design_64 = np.asarray(frozen_design_64, dtype=np.float64)
    if design_64.shape != (64, 64) or not np.all(np.isfinite(design_64)):
        raise ValueError("frozen design must be a finite 64x64 array")
    if np.any((design_64 < 0.0) | (design_64 > 1.0)):
        raise ValueError("frozen design values must lie in [0,1]")
    design_hash = array_sha256(design_64)
    if expected_design_hash is not None and design_hash != expected_design_hash:
        raise VerificationIntegrityError("frozen design hash does not match")

    fidelity_parameters: dict[Fidelity, tuple[int, int]] = {
        "low_64": (1, 64),
        "reference_128": (2, 128),
        "reference_256": (4, 256),
    }
    factor, resolution = fidelity_parameters[fidelity]
    transferred = replicate_design(design_64, factor=factor)
    grid = Grid2D(nx=resolution, ny=resolution)
    conductivity = k_low + (k_high - k_low) * transferred**interpolation_power
    scenario_bounds = (
        ("A", (0.40, 0.60, 0.62, 0.82)),
        ("B", (0.18, 0.38, 0.62, 0.82)),
        ("C", (0.62, 0.82, 0.62, 0.82)),
    )
    records: list[VerificationRecord] = []
    for scenario_id, bounds in scenario_bounds:
        source = area_overlap_rectangular_source(grid, bounds, 1.0)
        started = time.perf_counter()
        result = solve_steady(
            grid,
            conductivity,
            source,
            BoundaryConditions.production(),
        )
        elapsed = time.perf_counter() - started
        integrated_power = float(np.sum(source) * grid.dx * grid.dy)
        records.append(
            VerificationRecord(
                candidate_id=candidate_id,
                fidelity=fidelity,
                scenario_id=scenario_id,
                peak_temperature=float(np.max(result.temperature)),
                protected_zone_peak=_protected_zone_peak(result.temperature, grid),
                normalized_residual=result.normalized_residual,
                wall_seconds=elapsed,
                source_hash=array_sha256(source),
                integrated_power=integrated_power,
            )
        )
    peaks = np.array([record.peak_temperature for record in records])
    protected_peaks = np.array([record.protected_zone_peak for record in records])
    worst_peak = float(np.max(peaks))
    claim_matches = (
        None
        if claimed_worst_peak is None
        else bool(np.isclose(claimed_worst_peak, worst_peak, rtol=1.0e-6, atol=1.0e-10))
    )
    return CandidateVerification(
        candidate_id=candidate_id,
        fidelity=fidelity,
        grid_shape=grid.shape,
        design_hash_64=design_hash,
        transferred_design_hash=array_sha256(transferred),
        is_binary=bool(np.all((design_64 == 0.0) | (design_64 == 1.0))),
        material_fraction=float(np.mean(transferred)),
        total_variation=_total_variation(transferred),
        worst_peak=worst_peak,
        average_peak=float(np.mean(peaks)),
        protected_zone_peak=float(np.max(protected_peaks)),
        total_wall_seconds=float(sum(record.wall_seconds for record in records)),
        scenario_records=tuple(records),
        claimed_worst_peak=claimed_worst_peak,
        claim_matches=claim_matches,
    )


def relative_improvement(baseline_peak: float, candidate_peak: float) -> float:
    """Compute the unrounded registered baseline-relative improvement."""
    if not np.isfinite(baseline_peak) or baseline_peak <= 0.0:
        raise ValueError("baseline peak must be finite and positive")
    if not np.isfinite(candidate_peak):
        raise ValueError("candidate peak must be finite")
    return (baseline_peak - candidate_peak) / baseline_peak
