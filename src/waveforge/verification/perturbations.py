"""Literal robustness registry and separate morphology diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_erosion, label

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.verification.compare import Gate2Status, SeedVerdict
from waveforge.verification.high_fidelity import (
    array_sha256,
    replicate_design,
    verify_candidate,
)

PerturbationKind = Literal[
    "source_shift",
    "intensity_scale",
    "conductivity_scale",
]
MorphologyOperation = Literal["erosion", "dilation"]


@dataclass(frozen=True)
class PerturbationCase:
    """One pre-registered non-morphological verification case."""

    case_id: str
    kind: PerturbationKind
    scenario_id: str | None = None
    shift_cells_x: int = 0
    shift_cells_y: int = 0
    intensity_scale: float = 1.0
    k_high: float = 20.0


@dataclass(frozen=True)
class PerturbationEvaluation:
    """Independent `256×256` result for one candidate and one case."""

    candidate_id: str
    case_id: str
    grid_shape: tuple[int, int]
    design_hash_64: str
    material_fraction: float
    scenario_peaks: tuple[float, ...]
    scenario_residuals: tuple[float, ...]
    worst_peak: float
    k_high: float
    wall_seconds: float


@dataclass(frozen=True)
class MorphologyRecord:
    """One separately reported, budget-changing morphology diagnostic."""

    candidate_id: str
    operation: Literal["unperturbed", "erosion", "dilation"]
    material_fraction: float
    worst_peak_256: float
    component_count: int
    relative_degradation: float
    design_hash_64: str


def registered_primary_cases() -> tuple[PerturbationCase, ...]:
    """Return exactly the locked 28-case registry in stable order."""
    cases: list[PerturbationCase] = []
    deltas = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, 1),
        "down": (0, -1),
    }
    for scenario_id in ("A", "B", "C"):
        for distance in (1, 2):
            for direction, (unit_x, unit_y) in deltas.items():
                cases.append(
                    PerturbationCase(
                        case_id=f"shift_{scenario_id}_{distance}_{direction}",
                        kind="source_shift",
                        scenario_id=scenario_id,
                        shift_cells_x=distance * unit_x,
                        shift_cells_y=distance * unit_y,
                    )
                )
    cases.extend(
        (
            PerturbationCase(
                case_id="intensity_minus_5pct",
                kind="intensity_scale",
                intensity_scale=0.95,
            ),
            PerturbationCase(
                case_id="intensity_plus_5pct",
                kind="intensity_scale",
                intensity_scale=1.05,
            ),
            PerturbationCase(
                case_id="k_high_19",
                kind="conductivity_scale",
                k_high=19.0,
            ),
            PerturbationCase(
                case_id="k_high_21",
                kind="conductivity_scale",
                k_high=21.0,
            ),
        )
    )
    return tuple(cases)


def perturbed_source_batch(
    grid: Grid2D,
    *,
    case: PerturbationCase | None,
) -> tuple[NDArray[np.float64], float]:
    """Rasterize identical per-case source inputs for every candidate."""
    if case is not None and grid.shape != (256, 256):
        raise ValueError("registered perturbations require the 256x256 grid")
    nominal_bounds = {
        "A": (0.40, 0.60, 0.62, 0.82),
        "B": (0.18, 0.38, 0.62, 0.82),
        "C": (0.62, 0.82, 0.62, 0.82),
    }
    source_maps: list[NDArray[np.float64]] = []
    for scenario_id in ("A", "B", "C"):
        x_min, x_max, y_min, y_max = nominal_bounds[scenario_id]
        if (
            case is not None
            and case.kind == "source_shift"
            and case.scenario_id == scenario_id
        ):
            delta_x = case.shift_cells_x / 256.0
            delta_y = case.shift_cells_y / 256.0
            x_min += delta_x
            x_max += delta_x
            y_min += delta_y
            y_max += delta_y
        power = (
            case.intensity_scale
            if case is not None and case.kind == "intensity_scale"
            else 1.0
        )
        source_maps.append(
            area_overlap_rectangular_source(
                grid,
                (x_min, x_max, y_min, y_max),
                power,
            )
        )
    k_high = case.k_high if case is not None else 20.0
    return np.stack(source_maps), k_high


def apply_morphology(
    design: NDArray[np.float64],
    *,
    operation: MorphologyOperation,
) -> NDArray[np.float64]:
    """Apply one 3×3 iteration with low material outside the domain."""
    values = np.asarray(design, dtype=np.float64)
    if values.ndim != 2 or not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("morphology requires a strict binary design")
    structure = np.ones((3, 3), dtype=bool)
    boolean = values.astype(bool)
    if operation == "erosion":
        result = binary_erosion(
            boolean,
            structure=structure,
            iterations=1,
            border_value=0,
        )
    elif operation == "dilation":
        result = binary_dilation(
            boolean,
            structure=structure,
            iterations=1,
            border_value=0,
        )
    else:
        raise ValueError(f"unsupported morphology operation: {operation}")
    return result.astype(np.float64)


def four_neighbor_component_count(design: NDArray[np.float64]) -> int:
    """Count conductive components using cross-shaped connectivity."""
    values = np.asarray(design, dtype=np.float64)
    if values.ndim != 2 or not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("component counting requires a strict binary design")
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
    _, count = label(values.astype(bool), structure=structure)
    return int(count)


def classify_seed_robustness(
    improvements: list[float],
    *,
    valid: bool,
) -> SeedVerdict:
    """Apply the locked `23/28` robustness rule with invalidity precedence."""
    if not valid or len(improvements) != 28 or not np.all(np.isfinite(improvements)):
        return SeedVerdict(
            status=Gate2Status.INVALID_RUN,
            reason_codes=("ROBUSTNESS_NUMERICAL_OR_REGISTRY_FAILURE",),
            metrics={"case_count": len(improvements)},
        )
    passing_count = sum(value >= 0.02 for value in improvements)
    if passing_count < 23:
        return SeedVerdict(
            status=Gate2Status.NO_GO_EFFECT,
            reason_codes=("ROBUSTNESS_EFFECT_FAILURE",),
            metrics={"passing_cases": passing_count, "required": 23},
        )
    return SeedVerdict(
        status=Gate2Status.PASS,
        metrics={"passing_cases": passing_count, "required": 23},
    )


def evaluate_perturbation_case(
    candidate_id: str,
    frozen_design_64: NDArray[np.float64],
    case: PerturbationCase,
) -> PerturbationEvaluation:
    """Evaluate one unchanged candidate under one registered `256×256` case."""
    design_64 = np.asarray(frozen_design_64, dtype=np.float64)
    if design_64.shape != (64, 64):
        raise ValueError("perturbation candidate must be 64x64")
    transferred = replicate_design(design_64, factor=4)
    grid = Grid2D(nx=256, ny=256)
    sources, k_high = perturbed_source_batch(grid, case=case)
    conductivity = 1.0 + (k_high - 1.0) * transferred**3
    peaks: list[float] = []
    residuals: list[float] = []
    started = time.perf_counter()
    for source in sources:
        result = solve_steady(
            grid,
            conductivity,
            source,
            BoundaryConditions.production(),
        )
        peaks.append(float(np.max(result.temperature)))
        residuals.append(result.normalized_residual)
    return PerturbationEvaluation(
        candidate_id=candidate_id,
        case_id=case.case_id,
        grid_shape=grid.shape,
        design_hash_64=array_sha256(design_64),
        material_fraction=float(np.mean(transferred)),
        scenario_peaks=tuple(peaks),
        scenario_residuals=tuple(residuals),
        worst_peak=max(peaks),
        k_high=k_high,
        wall_seconds=time.perf_counter() - started,
    )


def morphology_diagnostics(
    candidate_id: str,
    frozen_binary_design_64: NDArray[np.float64],
) -> tuple[MorphologyRecord, ...]:
    """Report unperturbed/eroded/dilated metrics without budget repair."""
    base_design = np.asarray(frozen_binary_design_64, dtype=np.float64)
    designs = (
        ("unperturbed", base_design),
        ("erosion", apply_morphology(base_design, operation="erosion")),
        ("dilation", apply_morphology(base_design, operation="dilation")),
    )
    verified = [
        verify_candidate(
            f"{candidate_id}_{operation}",
            design,
            fidelity="reference_256",
            expected_design_hash=array_sha256(design),
        )
        for operation, design in designs
    ]
    base_peak = verified[0].worst_peak
    return tuple(
        MorphologyRecord(
            candidate_id=candidate_id,
            operation=operation,  # type: ignore[arg-type]
            material_fraction=result.material_fraction,
            worst_peak_256=result.worst_peak,
            component_count=four_neighbor_component_count(design),
            relative_degradation=(result.worst_peak - base_peak) / base_peak,
            design_hash_64=array_sha256(design),
        )
        for (operation, design), result in zip(designs, verified, strict=True)
    )
