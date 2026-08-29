"""Independent SciPy verification of frozen pure-NCA designs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import artifact_sha256, content_hash
from waveforge.verification.high_fidelity import (
    CandidateVerification,
    array_sha256,
    replicate_design,
    verify_candidate,
)

PEAK_THRESHOLD_256 = 0.1721575074379424
BINARY_BUDGET_MINIMUM = 0.24
BINARY_BUDGET_MAXIMUM = 0.26
PASSING_SEEDS_REQUIRED = 2


class NCAIntegrityError(RuntimeError):
    """A final design differs from the registered production artifact."""


class NCASeedStatus(StrEnum):
    """Per-seed effect status after a technically valid production run."""

    PASS = "PASS"
    NO_GO_EFFECT = "NO_GO_EFFECT"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class NCAGridDiagnostic:
    """Secondary or primary independent grid result."""

    resolution: int
    grid_shape: tuple[int, int]
    worst_peak: float
    average_peak: float
    protected_zone_peak: float
    material_fraction: float
    transferred_design_hash: str
    total_wall_seconds: float


@dataclass(frozen=True)
class NCAConnectivityDiagnostic:
    """Four-neighbor topology diagnostic with no verdict authority."""

    conductive_cell_count: int
    component_count: int
    sink_connected_cell_count: int
    sink_connected_fraction: float
    source_intersection_cell_counts: dict[str, int]
    sink_component_source_intersections: dict[str, bool]


@dataclass(frozen=True)
class NCASeedVerdict:
    """Locked primary decision for one final strict-binary design."""

    seed: int
    status: NCASeedStatus
    peak_256: float
    binary_fraction: float
    peak_pass: bool
    budget_pass: bool
    reason_codes: tuple[str, ...]

    @classmethod
    def classify(
        cls,
        *,
        seed: int,
        peak_256: float,
        binary_fraction: float,
        production_valid: bool,
    ) -> NCASeedVerdict:
        """Apply inclusive unrounded threshold and budget boundaries."""
        if (
            not production_valid
            or not math.isfinite(peak_256)
            or peak_256 <= 0.0
            or not math.isfinite(binary_fraction)
        ):
            return cls(
                seed=seed,
                status=NCASeedStatus.INVALID_RUN,
                peak_256=peak_256,
                binary_fraction=binary_fraction,
                peak_pass=False,
                budget_pass=False,
                reason_codes=("PRODUCTION_OR_VERIFICATION_INVALID",),
            )
        peak_pass = peak_256 <= PEAK_THRESHOLD_256
        budget_pass = BINARY_BUDGET_MINIMUM <= binary_fraction <= BINARY_BUDGET_MAXIMUM
        reasons: list[str] = []
        if not peak_pass:
            reasons.append("PEAK_THRESHOLD_FAILURE")
        if not budget_pass:
            reasons.append("BINARY_MATERIAL_BUDGET_FAILURE")
        return cls(
            seed=seed,
            status=(NCASeedStatus.PASS if not reasons else NCASeedStatus.NO_GO_EFFECT),
            peak_256=peak_256,
            binary_fraction=binary_fraction,
            peak_pass=peak_pass,
            budget_pass=budget_pass,
            reason_codes=tuple(reasons),
        )


@dataclass(frozen=True)
class NCAComparatorRecord:
    """Fixed post-result diagnostic comparator."""

    comparator_id: str
    worst_peak_256: float
    source_artifact: str
    source_artifact_sha256: str
    nca_relative_difference: float


@dataclass(frozen=True)
class NCASeedVerification:
    """Complete dual-grid verification of one frozen production seed."""

    candidate_id: str
    verification_128: CandidateVerification
    verification_256: CandidateVerification
    diagnostic_128: NCAGridDiagnostic
    diagnostic_256: NCAGridDiagnostic
    relative_128_to_256_change: float
    connectivity: NCAConnectivityDiagnostic
    verdict: NCASeedVerdict


@dataclass(frozen=True)
class NCACampaignVerdict:
    """Campaign outcome with technical-invalidity precedence."""

    status: str
    passing_seed_count: int
    required_passing_seed_count: int
    production_valid: bool
    reproduction_valid: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class NCAReproductionComparison:
    """Topology/verdict replay comparison for warn-only CUDA mode."""

    valid: bool
    binary_exact: bool
    binary_fraction_exact: bool
    verdict_exact: bool
    maximum_continuous_difference: float
    mean_continuous_difference: float
    reason_codes: tuple[str, ...]


def _grid_diagnostic(result: CandidateVerification) -> NCAGridDiagnostic:
    return NCAGridDiagnostic(
        resolution=result.grid_shape[0],
        grid_shape=result.grid_shape,
        worst_peak=result.worst_peak,
        average_peak=result.average_peak,
        protected_zone_peak=result.protected_zone_peak,
        material_fraction=result.material_fraction,
        transferred_design_hash=result.transferred_design_hash,
        total_wall_seconds=result.total_wall_seconds,
    )


def _connectivity(binary: NDArray[np.float64]) -> NCAConnectivityDiagnostic:
    conductive = binary == 1.0
    visited = np.zeros_like(conductive, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    height, width = conductive.shape
    for row in range(height):
        for column in range(width):
            if not conductive[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            component: list[tuple[int, int]] = []
            while stack:
                current_row, current_column = stack.pop()
                component.append((current_row, current_column))
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if (
                        0 <= next_row < height
                        and 0 <= next_column < width
                        and conductive[next_row, next_column]
                        and not visited[next_row, next_column]
                    ):
                        visited[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(component)

    sink_cells = {
        cell
        for component in components
        if any(row == 0 for row, _ in component)
        for cell in component
    }
    grid = Grid2D(nx=64, ny=64)
    scenario_bounds = {
        "A": (0.40, 0.60, 0.62, 0.82),
        "B": (0.18, 0.38, 0.62, 0.82),
        "C": (0.62, 0.82, 0.62, 0.82),
    }
    source_cells = {
        scenario_id: set(
            map(
                tuple,
                np.argwhere(area_overlap_rectangular_source(grid, bounds, 1.0) > 0.0),
            )
        )
        for scenario_id, bounds in scenario_bounds.items()
    }
    conductive_cells = set(map(tuple, np.argwhere(conductive)))
    conductive_count = len(conductive_cells)
    return NCAConnectivityDiagnostic(
        conductive_cell_count=conductive_count,
        component_count=len(components),
        sink_connected_cell_count=len(sink_cells),
        sink_connected_fraction=(
            len(sink_cells) / conductive_count if conductive_count else 0.0
        ),
        source_intersection_cell_counts={
            scenario_id: len(cells & conductive_cells)
            for scenario_id, cells in source_cells.items()
        },
        sink_component_source_intersections={
            scenario_id: bool(cells & sink_cells)
            for scenario_id, cells in source_cells.items()
        },
    )


def _parse_seed(candidate_id: str) -> int:
    suffix = candidate_id.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def verify_nca_seed(
    candidate_id: str,
    binary_design: NDArray[np.float64],
    *,
    continuous_design: NDArray[np.float64],
    expected_binary_content_hash: str | None = None,
    expected_continuous_content_hash: str | None = None,
    production_valid: bool = True,
) -> NCASeedVerification:
    """Verify a frozen strict-binary NCA design on independent SciPy grids."""
    raw_binary = np.asarray(binary_design)
    raw_continuous = np.asarray(continuous_design)
    if (
        expected_binary_content_hash is not None
        and content_hash(raw_binary) != expected_binary_content_hash
    ):
        raise NCAIntegrityError("binary production content hash mismatch")
    if (
        expected_continuous_content_hash is not None
        and content_hash(raw_continuous) != expected_continuous_content_hash
    ):
        raise NCAIntegrityError("continuous production content hash mismatch")
    binary = np.asarray(raw_binary, dtype=np.float64)
    continuous = np.asarray(raw_continuous, dtype=np.float64)
    if binary.shape != (64, 64) or continuous.shape != (64, 64):
        raise NCAIntegrityError("binary and continuous designs must be 64x64")
    if not np.isfinite(binary).all() or not np.isfinite(continuous).all():
        raise NCAIntegrityError("binary and continuous designs must be finite")
    if not np.all((binary == 0.0) | (binary == 1.0)):
        raise NCAIntegrityError("binary design is not strict 0/1")
    if not np.array_equal(binary, (continuous >= 0.5).astype(np.float64)):
        raise NCAIntegrityError("threshold design differs from strict D >= 0.5")
    frozen_hash = array_sha256(binary)
    verification_128 = verify_candidate(
        candidate_id,
        binary,
        fidelity="reference_128",
        expected_design_hash=frozen_hash,
    )
    verification_256 = verify_candidate(
        candidate_id,
        binary,
        fidelity="reference_256",
        expected_design_hash=frozen_hash,
    )
    expected_128 = array_sha256(replicate_design(binary, factor=2))
    expected_256 = array_sha256(replicate_design(binary, factor=4))
    if (
        verification_128.transferred_design_hash != expected_128
        or verification_256.transferred_design_hash != expected_256
    ):
        raise NCAIntegrityError("verification transfer is not exact replication")
    denominator = max(abs(verification_256.worst_peak), 1.0e-12)
    relative_change = (
        verification_128.worst_peak - verification_256.worst_peak
    ) / denominator
    verdict = NCASeedVerdict.classify(
        seed=_parse_seed(candidate_id),
        peak_256=verification_256.worst_peak,
        binary_fraction=float(binary.mean()),
        production_valid=production_valid,
    )
    return NCASeedVerification(
        candidate_id=candidate_id,
        verification_128=verification_128,
        verification_256=verification_256,
        diagnostic_128=_grid_diagnostic(verification_128),
        diagnostic_256=_grid_diagnostic(verification_256),
        relative_128_to_256_change=relative_change,
        connectivity=_connectivity(binary),
        verdict=verdict,
    )


def classify_nca_campaign(
    seed_verdicts: list[NCASeedVerdict] | tuple[NCASeedVerdict, ...],
    *,
    production_valid: bool = True,
    reproduction_valid: bool = True,
) -> NCACampaignVerdict:
    """Apply locked campaign rules with invalidity precedence."""
    passing = sum(verdict.status is NCASeedStatus.PASS for verdict in seed_verdicts)
    if not production_valid or any(
        verdict.status is NCASeedStatus.INVALID_RUN for verdict in seed_verdicts
    ):
        return NCACampaignVerdict(
            status="NCA_SPIKE_INVALID_PRODUCTION_RUN",
            passing_seed_count=passing,
            required_passing_seed_count=PASSING_SEEDS_REQUIRED,
            production_valid=False,
            reproduction_valid=reproduction_valid,
            reason_codes=("PRODUCTION_OR_VERIFICATION_INVALID",),
        )
    if not reproduction_valid:
        return NCACampaignVerdict(
            status="NCA_SPIKE_INVALID_REPRODUCIBILITY",
            passing_seed_count=passing,
            required_passing_seed_count=PASSING_SEEDS_REQUIRED,
            production_valid=True,
            reproduction_valid=False,
            reason_codes=("TOPOLOGY_OR_VERDICT_REPLAY_MISMATCH",),
        )
    if passing >= PASSING_SEEDS_REQUIRED:
        return NCACampaignVerdict(
            status="NCA_FEASIBILITY_GO",
            passing_seed_count=passing,
            required_passing_seed_count=PASSING_SEEDS_REQUIRED,
            production_valid=True,
            reproduction_valid=True,
            reason_codes=(),
        )
    return NCACampaignVerdict(
        status="NCA_NO_GO_EFFECT",
        passing_seed_count=passing,
        required_passing_seed_count=PASSING_SEEDS_REQUIRED,
        production_valid=True,
        reproduction_valid=True,
        reason_codes=("FEWER_THAN_TWO_SEEDS_PASS",),
    )


def compare_reproduction(
    original_continuous: NDArray[np.float64],
    original_binary: NDArray[np.float64],
    original_verdict: str,
    reproduced_continuous: NDArray[np.float64],
    reproduced_binary: NDArray[np.float64],
    reproduced_verdict: str,
) -> NCAReproductionComparison:
    """Accept drift only when strict topology, fraction and verdict are exact."""
    original_continuous = np.asarray(original_continuous, dtype=np.float64)
    reproduced_continuous = np.asarray(reproduced_continuous, dtype=np.float64)
    original_binary = np.asarray(original_binary, dtype=np.float64)
    reproduced_binary = np.asarray(reproduced_binary, dtype=np.float64)
    if original_continuous.shape != reproduced_continuous.shape:
        raise ValueError("continuous reproduction shapes differ")
    differences = np.abs(original_continuous - reproduced_continuous)
    binary_exact = np.array_equal(original_binary, reproduced_binary)
    fraction_exact = float(original_binary.mean()) == float(reproduced_binary.mean())
    verdict_exact = original_verdict == reproduced_verdict
    reasons: list[str] = []
    if not binary_exact:
        reasons.append("BINARY_TOPOLOGY_MISMATCH")
    if not fraction_exact:
        reasons.append("BINARY_FRACTION_MISMATCH")
    if not verdict_exact:
        reasons.append("SCIPY_256_VERDICT_MISMATCH")
    return NCAReproductionComparison(
        valid=not reasons,
        binary_exact=binary_exact,
        binary_fraction_exact=fraction_exact,
        verdict_exact=verdict_exact,
        maximum_continuous_difference=float(np.max(differences)),
        mean_continuous_difference=float(np.mean(differences)),
        reason_codes=tuple(reasons),
    )


def fixed_comparator_records(
    nca_peak_256: float,
    *,
    project_root: Path,
) -> tuple[NCAComparatorRecord, ...]:
    """Load preregistered diagnostic comparators and their source hashes."""
    registry = (
        (
            "waveforge_20260828",
            0.156506824943584,
            project_root / "artifacts/gate2_design/optimized_metrics.csv",
        ),
        (
            "waveforge_20260829",
            0.1574716324313547,
            project_root / "artifacts/gate2_design/optimized_metrics.csv",
        ),
        (
            "waveforge_20260830",
            0.15663546358885735,
            project_root / "artifacts/gate2_design/optimized_metrics.csv",
        ),
        (
            "parametric_branching_tree",
            0.1650978093408512,
            project_root / "artifacts/gate2a_challenge/tree_finalists_256.csv",
        ),
        (
            "straight_path",
            0.3169417981503212,
            project_root / "artifacts/gate2_design/baseline_metrics.csv",
        ),
    )
    records: list[NCAComparatorRecord] = []
    for comparator_id, comparator_peak, source in registry:
        if not source.is_file():
            raise NCAIntegrityError(f"comparator source artifact missing: {source}")
        records.append(
            NCAComparatorRecord(
                comparator_id=comparator_id,
                worst_peak_256=comparator_peak,
                source_artifact=source.relative_to(project_root).as_posix(),
                source_artifact_sha256=artifact_sha256(source),
                nca_relative_difference=(comparator_peak - nca_peak_256)
                / comparator_peak,
            )
        )
    return tuple(records)


def dataclass_payload(value: Any) -> Any:
    """Recursively expose dataclass/enum values for machine-readable output."""
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: dataclass_payload(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [dataclass_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: dataclass_payload(item) for key, item in value.items()}
    return value
