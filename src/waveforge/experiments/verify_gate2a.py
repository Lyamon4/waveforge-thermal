"""Gate 2A frozen-candidate registry and independent verification runner."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray

from waveforge.design.baselines import (
    dispersed_baseline,
    random_filtered_baseline,
    straight_path_baseline,
    uniform_relaxed_baseline,
)
from waveforge.design.optimize import array_sha256 as optimization_array_sha256
from waveforge.physics.grid import Grid2D
from waveforge.verification.compare import (
    Gate2Status,
    SeedVerdict,
    classify_nominal_seed,
    select_strongest_baseline,
)
from waveforge.verification.high_fidelity import (
    CandidateVerification,
    Fidelity,
    array_sha256,
    relative_improvement,
    verify_candidate,
)
from waveforge.verification.perturbations import (
    MorphologyRecord,
    PerturbationCase,
    PerturbationEvaluation,
    classify_seed_robustness,
    evaluate_perturbation_case,
    morphology_diagnostics,
    registered_primary_cases,
)

Representation = Literal["binary", "continuous"]
Category = Literal["random", "straight", "dispersed", "uniform", "single_A", "robust"]
PRODUCTION_SEEDS = (20260828, 20260829, 20260830)
RANDOM_BASELINE_SEEDS = (9101, 9102, 9103)


class CandidateIntegrityError(RuntimeError):
    """Raised when a mandatory frozen candidate cannot be trusted."""


@dataclass(frozen=True)
class FrozenCandidate:
    """One immutable design entering independent SciPy verification."""

    candidate_id: str
    category: Category
    representation: Representation
    seed: int | None
    design: NDArray[np.float64]
    design_hash: str
    source_runtime_seconds: float


@dataclass(frozen=True)
class ProductionCandidate:
    """Continuous and strict-binary maps from one completed optimizer run."""

    binary: FrozenCandidate
    continuous: FrozenCandidate


@dataclass(frozen=True)
class CandidateRegistry:
    """Complete locked binary and secondary continuous candidate sets."""

    binary: tuple[FrozenCandidate, ...]
    continuous: tuple[FrozenCandidate, ...]
    config_hash: str
    protocol_tag: str


@dataclass(frozen=True)
class NominalVerificationBundle:
    """Mandatory nominal verification records and per-seed decisions."""

    binary: tuple[CandidateVerification, ...]
    continuous: tuple[CandidateVerification, ...]
    seed_verdicts: dict[int, SeedVerdict]
    valid: bool


@dataclass(frozen=True)
class RobustnessComparison:
    """One robust-seed effect comparison under one registered case."""

    seed: int
    case_id: str
    candidate_id: str
    candidate_peak: float
    strongest_baseline_id: str
    strongest_baseline_peak: float
    relative_improvement: float
    passed_two_percent: bool


@dataclass(frozen=True)
class RobustnessVerificationBundle:
    """All registered perturbation solves and derived seed verdicts."""

    evaluations: tuple[PerturbationEvaluation, ...]
    comparisons: tuple[RobustnessComparison, ...]
    seed_verdicts: dict[int, SeedVerdict]
    valid: bool


@dataclass(frozen=True)
class MorphologyVerificationBundle:
    """Separate budget-changing diagnostics for every frozen binary design."""

    records: tuple[MorphologyRecord, ...]
    valid: bool


def _frozen_array(values: NDArray[np.float64]) -> NDArray[np.float64]:
    design = np.asarray(values, dtype=np.float64).copy()
    design.setflags(write=False)
    return design


def _optimization_runtime(metrics_path: Path) -> float:
    try:
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as error:
        raise CandidateIntegrityError("optimization metrics are unreadable") from error
    if len(rows) != 600:
        raise CandidateIntegrityError("optimization metrics must contain 600 rows")
    try:
        values = [float(row["wall_seconds"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise CandidateIntegrityError(
            "optimization runtime column is invalid"
        ) from error
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise CandidateIntegrityError("optimization runtimes must be finite")
    return float(sum(values))


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateIntegrityError(f"unreadable artifact: {path.name}") from error
    if not isinstance(payload, dict):
        raise CandidateIntegrityError(f"artifact is not a JSON object: {path.name}")
    return payload


def load_production_candidate(
    run_dir: Path,
    *,
    candidate_id: str,
    category: Literal["single_A", "robust"],
    seed: int,
    expected_config_hash: str,
    expected_protocol_tag: str,
) -> ProductionCandidate:
    """Load one run and validate every immutable identity needed downstream."""
    result = _load_json(run_dir / "optimization_result.json")
    required_identity = (
        result.get("status") == "PASS"
        and result.get("completed_iterations") == 600
        and result.get("seed") == seed
        and result.get("config_sha256") == expected_config_hash
        and result.get("protocol_tag") == expected_protocol_tag
    )
    if not required_identity:
        raise CandidateIntegrityError("production run identity/status mismatch")
    try:
        binary_native = np.load(run_dir / "design_binary_64.npy", allow_pickle=False)
        continuous_native = np.load(
            run_dir / "design_continuous_64.npy",
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise CandidateIntegrityError(
            "production design arrays are unreadable"
        ) from error
    for name, values in (
        ("binary", binary_native),
        ("continuous", continuous_native),
    ):
        if values.shape != (64, 64) or not np.all(np.isfinite(values)):
            raise CandidateIntegrityError(f"{name} design shape/finite check failed")
        if np.any((values < 0.0) | (values > 1.0)):
            raise CandidateIntegrityError(f"{name} design domain check failed")
    if not np.all((binary_native == 0.0) | (binary_native == 1.0)):
        raise CandidateIntegrityError("binary design is not strict binary")

    binary_optimization_hash = optimization_array_sha256(
        torch.from_numpy(binary_native)
    )
    continuous_optimization_hash = optimization_array_sha256(
        torch.from_numpy(continuous_native)
    )
    if binary_optimization_hash != result.get("binary_design_sha256"):
        raise CandidateIntegrityError("binary design hash mismatch")
    if continuous_optimization_hash != result.get("continuous_design_sha256"):
        raise CandidateIntegrityError("continuous design hash mismatch")
    if not math.isclose(
        float(np.mean(binary_native)),
        float(result.get("binary_material_fraction", math.nan)),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CandidateIntegrityError("binary material fraction mismatch")
    if not math.isclose(
        float(np.mean(continuous_native)),
        float(result.get("continuous_material_fraction", math.nan)),
        rel_tol=0.0,
        abs_tol=1.0e-7,
    ):
        raise CandidateIntegrityError("continuous material fraction mismatch")

    runtime = _optimization_runtime(run_dir / "optimization_metrics.csv")
    binary = _frozen_array(binary_native)
    continuous = _frozen_array(continuous_native)
    return ProductionCandidate(
        binary=FrozenCandidate(
            candidate_id=candidate_id,
            category=category,
            representation="binary",
            seed=seed,
            design=binary,
            design_hash=array_sha256(binary),
            source_runtime_seconds=runtime,
        ),
        continuous=FrozenCandidate(
            candidate_id=f"{candidate_id}_continuous",
            category=category,
            representation="continuous",
            seed=seed,
            design=continuous,
            design_hash=array_sha256(continuous),
            source_runtime_seconds=runtime,
        ),
    )


def _baseline_candidate(
    *,
    candidate_id: str,
    category: Literal["random", "straight", "dispersed", "uniform"],
    representation: Representation,
    seed: int | None,
    design: NDArray[np.float64],
    runtime: float,
) -> FrozenCandidate:
    frozen = _frozen_array(design)
    return FrozenCandidate(
        candidate_id=candidate_id,
        category=category,
        representation=representation,
        seed=seed,
        design=frozen,
        design_hash=array_sha256(frozen),
        source_runtime_seconds=runtime,
    )


def build_candidate_registry(production_root: Path) -> CandidateRegistry:
    """Build the exact pre-registered candidate sets and validate all runs."""
    manifest = _load_json(production_root.parent / "production_manifest.json")
    config_hash = manifest.get("config_sha256")
    protocol_tag = manifest.get("protocol_tag")
    if not isinstance(config_hash, str) or not isinstance(protocol_tag, str):
        raise CandidateIntegrityError("production manifest identity is invalid")

    grid = Grid2D(nx=64, ny=64)
    binary: list[FrozenCandidate] = []
    for seed in RANDOM_BASELINE_SEEDS:
        started = time.perf_counter()
        baseline = random_filtered_baseline(grid, seed)
        runtime = time.perf_counter() - started
        binary.append(
            _baseline_candidate(
                candidate_id=baseline.name,
                category="random",
                representation="binary",
                seed=seed,
                design=baseline.design,
                runtime=runtime,
            )
        )
    for constructor, category in (
        (straight_path_baseline, "straight"),
        (dispersed_baseline, "dispersed"),
    ):
        started = time.perf_counter()
        baseline = constructor(grid)
        runtime = time.perf_counter() - started
        binary.append(
            _baseline_candidate(
                candidate_id=baseline.name,
                category=category,  # type: ignore[arg-type]
                representation="binary",
                seed=None,
                design=baseline.design,
                runtime=runtime,
            )
        )

    continuous: list[FrozenCandidate] = []
    started = time.perf_counter()
    uniform = uniform_relaxed_baseline(grid)
    runtime = time.perf_counter() - started
    continuous.append(
        _baseline_candidate(
            candidate_id=uniform.name,
            category="uniform",
            representation="continuous",
            seed=None,
            design=uniform.design,
            runtime=runtime,
        )
    )

    for scope, category in (("single_A", "single_A"), ("robust", "robust")):
        for seed in PRODUCTION_SEEDS:
            loaded = load_production_candidate(
                production_root / scope / str(seed),
                candidate_id=f"{scope}_{seed}",
                category=category,  # type: ignore[arg-type]
                seed=seed,
                expected_config_hash=config_hash,
                expected_protocol_tag=protocol_tag,
            )
            binary.append(loaded.binary)
            continuous.append(loaded.continuous)
    return CandidateRegistry(
        binary=tuple(binary),
        continuous=tuple(continuous),
        config_hash=config_hash,
        protocol_tag=protocol_tag,
    )


def baseline_ids_for_seed(seed: int) -> tuple[str, ...]:
    """Return the exact six-member budget-matched set for robust seed `seed`."""
    if seed not in PRODUCTION_SEEDS:
        raise ValueError("seed is not registered for Gate 2A")
    return (
        "random_filtered_seed_9101",
        "random_filtered_seed_9102",
        "random_filtered_seed_9103",
        "straight_path",
        "evenly_dispersed_binary",
        f"single_A_{seed}",
    )


def _verification_record_is_valid(
    result: CandidateVerification,
    candidate: FrozenCandidate,
    fidelity: Fidelity,
) -> bool:
    expected_resolution = {
        "low_64": 64,
        "reference_128": 128,
        "reference_256": 256,
    }[fidelity]
    aggregate = (
        result.material_fraction,
        result.total_variation,
        result.worst_peak,
        result.average_peak,
        result.protected_zone_peak,
        result.total_wall_seconds,
    )
    if (
        result.candidate_id != candidate.candidate_id
        or result.fidelity != fidelity
        or result.grid_shape != (expected_resolution, expected_resolution)
        or result.design_hash_64 != candidate.design_hash
        or result.is_binary != (candidate.representation == "binary")
        or not all(math.isfinite(value) for value in aggregate)
        or result.worst_peak <= 0.0
        or result.total_wall_seconds < 0.0
        or not math.isclose(
            result.material_fraction,
            float(np.mean(candidate.design)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        return False
    if tuple(record.scenario_id for record in result.scenario_records) != (
        "A",
        "B",
        "C",
    ):
        return False
    return all(
        record.candidate_id == candidate.candidate_id
        and record.fidelity == fidelity
        and math.isfinite(record.peak_temperature)
        and record.peak_temperature > 0.0
        and math.isfinite(record.protected_zone_peak)
        and math.isfinite(record.normalized_residual)
        and record.normalized_residual <= 1.0e-10
        and math.isfinite(record.wall_seconds)
        and record.wall_seconds >= 0.0
        and math.isclose(
            record.integrated_power,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for record in result.scenario_records
    )


def run_nominal_verification(
    registry: CandidateRegistry,
    *,
    verifier: Callable[..., CandidateVerification] = verify_candidate,
) -> NominalVerificationBundle:
    """Run mandatory nominal SciPy verification without conditional 256 skips."""
    binary_results: list[CandidateVerification] = []
    binary_validity: list[bool] = []
    for candidate in registry.binary:
        for fidelity in ("low_64", "reference_128", "reference_256"):
            result = verifier(
                candidate.candidate_id,
                candidate.design,
                fidelity=fidelity,
                expected_design_hash=candidate.design_hash,
            )
            binary_results.append(result)
            binary_validity.append(
                _verification_record_is_valid(result, candidate, fidelity)
            )

    continuous_results: list[CandidateVerification] = []
    continuous_validity: list[bool] = []
    for candidate in registry.continuous:
        result = verifier(
            candidate.candidate_id,
            candidate.design,
            fidelity="reference_128",
            expected_design_hash=candidate.design_hash,
        )
        continuous_results.append(result)
        continuous_validity.append(
            _verification_record_is_valid(result, candidate, "reference_128")
        )
    valid = all(binary_validity) and all(continuous_validity)

    primary = {
        result.candidate_id: result
        for result in binary_results
        if result.fidelity == "reference_256"
    }
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in (*registry.binary, *registry.continuous)
    }
    seed_verdicts: dict[int, SeedVerdict] = {}
    for seed in PRODUCTION_SEEDS:
        robust_id = f"robust_{seed}"
        robust_continuous_id = f"{robust_id}_continuous"
        baseline_peaks = {
            baseline_id: primary[baseline_id].worst_peak
            for baseline_id in baseline_ids_for_seed(seed)
        }
        seed_verdicts[seed] = classify_nominal_seed(
            candidate_peak=primary[robust_id].worst_peak,
            baseline_peaks=baseline_peaks,
            continuous_fraction=float(
                np.mean(candidate_by_id[robust_continuous_id].design)
            ),
            binary_fraction=primary[robust_id].material_fraction,
            valid=valid,
        )
    return NominalVerificationBundle(
        binary=tuple(binary_results),
        continuous=tuple(continuous_results),
        seed_verdicts=seed_verdicts,
        valid=valid,
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise CandidateIntegrityError(f"refusing to write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_nominal_artifacts(
    bundle: NominalVerificationBundle,
    registry: CandidateRegistry,
    output_dir: Path,
) -> None:
    """Serialize immutable nominal metrics without recomputing decisions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_id = {
        candidate.candidate_id: candidate
        for candidate in (*registry.binary, *registry.continuous)
    }

    def aggregate_row(result: CandidateVerification) -> dict[str, object]:
        candidate = candidate_by_id[result.candidate_id]
        return {
            "candidate_id": result.candidate_id,
            "category": candidate.category,
            "seed": "" if candidate.seed is None else candidate.seed,
            "representation": candidate.representation,
            "fidelity": result.fidelity,
            "grid_ny": result.grid_shape[0],
            "grid_nx": result.grid_shape[1],
            "design_hash_64": result.design_hash_64,
            "transferred_design_hash": result.transferred_design_hash,
            "is_binary": result.is_binary,
            "material_fraction": result.material_fraction,
            "total_variation": result.total_variation,
            "worst_peak": result.worst_peak,
            "average_peak": result.average_peak,
            "protected_zone_peak": result.protected_zone_peak,
            "maximum_normalized_residual": max(
                record.normalized_residual for record in result.scenario_records
            ),
            "verification_wall_seconds": result.total_wall_seconds,
            "source_runtime_seconds": candidate.source_runtime_seconds,
        }

    _write_rows(
        output_dir / "nominal_binary_verification.csv",
        [aggregate_row(result) for result in bundle.binary],
    )
    _write_rows(
        output_dir / "nominal_continuous_verification.csv",
        [aggregate_row(result) for result in bundle.continuous],
    )

    scenario_rows: list[dict[str, object]] = []
    for result in (*bundle.binary, *bundle.continuous):
        candidate = candidate_by_id[result.candidate_id]
        for record in result.scenario_records:
            scenario_rows.append(
                {
                    "candidate_id": record.candidate_id,
                    "category": candidate.category,
                    "seed": "" if candidate.seed is None else candidate.seed,
                    "representation": candidate.representation,
                    "fidelity": record.fidelity,
                    "scenario_id": record.scenario_id,
                    "peak_temperature": record.peak_temperature,
                    "protected_zone_peak": record.protected_zone_peak,
                    "normalized_residual": record.normalized_residual,
                    "wall_seconds": record.wall_seconds,
                    "source_hash": record.source_hash,
                    "integrated_power": record.integrated_power,
                }
            )
    _write_rows(output_dir / "nominal_scenario_records.csv", scenario_rows)

    verdict_payload = {
        "schema_version": 2,
        "valid": bundle.valid,
        "config_sha256": registry.config_hash,
        "protocol_tag": registry.protocol_tag,
        "seeds": {
            str(seed): {
                "status": verdict.status.value,
                "reason_codes": list(verdict.reason_codes),
                "metrics": verdict.metrics,
            }
            for seed, verdict in bundle.seed_verdicts.items()
        },
    }
    (output_dir / "nominal_verdicts.json").write_text(
        json.dumps(verdict_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _perturbation_evaluation_is_valid(
    evaluation: PerturbationEvaluation,
    candidate: FrozenCandidate,
    case: PerturbationCase,
) -> bool:
    values = (
        *evaluation.scenario_peaks,
        *evaluation.scenario_residuals,
        evaluation.worst_peak,
        evaluation.k_high,
        evaluation.wall_seconds,
        evaluation.material_fraction,
    )
    return (
        evaluation.candidate_id == candidate.candidate_id
        and evaluation.case_id == case.case_id
        and evaluation.grid_shape == (256, 256)
        and evaluation.design_hash_64 == candidate.design_hash
        and len(evaluation.scenario_peaks) == 3
        and len(evaluation.scenario_residuals) == 3
        and all(math.isfinite(value) for value in values)
        and all(peak > 0.0 for peak in evaluation.scenario_peaks)
        and all(
            0.0 <= residual <= 1.0e-10 for residual in evaluation.scenario_residuals
        )
        and math.isclose(
            evaluation.worst_peak,
            max(evaluation.scenario_peaks),
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            evaluation.material_fraction,
            float(np.mean(candidate.design)),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and evaluation.k_high == case.k_high
        and evaluation.wall_seconds >= 0.0
    )


def run_robustness_verification(
    registry: CandidateRegistry,
    nominal: NominalVerificationBundle,
    *,
    evaluator: Callable[..., PerturbationEvaluation] = evaluate_perturbation_case,
) -> RobustnessVerificationBundle:
    """Evaluate the exact 28-case registry with per-case baseline selection."""
    eligible_seeds = tuple(
        seed
        for seed in PRODUCTION_SEEDS
        if nominal.seed_verdicts[seed].status is Gate2Status.PASS
    )
    required_ids: set[str] = set()
    for seed in eligible_seeds:
        required_ids.update(baseline_ids_for_seed(seed))
        required_ids.add(f"robust_{seed}")
    candidates = tuple(
        candidate
        for candidate in registry.binary
        if candidate.candidate_id in required_ids
    )
    cases = registered_primary_cases()
    evaluations: list[PerturbationEvaluation] = []
    validity: list[bool] = []
    for case in cases:
        for candidate in candidates:
            evaluation = evaluator(
                candidate.candidate_id,
                candidate.design,
                case,
            )
            evaluations.append(evaluation)
            validity.append(
                _perturbation_evaluation_is_valid(evaluation, candidate, case)
            )
    valid = nominal.valid and all(validity) and len(cases) == 28
    by_candidate_case = {
        (evaluation.candidate_id, evaluation.case_id): evaluation
        for evaluation in evaluations
    }
    comparisons: list[RobustnessComparison] = []
    seed_verdicts: dict[int, SeedVerdict] = {}
    for seed in eligible_seeds:
        improvements: list[float] = []
        for case in cases:
            baseline_peaks = {
                baseline_id: by_candidate_case[(baseline_id, case.case_id)].worst_peak
                for baseline_id in baseline_ids_for_seed(seed)
            }
            strongest = select_strongest_baseline(baseline_peaks)
            candidate_id = f"robust_{seed}"
            candidate_peak = by_candidate_case[(candidate_id, case.case_id)].worst_peak
            improvement = relative_improvement(
                strongest.worst_peak,
                candidate_peak,
            )
            improvements.append(improvement)
            comparisons.append(
                RobustnessComparison(
                    seed=seed,
                    case_id=case.case_id,
                    candidate_id=candidate_id,
                    candidate_peak=candidate_peak,
                    strongest_baseline_id=strongest.candidate_id,
                    strongest_baseline_peak=strongest.worst_peak,
                    relative_improvement=improvement,
                    passed_two_percent=improvement >= 0.02,
                )
            )
        seed_verdicts[seed] = classify_seed_robustness(
            improvements,
            valid=valid,
        )
    return RobustnessVerificationBundle(
        evaluations=tuple(evaluations),
        comparisons=tuple(comparisons),
        seed_verdicts=seed_verdicts,
        valid=valid,
    )


def run_morphology_verification(
    registry: CandidateRegistry,
    *,
    diagnostic: Callable[..., tuple[MorphologyRecord, ...]] = morphology_diagnostics,
) -> MorphologyVerificationBundle:
    """Run morphology separately for every frozen strict-binary design."""
    records: list[MorphologyRecord] = []
    valid = True
    for candidate in registry.binary:
        candidate_records = diagnostic(candidate.candidate_id, candidate.design)
        records.extend(candidate_records)
        if tuple(record.operation for record in candidate_records) != (
            "unperturbed",
            "erosion",
            "dilation",
        ):
            valid = False
            continue
        base, eroded, dilated = candidate_records
        numerical = tuple(
            value
            for record in candidate_records
            for value in (
                record.material_fraction,
                record.worst_peak_256,
                record.relative_degradation,
            )
        )
        valid = valid and (
            len(candidate_records) == 3
            and all(
                record.candidate_id == candidate.candidate_id
                for record in candidate_records
            )
            and base.design_hash_64 == candidate.design_hash
            and all(math.isfinite(value) for value in numerical)
            and all(record.worst_peak_256 > 0.0 for record in candidate_records)
            and all(record.component_count >= 0 for record in candidate_records)
            and math.isclose(base.relative_degradation, 0.0, abs_tol=1.0e-15)
            and eroded.material_fraction <= base.material_fraction
            and dilated.material_fraction >= base.material_fraction
        )
    valid = valid and len(records) == len(registry.binary) * 3
    return MorphologyVerificationBundle(records=tuple(records), valid=valid)


def write_robustness_artifacts(
    robustness: RobustnessVerificationBundle,
    morphology: MorphologyVerificationBundle,
    registry: CandidateRegistry,
    output_dir: Path,
) -> None:
    """Write the literal registry, raw solves and derived robustness decisions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_rows = [
        {
            "candidate_id": evaluation.candidate_id,
            "case_id": evaluation.case_id,
            "grid_ny": evaluation.grid_shape[0],
            "grid_nx": evaluation.grid_shape[1],
            "design_hash_64": evaluation.design_hash_64,
            "material_fraction": evaluation.material_fraction,
            "scenario_A_peak": evaluation.scenario_peaks[0],
            "scenario_B_peak": evaluation.scenario_peaks[1],
            "scenario_C_peak": evaluation.scenario_peaks[2],
            "scenario_A_residual": evaluation.scenario_residuals[0],
            "scenario_B_residual": evaluation.scenario_residuals[1],
            "scenario_C_residual": evaluation.scenario_residuals[2],
            "worst_peak": evaluation.worst_peak,
            "k_high": evaluation.k_high,
            "wall_seconds": evaluation.wall_seconds,
        }
        for evaluation in robustness.evaluations
    ]
    _write_rows(output_dir / "perturbation_evaluations.csv", evaluation_rows)
    _write_rows(
        output_dir / "robustness_metrics.csv",
        [asdict(comparison) for comparison in robustness.comparisons],
    )
    _write_rows(
        output_dir / "morphology_metrics.csv",
        [asdict(record) for record in morphology.records],
    )
    registry_payload = {
        "schema_version": 2,
        "config_sha256": registry.config_hash,
        "protocol_tag": registry.protocol_tag,
        "cases": [asdict(case) for case in registered_primary_cases()],
    }
    (output_dir / "perturbation_registry.json").write_text(
        json.dumps(registry_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict_payload = {
        "schema_version": 2,
        "valid": robustness.valid and morphology.valid,
        "robustness_valid": robustness.valid,
        "morphology_valid": morphology.valid,
        "config_sha256": registry.config_hash,
        "protocol_tag": registry.protocol_tag,
        "seeds": {
            str(seed): {
                "status": verdict.status.value,
                "reason_codes": list(verdict.reason_codes),
                "metrics": verdict.metrics,
            }
            for seed, verdict in robustness.seed_verdicts.items()
        },
    }
    (output_dir / "robustness_verdicts.json").write_text(
        json.dumps(verdict_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run and serialize one mandatory Gate 2A verification stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("nominal", "robustness"),
        default="nominal",
    )
    parser.add_argument(
        "--production-root",
        type=Path,
        default=Path("artifacts/gate2_design/production"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gate2_design/verification"),
    )
    arguments = parser.parse_args()
    registry = build_candidate_registry(arguments.production_root)
    nominal = run_nominal_verification(registry)
    if arguments.stage == "nominal":
        write_nominal_artifacts(nominal, registry, arguments.output)
        payload = {
            "stage": "nominal",
            "valid": nominal.valid,
            "binary_records": len(nominal.binary),
            "continuous_records": len(nominal.continuous),
            "seed_statuses": {
                str(seed): verdict.status.value
                for seed, verdict in nominal.seed_verdicts.items()
            },
        }
        valid = nominal.valid
    else:
        robustness = run_robustness_verification(registry, nominal)
        morphology = run_morphology_verification(registry)
        write_robustness_artifacts(
            robustness,
            morphology,
            registry,
            arguments.output,
        )
        payload = {
            "stage": "robustness",
            "valid": robustness.valid and morphology.valid,
            "perturbation_evaluations": len(robustness.evaluations),
            "comparisons": len(robustness.comparisons),
            "morphology_records": len(morphology.records),
            "seed_statuses": {
                str(seed): verdict.status.value
                for seed, verdict in robustness.seed_verdicts.items()
            },
        }
        valid = robustness.valid and morphology.valid
    print(json.dumps(payload, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
