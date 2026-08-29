"""Exhaustive prospective strong branching-baseline challenge."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from waveforge.design.branching_baseline import (
    BranchingTreeParameters,
    build_branching_tree,
    candidate_axes,
    iter_candidate_parameters,
)
from waveforge.verification.challenge import (
    ChallengeEvaluation,
    ChallengeSeedComparison,
    ChallengeStatus,
    ChallengeVerdict,
    classify_challenge,
    evaluate_frozen_binary_design,
)
from waveforge.verification.perturbations import (
    MorphologyRecord,
    PerturbationCase,
    PerturbationEvaluation,
    evaluate_perturbation_case,
    morphology_diagnostics,
    registered_primary_cases,
)

Evaluator = Callable[..., ChallengeEvaluation]
PerturbationEvaluator = Callable[
    [str, NDArray[np.float64], PerturbationCase],
    PerturbationEvaluation,
]
MorphologyEvaluator = Callable[
    [str, NDArray[np.float64]],
    tuple[MorphologyRecord, ...],
]
PRODUCTION_SEEDS = (20260828, 20260829, 20260830)


@dataclass(frozen=True)
class SearchRecord:
    """One candidate evaluation at one fidelity with deterministic rank."""

    parameters: BranchingTreeParameters
    evaluation: ChallengeEvaluation
    rank: int = 0


@dataclass(frozen=True)
class SearchResult:
    """Complete `all→20→5` multi-fidelity funnel result."""

    search_64: tuple[SearchRecord, ...]
    finalists_128: tuple[SearchRecord, ...]
    finalists_256: tuple[SearchRecord, ...]
    winner: SearchRecord


def _rank(records: list[SearchRecord]) -> tuple[SearchRecord, ...]:
    ordered = sorted(
        records,
        key=lambda record: (
            record.evaluation.worst_peak,
            record.parameters.candidate_id,
        ),
    )
    return tuple(
        replace(record, rank=index + 1) for index, record in enumerate(ordered)
    )


def _evaluate(
    parameters: Sequence[BranchingTreeParameters],
    *,
    resolution: int,
    evaluator: Evaluator,
) -> tuple[SearchRecord, ...]:
    records: list[SearchRecord] = []
    for candidate in parameters:
        design = build_branching_tree(candidate).design
        evaluation = evaluator(
            candidate.candidate_id,
            design,
            resolution=resolution,
        )
        records.append(SearchRecord(parameters=candidate, evaluation=evaluation))
    return _rank(records)


def _row(record: SearchRecord) -> dict[str, object]:
    evaluation = record.evaluation
    return {
        "rank": record.rank,
        "candidate_id": record.parameters.candidate_id,
        "x_sink": record.parameters.x_sink,
        "x_junction": record.parameters.x_junction,
        "y_junction": record.parameters.y_junction,
        "trunk_to_branch_width_ratio": (record.parameters.trunk_to_branch_width_ratio),
        "resolution": evaluation.resolution,
        "design_hash_64": evaluation.design_hash_64,
        "transferred_design_hash": evaluation.transferred_design_hash,
        "material_fraction": evaluation.material_fraction,
        "scenario_A_peak": evaluation.scenario_peaks[0],
        "scenario_B_peak": evaluation.scenario_peaks[1],
        "scenario_C_peak": evaluation.scenario_peaks[2],
        "worst_peak": evaluation.worst_peak,
        "average_peak": evaluation.average_peak,
        "maximum_residual": evaluation.maximum_residual,
        "wall_seconds": evaluation.wall_seconds,
    }


def _write_csv(path: Path, records: tuple[SearchRecord, ...]) -> None:
    if not records:
        raise ValueError("refusing to write an empty search table")
    rows = [_row(record) for record in records]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty challenge table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()


def run_search(
    output_dir: Path,
    *,
    parameters: Sequence[BranchingTreeParameters] | None = None,
    evaluator: Evaluator = evaluate_frozen_binary_design,
) -> SearchResult:
    """Run and serialize the locked `64→128→256` candidate funnel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    production = parameters is None
    selected_parameters = (
        tuple(iter_candidate_parameters()) if parameters is None else tuple(parameters)
    )
    if not selected_parameters:
        raise ValueError("challenge candidate registry must not be empty")
    search_64 = _evaluate(
        selected_parameters,
        resolution=64,
        evaluator=evaluator,
    )
    finalist_parameters_128 = tuple(
        record.parameters for record in search_64[: min(20, len(search_64))]
    )
    finalists_128 = _evaluate(
        finalist_parameters_128,
        resolution=128,
        evaluator=evaluator,
    )
    finalist_parameters_256 = tuple(
        record.parameters for record in finalists_128[: min(5, len(finalists_128))]
    )
    finalists_256 = _evaluate(
        finalist_parameters_256,
        resolution=256,
        evaluator=evaluator,
    )
    if production and (
        len(search_64) != 41055 or len(finalists_128) != 20 or len(finalists_256) != 5
    ):
        raise RuntimeError("production challenge funnel count mismatch")

    _write_csv(output_dir / "tree_search_64.csv", search_64)
    _write_csv(output_dir / "tree_finalists_128.csv", finalists_128)
    _write_csv(output_dir / "tree_finalists_256.csv", finalists_256)
    return SearchResult(
        search_64=search_64,
        finalists_128=finalists_128,
        finalists_256=finalists_256,
        winner=finalists_256[0],
    )


def _write_candidate_registry(
    output_dir: Path,
    search: SearchResult,
) -> None:
    axes = candidate_axes()
    spec_path = output_dir / "challenge_spec.md"
    if not spec_path.is_file():
        raise FileNotFoundError("locked challenge_spec.md is missing")
    payload = {
        "schema_version": 1,
        "post_result_challenge": True,
        "spec_sha256": _file_sha256(spec_path),
        "implementation_git_sha": _git_sha(),
        "candidate_count": 41055,
        "axes": {
            "x_sink": list(axes[0]),
            "x_junction": list(axes[1]),
            "y_junction": list(axes[2]),
            "trunk_to_branch_width_ratio": list(axes[3]),
        },
        "score": "negative_minimum_normalized_segment_distance_v1",
        "selected_cells": 1024,
        "tie_break": "lower_row_major_index",
        "transfer": {"128": "exact_2x2_replication", "256": "exact_4x4_replication"},
        "search_counts": {
            "64": len(search.search_64),
            "128": len(search.finalists_128),
            "256": len(search.finalists_256),
        },
        "winner": {
            "candidate_id": search.winner.parameters.candidate_id,
            "parameters": {
                "x_sink": search.winner.parameters.x_sink,
                "x_junction": search.winner.parameters.x_junction,
                "y_junction": search.winner.parameters.y_junction,
                "trunk_to_branch_width_ratio": (
                    search.winner.parameters.trunk_to_branch_width_ratio
                ),
            },
        },
    }
    (output_dir / "candidate_registry.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validated_registered_cases(gate2_root: Path) -> tuple[PerturbationCase, ...]:
    cases = registered_primary_cases()
    payload = json.loads(
        (gate2_root / "verification" / "perturbation_registry.json").read_text(
            encoding="utf-8"
        )
    )
    if payload.get("cases") != [asdict(case) for case in cases] or len(cases) != 28:
        raise RuntimeError("locked perturbation registry mismatch")
    return cases


def run_comparison(
    output_dir: Path,
    search: SearchResult,
    *,
    gate2_root: Path = Path("artifacts/gate2_design"),
    perturbation_evaluator: PerturbationEvaluator = evaluate_perturbation_case,
    morphology_evaluator: MorphologyEvaluator = morphology_diagnostics,
    enforce_production_counts: bool = True,
) -> ChallengeVerdict:
    """Compare the selected tree with frozen WaveForge raw verification data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = True
    if enforce_production_counts and (
        len(search.search_64) != 41055
        or len(search.finalists_128) != 20
        or len(search.finalists_256) != 5
    ):
        valid = False
    winner = search.winner
    if (
        winner.evaluation.resolution != 256
        or not math.isclose(
            winner.evaluation.material_fraction,
            0.25,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
        or winner.evaluation.maximum_residual > 1.0e-10
    ):
        valid = False
    tree_design = build_branching_tree(winner.parameters).design
    tree_id = winner.parameters.candidate_id

    nominal_rows = _read_rows(
        gate2_root / "verification" / "nominal_binary_verification.csv"
    )
    nominal_index = {
        row["candidate_id"]: row
        for row in nominal_rows
        if row["fidelity"] == "reference_256"
    }
    cases = _validated_registered_cases(gate2_root)
    tree_case_evaluations = tuple(
        perturbation_evaluator(tree_id, tree_design, case) for case in cases
    )
    if len(tree_case_evaluations) != 28:
        valid = False
    tree_case_index = {
        evaluation.case_id: evaluation for evaluation in tree_case_evaluations
    }
    waveforge_perturbation_rows = _read_rows(
        gate2_root / "verification" / "perturbation_evaluations.csv"
    )
    waveforge_case_index = {
        (row["candidate_id"], row["case_id"]): row
        for row in waveforge_perturbation_rows
    }

    robustness_rows: list[dict[str, object]] = []
    comparisons: list[ChallengeSeedComparison] = []
    nominal_output_rows: list[dict[str, object]] = []
    for seed in PRODUCTION_SEEDS:
        waveforge_id = f"robust_{seed}"
        waveforge_nominal_peak = float(nominal_index[waveforge_id]["worst_peak"])
        tree_nominal_peak = winner.evaluation.worst_peak
        nominal_improvement = (
            tree_nominal_peak - waveforge_nominal_peak
        ) / tree_nominal_peak
        passing_cases = 0
        for case in cases:
            tree_evaluation = tree_case_index[case.case_id]
            waveforge_peak = float(
                waveforge_case_index[(waveforge_id, case.case_id)]["worst_peak"]
            )
            improvement = (
                tree_evaluation.worst_peak - waveforge_peak
            ) / tree_evaluation.worst_peak
            passed = improvement >= 0.02
            passing_cases += int(passed)
            robustness_rows.append(
                {
                    "seed": seed,
                    "case_id": case.case_id,
                    "tree_candidate_id": tree_id,
                    "tree_peak": tree_evaluation.worst_peak,
                    "waveforge_candidate_id": waveforge_id,
                    "waveforge_peak": waveforge_peak,
                    "relative_improvement": improvement,
                    "passed_two_percent": passed,
                }
            )
        comparison = ChallengeSeedComparison(
            seed=seed,
            nominal_improvement=nominal_improvement,
            robustness_passing_cases=passing_cases,
        )
        comparisons.append(comparison)
        nominal_output_rows.append(
            {
                "seed": seed,
                "tree_candidate_id": tree_id,
                "tree_peak": tree_nominal_peak,
                "waveforge_candidate_id": waveforge_id,
                "waveforge_peak": waveforge_nominal_peak,
                "relative_improvement": nominal_improvement,
                "nominal_pass_5pct": comparison.nominal_pass_5pct,
                "robustness_passing_cases": passing_cases,
                "seed_strong_pass": comparison.seed_strong_pass,
            }
        )

    _write_rows(output_dir / "waveforge_vs_tree.csv", nominal_output_rows)
    _write_rows(output_dir / "challenge_robustness.csv", robustness_rows)
    tree_morphology = morphology_evaluator(tree_id, tree_design)
    morphology_rows = [
        {"method": "tree", **asdict(record)} for record in tree_morphology
    ]
    waveforge_morphology = _read_rows(
        gate2_root / "verification" / "morphology_metrics.csv"
    )
    morphology_rows.extend(
        {"method": "waveforge", **row}
        for row in waveforge_morphology
        if row["candidate_id"] in {f"robust_{seed}" for seed in PRODUCTION_SEEDS}
    )
    _write_rows(output_dir / "challenge_morphology.csv", morphology_rows)
    _write_candidate_registry(output_dir, search)

    verdict = classify_challenge(tuple(comparisons), valid=valid)
    verdict_payload = {
        "schema_version": 1,
        "status": verdict.status.value,
        "reason_codes": list(verdict.reason_codes),
        "post_result_challenge": True,
        "valid": verdict.status is not ChallengeStatus.INVALID_RUN,
        "winner": {
            "candidate_id": tree_id,
            "parameters": asdict(winner.parameters),
            "worst_peak_256": winner.evaluation.worst_peak,
            "material_fraction": winner.evaluation.material_fraction,
        },
        "metrics": verdict.metrics,
    }
    (output_dir / "challenge_verdict.json").write_text(
        json.dumps(verdict_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return verdict
