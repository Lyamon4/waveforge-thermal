"""Tests for registered robustness comparison orchestration."""

import csv
import json
from pathlib import Path

from waveforge.experiments.verify_gate2a import (
    NominalVerificationBundle,
    build_candidate_registry,
    run_morphology_verification,
    run_robustness_verification,
    write_robustness_artifacts,
)
from waveforge.verification.compare import Gate2Status, SeedVerdict
from waveforge.verification.high_fidelity import array_sha256
from waveforge.verification.perturbations import (
    MorphologyRecord,
    PerturbationEvaluation,
)


def test_robustness_evaluates_every_candidate_case_and_reselects_baseline(
    tmp_path: Path,
) -> None:
    """Reusing nominal baseline identity or skipping registered cases must fail."""
    registry = build_candidate_registry(Path("artifacts/gate2_design/production"))
    nominal = NominalVerificationBundle(
        binary=(),
        continuous=(),
        seed_verdicts={
            seed: SeedVerdict(Gate2Status.PASS)
            for seed in (20260828, 20260829, 20260830)
        },
        valid=True,
    )
    calls: list[tuple[str, str]] = []

    def fake_evaluator(candidate_id, design, case):
        calls.append((candidate_id, case.case_id))
        peak = 0.9 if candidate_id.startswith("robust_") else 1.0
        return PerturbationEvaluation(
            candidate_id=candidate_id,
            case_id=case.case_id,
            grid_shape=(256, 256),
            design_hash_64=array_sha256(design),
            material_fraction=float(design.mean()),
            scenario_peaks=(peak, peak, peak),
            scenario_residuals=(1.0e-13, 1.0e-13, 1.0e-13),
            worst_peak=peak,
            k_high=case.k_high,
            wall_seconds=0.03,
        )

    result = run_robustness_verification(
        registry,
        nominal,
        evaluator=fake_evaluator,
    )

    assert result.valid
    assert len(calls) == 11 * 28
    assert len(result.evaluations) == 11 * 28
    assert len(result.comparisons) == 3 * 28
    assert all(
        verdict.status is Gate2Status.PASS for verdict in result.seed_verdicts.values()
    )
    assert all(
        verdict.metrics["passing_cases"] == 28
        for verdict in result.seed_verdicts.values()
    )
    assert {comparison.strongest_baseline_id for comparison in result.comparisons} == {
        "evenly_dispersed_binary"
    }

    def fake_morphology(candidate_id, design):
        base_fraction = float(design.mean())
        return tuple(
            MorphologyRecord(
                candidate_id=candidate_id,
                operation=operation,
                material_fraction=fraction,
                worst_peak_256=1.0,
                component_count=1,
                relative_degradation=0.0,
                design_hash_64=(
                    array_sha256(design) if operation == "unperturbed" else operation
                ),
            )
            for operation, fraction in (
                ("unperturbed", base_fraction),
                ("erosion", max(0.0, base_fraction - 0.01)),
                ("dilation", min(1.0, base_fraction + 0.01)),
            )
        )

    morphology = run_morphology_verification(
        registry,
        diagnostic=fake_morphology,
    )
    assert morphology.valid
    assert len(morphology.records) == 11 * 3

    write_robustness_artifacts(result, morphology, registry, tmp_path)
    with (tmp_path / "perturbation_evaluations.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 11 * 28
    with (tmp_path / "robustness_metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3 * 28
    with (tmp_path / "morphology_metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 11 * 3
    registry_payload = json.loads((tmp_path / "perturbation_registry.json").read_text())
    assert len(registry_payload["cases"]) == 28
    verdict_payload = json.loads((tmp_path / "robustness_verdicts.json").read_text())
    assert {entry["status"] for entry in verdict_payload["seeds"].values()} == {"PASS"}


def test_nominal_no_go_seed_is_reported_without_running_perturbations() -> None:
    """A scientific nominal failure must not become a missing-artifact invalidity."""
    registry = build_candidate_registry(Path("artifacts/gate2_design/production"))
    nominal = NominalVerificationBundle(
        binary=(),
        continuous=(),
        seed_verdicts={
            20260828: SeedVerdict(Gate2Status.PASS),
            20260829: SeedVerdict(Gate2Status.NO_GO_EFFECT),
            20260830: SeedVerdict(Gate2Status.NO_GO_EFFECT),
        },
        valid=True,
    )
    calls: list[tuple[str, str]] = []

    def fake_evaluator(candidate_id, design, case):
        calls.append((candidate_id, case.case_id))
        peak = 0.9 if candidate_id.startswith("robust_") else 1.0
        return PerturbationEvaluation(
            candidate_id=candidate_id,
            case_id=case.case_id,
            grid_shape=(256, 256),
            design_hash_64=array_sha256(design),
            material_fraction=float(design.mean()),
            scenario_peaks=(peak, peak, peak),
            scenario_residuals=(1.0e-13, 1.0e-13, 1.0e-13),
            worst_peak=peak,
            k_high=case.k_high,
            wall_seconds=0.03,
        )

    result = run_robustness_verification(
        registry,
        nominal,
        evaluator=fake_evaluator,
    )

    assert len(calls) == 7 * 28
    assert result.seed_verdicts[20260828].status is Gate2Status.PASS
    for seed in (20260829, 20260830):
        assert result.seed_verdicts[seed].status is Gate2Status.NO_GO_EFFECT
        assert result.seed_verdicts[seed].reason_codes == ("NOMINAL_GATE_NOT_PASSED",)
