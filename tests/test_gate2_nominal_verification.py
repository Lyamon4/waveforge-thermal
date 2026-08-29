"""Tests for mandatory nominal Gate 2A verification orchestration."""

import csv
import json
from pathlib import Path

from waveforge.experiments.verify_gate2a import (
    build_candidate_registry,
    run_nominal_verification,
    write_nominal_artifacts,
)
from waveforge.verification.compare import Gate2Status
from waveforge.verification.high_fidelity import (
    CandidateVerification,
    VerificationRecord,
)


def test_nominal_orchestration_verifies_all_locked_fidelities(
    tmp_path: Path,
) -> None:
    """Conditional 256 verification or a binary/continuous omission must fail."""
    registry = build_candidate_registry(Path("artifacts/gate2_design/production"))
    calls: list[tuple[str, str]] = []

    def fake_verifier(candidate_id: str, design, *, fidelity, **kwargs):
        calls.append((candidate_id, fidelity))
        is_robust = candidate_id.startswith("robust_")
        is_continuous = candidate_id.endswith("_continuous") or candidate_id == (
            "uniform_relaxed"
        )
        peak = 0.90 if is_robust else 1.0
        records = tuple(
            VerificationRecord(
                candidate_id=candidate_id,
                fidelity=fidelity,
                scenario_id=scenario_id,
                peak_temperature=peak,
                protected_zone_peak=0.5 * peak,
                normalized_residual=1.0e-13,
                wall_seconds=0.01,
                source_hash=f"source_{scenario_id}",
                integrated_power=1.0,
            )
            for scenario_id in ("A", "B", "C")
        )
        resolution = {"low_64": 64, "reference_128": 128, "reference_256": 256}[
            fidelity
        ]
        return CandidateVerification(
            candidate_id=candidate_id,
            fidelity=fidelity,
            grid_shape=(resolution, resolution),
            design_hash_64=kwargs["expected_design_hash"],
            transferred_design_hash="transferred",
            is_binary=not is_continuous,
            material_fraction=float(design.mean()),
            total_variation=0.1,
            worst_peak=peak,
            average_peak=peak,
            protected_zone_peak=0.5 * peak,
            total_wall_seconds=0.03,
            scenario_records=records,
            claimed_worst_peak=None,
            claim_matches=None,
        )

    result = run_nominal_verification(registry, verifier=fake_verifier)

    assert len(result.binary) == 33
    assert len(result.continuous) == 7
    assert len(calls) == 40
    for candidate in registry.binary:
        assert {
            fidelity
            for candidate_id, fidelity in calls
            if candidate_id == candidate.candidate_id
        } == {"low_64", "reference_128", "reference_256"}
    for candidate in registry.continuous:
        assert (candidate.candidate_id, "reference_128") in calls
    assert result.valid
    assert all(
        verdict.status is Gate2Status.PASS for verdict in result.seed_verdicts.values()
    )
    assert all(
        verdict.metrics["strongest_baseline_id"] == "evenly_dispersed_binary"
        for verdict in result.seed_verdicts.values()
    )

    write_nominal_artifacts(result, registry, tmp_path)
    with (tmp_path / "nominal_binary_verification.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 33
    with (tmp_path / "nominal_continuous_verification.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 7
    with (tmp_path / "nominal_scenario_records.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 120
    payload = json.loads((tmp_path / "nominal_verdicts.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["valid"] is True
    assert {entry["status"] for entry in payload["seeds"].values()} == {"PASS"}
