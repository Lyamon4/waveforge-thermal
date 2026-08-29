"""Measure the locked Stage C teacher cost and lower-fidelity agreement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from waveforge.ml.teacher import (
    TeacherConfig,
    TeacherStatus,
    optimize_teacher,
    verify_teacher_at_64,
)

PILOTS = (
    ("pilot_1", ((0.30, 0.70), (0.50, 0.70), (0.70, 0.70)), 31001),
    ("pilot_2", ((0.20, 0.60), (0.50, 0.70), (0.80, 0.60)), 31002),
    ("pilot_3", ((0.30, 0.60), (0.50, 0.70), (0.70, 0.60)), 31003),
)


class TeacherCostStatus(StrEnum):
    """Pre-dataset decision states with invalid-run precedence."""

    PASS = "PASS"
    ML_NO_GO_TEACHER_COST = "ML_NO_GO_TEACHER_COST"
    ML_NO_GO_TEACHER_FIDELITY = "ML_NO_GO_TEACHER_FIDELITY"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class PilotMeasurement:
    """One teacher fidelity measured and independently verified at `64×64`."""

    pilot_id: str
    resolution: int
    status: TeacherStatus
    wall_seconds: float
    verified_peak_64: float
    binary_fraction: float
    maximum_scipy_residual: float
    artifact_bytes: int
    result_sha256: str
    binary_sha256: str

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class TeacherCostAssessment:
    """Complete machine-readable outcome of the cost/fidelity gate."""

    status: TeacherCostStatus
    reason_codes: tuple[str, ...]
    base_spec_sha256: str
    amendment_sha256: str
    accepted_teacher_resolution: int | None
    spearman_rank_correlation: float | None
    relative_degradations: tuple[float, ...]
    median_relative_degradation: float | None
    maximum_relative_degradation: float | None
    projected_teacher_hours: float | None
    projected_artifact_gib: float | None
    measurements: tuple[PilotMeasurement, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "base_spec_sha256": self.base_spec_sha256,
            "amendment_sha256": self.amendment_sha256,
            "accepted_teacher_resolution": self.accepted_teacher_resolution,
            "fidelity_metrics": {
                "spearman_rank_correlation": self.spearman_rank_correlation,
                "relative_degradations": list(self.relative_degradations),
                "median_relative_degradation": (self.median_relative_degradation),
                "maximum_relative_degradation": (self.maximum_relative_degradation),
                "required_spearman": 0.5,
                "maximum_median_degradation": 0.10,
                "maximum_single_degradation": 0.20,
            },
            "cost_metrics": {
                "projected_teacher_hours": self.projected_teacher_hours,
                "teacher_hour_ceiling": 8.0,
                "projected_artifact_gib": self.projected_artifact_gib,
                "artifact_gib_ceiling": 5.0,
                "contingency_factor": 1.15,
                "future_training_validation_tasks": 20,
                "held_out_full_reference_tasks": 8,
            },
            "measurements": [item.to_payload() for item in self.measurements],
        }


def _invalid(
    measurements: tuple[PilotMeasurement, ...],
    base_spec_sha256: str,
    amendment_sha256: str,
    *reasons: str,
) -> TeacherCostAssessment:
    return TeacherCostAssessment(
        status=TeacherCostStatus.INVALID_RUN,
        reason_codes=tuple(reasons),
        base_spec_sha256=base_spec_sha256,
        amendment_sha256=amendment_sha256,
        accepted_teacher_resolution=None,
        spearman_rank_correlation=None,
        relative_degradations=(),
        median_relative_degradation=None,
        maximum_relative_degradation=None,
        projected_teacher_hours=None,
        projected_artifact_gib=None,
        measurements=measurements,
    )


def assess_teacher_cost(
    measurements: tuple[PilotMeasurement, ...],
    *,
    base_spec_sha256: str,
    amendment_sha256: str,
) -> TeacherCostAssessment:
    """Recompute the preregistered teacher gate from raw pilot measurements."""
    expected_keys = {
        (f"pilot_{index}", resolution)
        for index in range(1, 4)
        for resolution in (32, 64)
    }
    observed_keys = {(item.pilot_id, item.resolution) for item in measurements}
    if len(measurements) != 6 or observed_keys != expected_keys:
        return _invalid(
            measurements,
            base_spec_sha256,
            amendment_sha256,
            "INCOMPLETE_OR_DUPLICATE_PILOT_MATRIX",
        )
    ordered = tuple(
        sorted(measurements, key=lambda item: (item.pilot_id, item.resolution))
    )
    for item in ordered:
        if (
            item.status is TeacherStatus.INVALID_RUN
            or not math.isfinite(item.wall_seconds)
            or item.wall_seconds <= 0.0
            or not math.isfinite(item.verified_peak_64)
            or item.verified_peak_64 <= 0.0
            or not math.isfinite(item.maximum_scipy_residual)
            or item.maximum_scipy_residual > 1.0e-10
            or item.artifact_bytes <= 0
        ):
            return _invalid(
                ordered,
                base_spec_sha256,
                amendment_sha256,
                "INVALID_PILOT_NUMERICS_OR_ARTIFACT",
            )

    by_key = {(item.pilot_id, item.resolution): item for item in ordered}
    low = tuple(by_key[(f"pilot_{index}", 32)] for index in range(1, 4))
    high = tuple(by_key[(f"pilot_{index}", 64)] for index in range(1, 4))
    low_peaks = np.asarray([item.verified_peak_64 for item in low])
    high_peaks = np.asarray([item.verified_peak_64 for item in high])
    correlation = float(spearmanr(low_peaks, high_peaks).statistic)
    degradations = tuple(
        float(value) for value in (low_peaks - high_peaks) / high_peaks
    )
    median_degradation = float(np.median(degradations))
    maximum_degradation = max(degradations)

    fidelity_reasons: list[str] = []
    if not math.isfinite(correlation) or correlation < 0.5:
        fidelity_reasons.append("RANKING_NOT_PRESERVED")
    if median_degradation > 0.10:
        fidelity_reasons.append("MEDIAN_DEGRADATION_EXCEEDS_10_PERCENT")
    if maximum_degradation > 0.20:
        fidelity_reasons.append("PILOT_DEGRADATION_EXCEEDS_20_PERCENT")
    if any(
        item.status is not TeacherStatus.PASS
        or not 0.24 <= item.binary_fraction <= 0.26
        for item in ordered
    ):
        fidelity_reasons.append("PILOT_BINARY_BUDGET_FAILURE")

    pilot_seconds = sum(item.wall_seconds for item in ordered)
    median_32_seconds = float(np.median([item.wall_seconds for item in low]))
    median_64_seconds = float(np.median([item.wall_seconds for item in high]))
    future_seconds = 1.15 * (20 * median_32_seconds + 8 * median_64_seconds)
    projected_hours = (pilot_seconds + future_seconds) / 3600.0
    pilot_bytes = sum(item.artifact_bytes for item in ordered)
    median_32_bytes = float(np.median([item.artifact_bytes for item in low]))
    median_64_bytes = float(np.median([item.artifact_bytes for item in high]))
    projected_bytes = pilot_bytes + 1.15 * (20 * median_32_bytes + 8 * median_64_bytes)
    projected_gib = projected_bytes / 1024**3

    common = {
        "base_spec_sha256": base_spec_sha256,
        "amendment_sha256": amendment_sha256,
        "spearman_rank_correlation": correlation,
        "relative_degradations": degradations,
        "median_relative_degradation": median_degradation,
        "maximum_relative_degradation": maximum_degradation,
        "projected_teacher_hours": projected_hours,
        "projected_artifact_gib": projected_gib,
        "measurements": ordered,
    }
    if fidelity_reasons:
        return TeacherCostAssessment(
            status=TeacherCostStatus.ML_NO_GO_TEACHER_FIDELITY,
            reason_codes=tuple(fidelity_reasons),
            accepted_teacher_resolution=None,
            **common,
        )
    cost_reasons: list[str] = []
    if projected_hours > 8.0:
        cost_reasons.append("TEACHER_WALLCLOCK_EXCEEDS_8_HOURS")
    if projected_gib > 5.0:
        cost_reasons.append("TEACHER_ARTIFACTS_EXCEED_5_GIB")
    if cost_reasons:
        return TeacherCostAssessment(
            status=TeacherCostStatus.ML_NO_GO_TEACHER_COST,
            reason_codes=tuple(cost_reasons),
            accepted_teacher_resolution=None,
            **common,
        )
    return TeacherCostAssessment(
        status=TeacherCostStatus.PASS,
        reason_codes=(),
        accepted_teacher_resolution=32,
        **common,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _measurement_from_artifacts(
    pilot_id: str,
    centers: tuple[tuple[float, float], ...],
    resolution: int,
    directory: Path,
) -> PilotMeasurement:
    result_path = directory / "teacher_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    status = TeacherStatus(payload["status"])
    binary_path = directory / f"design_binary_{resolution}.npy"
    if not binary_path.is_file():
        return PilotMeasurement(
            pilot_id=pilot_id,
            resolution=resolution,
            status=TeacherStatus.INVALID_RUN,
            wall_seconds=float(payload.get("total_wall_seconds", math.nan)),
            verified_peak_64=math.nan,
            binary_fraction=math.nan,
            maximum_scipy_residual=math.inf,
            artifact_bytes=_artifact_bytes(directory),
            result_sha256=_sha256(result_path),
            binary_sha256="",
        )
    binary = np.load(binary_path, allow_pickle=False)
    verification = verify_teacher_at_64(
        centers,
        binary,
        source_resolution=resolution,
    )
    verification_payload = {
        "schema_version": 1,
        "pilot_id": pilot_id,
        "source_resolution": resolution,
        "scenario_peaks": list(verification.scenario_peaks),
        "worst_peak": verification.worst_peak,
        "material_fraction": verification.material_fraction,
        "maximum_residual": verification.maximum_residual,
        "binary_file_sha256": _sha256(binary_path),
    }
    (directory / "verification_64.json").write_text(
        json.dumps(verification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PilotMeasurement(
        pilot_id=pilot_id,
        resolution=resolution,
        status=status,
        wall_seconds=float(payload["total_wall_seconds"]),
        verified_peak_64=verification.worst_peak,
        binary_fraction=verification.material_fraction,
        maximum_scipy_residual=verification.maximum_residual,
        artifact_bytes=_artifact_bytes(directory),
        result_sha256=_sha256(result_path),
        binary_sha256=_sha256(binary_path),
    )


def run_teacher_cost_assessment(
    output_dir: Path,
    *,
    base_spec_path: Path,
    amendment_path: Path,
) -> TeacherCostAssessment:
    """Run or resume exactly six locked pilot optimizations, then assess."""
    measurements: list[PilotMeasurement] = []
    pilot_root = output_dir / "cost_pilots"
    for pilot_id, centers, seed in PILOTS:
        for resolution, iterations in ((32, 200), (64, 600)):
            directory = pilot_root / pilot_id / f"teacher_{resolution}"
            result_path = directory / "teacher_result.json"
            if not result_path.is_file():
                print(
                    json.dumps(
                        {
                            "event": "teacher_start",
                            "pilot_id": pilot_id,
                            "resolution": resolution,
                            "seed": seed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                optimize_teacher(
                    centers,
                    seed=seed,
                    config=TeacherConfig(
                        resolution=resolution,
                        iterations=iterations,
                    ),
                    output_dir=directory,
                )
                print(
                    json.dumps(
                        {
                            "event": "teacher_complete",
                            "pilot_id": pilot_id,
                            "resolution": resolution,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            measurements.append(
                _measurement_from_artifacts(
                    pilot_id,
                    centers,
                    resolution,
                    directory,
                )
            )
    assessment = assess_teacher_cost(
        tuple(measurements),
        base_spec_sha256=_sha256(base_spec_path),
        amendment_sha256=_sha256(amendment_path),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "teacher_cost_report.json.tmp"
    temporary.write_text(
        json.dumps(assessment.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_dir / "teacher_cost_report.json")
    return assessment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ml_warmstart_spike"),
    )
    parser.add_argument(
        "--base-spec",
        type=Path,
        default=Path("docs/superpowers/specs/2026-08-29-ml-warmstart-spike-design.md"),
    )
    parser.add_argument(
        "--amendment",
        type=Path,
        default=Path(
            "docs/superpowers/specs/2026-08-29-ml-warmstart-spike-amendment-1.md"
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    result = run_teacher_cost_assessment(
        arguments.output,
        base_spec_path=arguments.base_spec,
        amendment_path=arguments.amendment,
    )
    print(
        json.dumps(
            {
                "status": result.status.value,
                "projected_teacher_hours": result.projected_teacher_hours,
                "spearman_rank_correlation": result.spearman_rank_correlation,
            },
            sort_keys=True,
        )
    )
    return 2 if result.status is TeacherCostStatus.INVALID_RUN else 0


if __name__ == "__main__":
    raise SystemExit(main())
