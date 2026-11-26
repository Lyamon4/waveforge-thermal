"""Machine-readable Gate 2A verdict contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Gate2Status(StrEnum):
    """Mutually exclusive scientific and numerical campaign outcomes."""

    PASS = "PASS"
    NO_GO_EFFECT = "NO_GO_EFFECT"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class SeedVerdict:
    """Verdict for one registered production seed."""

    status: Gate2Status
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""


@dataclass(frozen=True)
class CampaignVerdict:
    """Aggregate verdict with numerical-invalidity precedence."""

    status: Gate2Status
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""


@dataclass(frozen=True)
class BaselineSelection:
    """Strongest verified member of one locked comparison set."""

    candidate_id: str
    worst_peak: float


def select_strongest_baseline(
    baseline_peaks: dict[str, float],
) -> BaselineSelection:
    """Select the lowest finite positive unrounded verified peak."""
    if not baseline_peaks:
        raise ValueError("baseline registry must not be empty")
    if any(
        not candidate_id or not math.isfinite(peak) or peak <= 0.0
        for candidate_id, peak in baseline_peaks.items()
    ):
        raise ValueError("baseline peaks must be finite and positive")
    candidate_id, peak = min(
        baseline_peaks.items(),
        key=lambda item: (item[1], item[0]),
    )
    return BaselineSelection(candidate_id=candidate_id, worst_peak=peak)


def classify_nominal_seed(
    *,
    candidate_peak: float,
    baseline_peaks: dict[str, float],
    continuous_fraction: float,
    binary_fraction: float,
    valid: bool,
    material_target: float = 0.25,
    material_tolerance: float = 0.01,
    effect_threshold: float = 0.05,
) -> SeedVerdict:
    """Apply locked nominal budget/effect rules with invalidity precedence."""
    numerical_values = (
        candidate_peak,
        continuous_fraction,
        binary_fraction,
        material_target,
        material_tolerance,
        effect_threshold,
    )
    if (
        not valid
        or not all(math.isfinite(value) for value in numerical_values)
        or candidate_peak <= 0.0
        or material_tolerance < 0.0
        or effect_threshold < 0.0
    ):
        return SeedVerdict(
            status=Gate2Status.INVALID_RUN,
            reason_codes=("NOMINAL_NUMERICAL_OR_REGISTRY_FAILURE",),
        )
    try:
        strongest = select_strongest_baseline(baseline_peaks)
    except ValueError:
        return SeedVerdict(
            status=Gate2Status.INVALID_RUN,
            reason_codes=("NOMINAL_NUMERICAL_OR_REGISTRY_FAILURE",),
        )

    relative_improvement = (
        strongest.worst_peak - candidate_peak
    ) / strongest.worst_peak
    lower = material_target - material_tolerance
    upper = material_target + material_tolerance
    budget_pass = (
        lower <= continuous_fraction <= upper and lower <= binary_fraction <= upper
    )
    effect_pass = relative_improvement >= effect_threshold
    metrics = {
        "candidate_peak": candidate_peak,
        "strongest_baseline_id": strongest.candidate_id,
        "strongest_baseline_peak": strongest.worst_peak,
        "relative_improvement": relative_improvement,
        "effect_threshold": effect_threshold,
        "continuous_material_fraction": continuous_fraction,
        "binary_material_fraction": binary_fraction,
        "material_target": material_target,
        "material_tolerance": material_tolerance,
        "budget_pass": budget_pass,
        "effect_pass": effect_pass,
    }
    reasons: list[str] = []
    if not budget_pass:
        reasons.append("MATERIAL_BUDGET_FAILURE")
    if not effect_pass:
        reasons.append("NOMINAL_EFFECT_BELOW_THRESHOLD")
    if reasons:
        return SeedVerdict(
            status=Gate2Status.NO_GO_EFFECT,
            reason_codes=tuple(reasons),
            metrics=metrics,
        )
    return SeedVerdict(status=Gate2Status.PASS, metrics=metrics)


def classify_campaign(
    *,
    valid: bool,
    passing_seed_count: int,
    required: int,
    metrics: dict[str, Any] | None = None,
    config_hash: str = "",
) -> CampaignVerdict:
    """Classify the campaign without confusing invalidity with no effect."""
    if passing_seed_count < 0 or required < 1:
        raise ValueError("seed counts must be non-negative and required positive")
    shared = {"passing_seed_count": passing_seed_count, "required": required}
    shared.update(metrics or {})
    if not valid:
        return CampaignVerdict(
            status=Gate2Status.INVALID_RUN,
            reason_codes=("NUMERICAL_OR_INTEGRITY_FAILURE",),
            metrics=shared,
            config_hash=config_hash,
        )
    if passing_seed_count < required:
        return CampaignVerdict(
            status=Gate2Status.NO_GO_EFFECT,
            reason_codes=("INSUFFICIENT_PASSING_SEEDS",),
            metrics=shared,
            config_hash=config_hash,
        )
    return CampaignVerdict(
        status=Gate2Status.PASS,
        metrics=shared,
        config_hash=config_hash,
    )
