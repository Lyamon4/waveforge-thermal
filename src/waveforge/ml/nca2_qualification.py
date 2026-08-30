"""Pure multi-seed stability classification and NCA-2 protocol selection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Literal

ProtocolId = Literal["A", "B"]
DEVELOPMENT_SEEDS = (20260901, 20260902, 20260903)


@dataclass(frozen=True)
class DevelopmentSeedMetrics:
    protocol_id: ProtocolId
    seed: int
    checkpoint_peaks: tuple[float, float, float, float, float]
    final_binary_fraction: float
    numerically_valid: bool
    engineering_connectivity_pass: bool
    late_best_tmax: float
    late_degradation: float
    stable: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ProtocolQualification:
    protocol_id: ProtocolId
    seeds: tuple[DevelopmentSeedMetrics, ...]
    eligible: bool
    stable_seed_count: int
    median_late_degradation: float
    median_final_tmax: float
    worst_final_tmax: float
    selected: bool = False


@dataclass(frozen=True)
class NCA2QualificationVerdict:
    status: str
    production_authorized: bool
    selected_protocol: ProtocolId | None
    selection_reason: str
    protocols: tuple[ProtocolQualification, ProtocolQualification]


def _validate_protocol_id(protocol_id: str) -> ProtocolId:
    if protocol_id not in ("A", "B"):
        raise ValueError(f"unregistered NCA-2 protocol: {protocol_id}")
    return protocol_id


def classify_development_seed(
    *,
    protocol_id: str,
    seed: int,
    checkpoint_peaks: tuple[float, float, float, float, float],
    binary_fraction: float,
    numerically_valid: bool,
    connectivity_pass: bool,
) -> DevelopmentSeedMetrics:
    """Classify stability without giving connectivity selection authority."""
    locked_protocol = _validate_protocol_id(protocol_id)
    if seed not in DEVELOPMENT_SEEDS:
        raise ValueError(f"unregistered development seed: {seed}")
    if len(checkpoint_peaks) != 5:
        raise ValueError("exactly five qualification checkpoint peaks are required")
    peaks = tuple(float(value) for value in checkpoint_peaks)
    peaks_finite = all(math.isfinite(value) and value > 0.0 for value in peaks)
    fraction_finite = math.isfinite(binary_fraction)
    valid = bool(numerically_valid and peaks_finite and fraction_finite)
    late_best = min(peaks) if peaks_finite else math.inf
    late_degradation = (
        max(0.0, (peaks[-1] - late_best) / late_best) if peaks_finite else math.inf
    )
    reasons: list[str] = []
    if not valid:
        reasons.append("NUMERICAL_INVALIDITY")
    if not fraction_finite or not 0.24 <= binary_fraction <= 0.26:
        reasons.append("BINARY_BUDGET_FAILURE")
    if not math.isfinite(late_degradation) or late_degradation > 0.05:
        reasons.append("LATE_DEGRADATION_FAILURE")
    return DevelopmentSeedMetrics(
        protocol_id=locked_protocol,
        seed=seed,
        checkpoint_peaks=peaks,
        final_binary_fraction=float(binary_fraction),
        numerically_valid=valid,
        engineering_connectivity_pass=bool(connectivity_pass),
        late_best_tmax=late_best,
        late_degradation=late_degradation,
        stable=not reasons,
        reason_codes=tuple(reasons),
    )


def summarize_protocol(
    protocol_id: str,
    seeds: tuple[DevelopmentSeedMetrics, ...],
) -> ProtocolQualification:
    """Aggregate the three registered development seeds."""
    locked_protocol = _validate_protocol_id(protocol_id)
    exact_registry = tuple(metric.seed for metric in seeds) == DEVELOPMENT_SEEDS
    matching_protocol = all(metric.protocol_id == locked_protocol for metric in seeds)
    eligible = bool(
        exact_registry
        and matching_protocol
        and all(metric.numerically_valid for metric in seeds)
    )
    if not seeds:
        return ProtocolQualification(
            protocol_id=locked_protocol,
            seeds=seeds,
            eligible=False,
            stable_seed_count=0,
            median_late_degradation=math.inf,
            median_final_tmax=math.inf,
            worst_final_tmax=math.inf,
        )
    degradations = [metric.late_degradation for metric in seeds]
    final_peaks = [metric.checkpoint_peaks[-1] for metric in seeds]
    return ProtocolQualification(
        protocol_id=locked_protocol,
        seeds=seeds,
        eligible=eligible,
        stable_seed_count=sum(metric.stable for metric in seeds),
        median_late_degradation=float(statistics.median(degradations)),
        median_final_tmax=float(statistics.median(final_peaks)),
        worst_final_tmax=max(final_peaks),
    )


def _selected_verdict(
    protocol_a: ProtocolQualification,
    protocol_b: ProtocolQualification,
    *,
    selected: ProtocolQualification,
    reason: str,
) -> NCA2QualificationVerdict:
    protocols = (
        replace(protocol_a, selected=selected.protocol_id == "A"),
        replace(protocol_b, selected=selected.protocol_id == "B"),
    )
    return NCA2QualificationVerdict(
        status="PASS",
        production_authorized=True,
        selected_protocol=selected.protocol_id,
        selection_reason=reason,
        protocols=protocols,
    )


def select_nca2_protocol(
    protocol_a: ProtocolQualification,
    protocol_b: ProtocolQualification,
) -> NCA2QualificationVerdict:
    """Apply the locked stability-first lexicographic ranking."""
    if protocol_a.protocol_id != "A" or protocol_b.protocol_id != "B":
        raise ValueError("qualification protocols must be supplied as A then B")
    selectable = [
        protocol
        for protocol in (protocol_a, protocol_b)
        if protocol.eligible and protocol.stable_seed_count >= 2
    ]
    if not selectable:
        return NCA2QualificationVerdict(
            status="NCA2_QUALIFICATION_FAIL",
            production_authorized=False,
            selected_protocol=None,
            selection_reason="NO_PROTOCOL_WITH_TWO_STABLE_DEVELOPMENT_SEEDS",
            protocols=(protocol_a, protocol_b),
        )
    if len(selectable) == 1:
        return _selected_verdict(
            protocol_a,
            protocol_b,
            selected=selectable[0],
            reason="ONLY_ELIGIBLE_PROTOCOL",
        )

    if protocol_a.stable_seed_count != protocol_b.stable_seed_count:
        selected = max(selectable, key=lambda protocol: protocol.stable_seed_count)
        return _selected_verdict(
            protocol_a,
            protocol_b,
            selected=selected,
            reason="MORE_STABLE_DEVELOPMENT_SEEDS",
        )

    degradation_delta = abs(
        protocol_a.median_late_degradation - protocol_b.median_late_degradation
    )
    if degradation_delta > 1.0e-3:
        selected = min(
            selectable,
            key=lambda protocol: protocol.median_late_degradation,
        )
        return _selected_verdict(
            protocol_a,
            protocol_b,
            selected=selected,
            reason="LOWER_MEDIAN_LATE_DEGRADATION",
        )

    median_best = min(protocol.median_final_tmax for protocol in selectable)
    median_tolerance = 1.0e-3 * max(abs(median_best), 1.0e-12)
    if abs(protocol_a.median_final_tmax - protocol_b.median_final_tmax) > (
        median_tolerance
    ):
        selected = min(selectable, key=lambda protocol: protocol.median_final_tmax)
        return _selected_verdict(
            protocol_a,
            protocol_b,
            selected=selected,
            reason="LOWER_MEDIAN_FINAL_TMAX",
        )

    worst_best = min(protocol.worst_final_tmax for protocol in selectable)
    worst_tolerance = 1.0e-3 * max(abs(worst_best), 1.0e-12)
    if abs(protocol_a.worst_final_tmax - protocol_b.worst_final_tmax) > (
        worst_tolerance
    ):
        selected = min(selectable, key=lambda protocol: protocol.worst_final_tmax)
        return _selected_verdict(
            protocol_a,
            protocol_b,
            selected=selected,
            reason="LOWER_WORST_FINAL_TMAX",
        )

    return _selected_verdict(
        protocol_a,
        protocol_b,
        selected=protocol_b,
        reason="PRACTICAL_TIE_FAVORS_DECAY",
    )
