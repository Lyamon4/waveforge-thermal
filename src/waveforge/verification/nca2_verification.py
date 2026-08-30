"""Independent SciPy verification and thermal verdict for NCA-2."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waveforge.reproducibility import content_hash
from waveforge.verification.high_fidelity import (
    CandidateVerification,
    array_sha256,
    replicate_design,
    verify_candidate,
)
from waveforge.verification.nca_verification import (
    NCAConnectivityDiagnostic,
    connectivity_diagnostic,
)

TREE_PEAK_256 = 0.1650978093408512
PASS_PEAK_256 = 0.1617958531540342
NONCOLLAPSE_PEAK_256 = 0.1683997655276682
BUDGET_MINIMUM = 0.24
BUDGET_MAXIMUM = 0.26
PRODUCTION_SEEDS = (20260911, 20260912, 20260913)
PREVIOUS_WAVEFORGE_PEAKS = {
    20260828: 0.156506824943584,
    20260829: 0.1574716324313547,
    20260830: 0.15663546358885735,
}


class NCA2VerificationError(RuntimeError):
    """A frozen design or independent transfer violated the locked contract."""


@dataclass(frozen=True)
class NCA2SeedVerdict:
    """Thermal effect classification independent of connectivity diagnostics."""

    seed: int
    peak_256: float
    binary_fraction: float
    numerically_valid: bool
    budget_pass: bool
    primary_pass: bool
    noncollapse_pass: bool
    tree_improvement: float
    reason_codes: tuple[str, ...]

    @classmethod
    def classify(
        cls,
        *,
        seed: int,
        peak_256: float,
        binary_fraction: float,
        numerically_valid: bool,
    ) -> NCA2SeedVerdict:
        """Apply all preregistered inclusive per-seed thresholds."""
        finite = bool(
            numerically_valid
            and math.isfinite(peak_256)
            and peak_256 > 0.0
            and math.isfinite(binary_fraction)
        )
        budget_pass = bool(
            finite and BUDGET_MINIMUM <= binary_fraction <= BUDGET_MAXIMUM
        )
        primary_pass = bool(finite and budget_pass and peak_256 <= PASS_PEAK_256)
        noncollapse_pass = bool(
            finite and budget_pass and peak_256 <= NONCOLLAPSE_PEAK_256
        )
        reasons: list[str] = []
        if not finite:
            reasons.append("NUMERICAL_INVALIDITY")
        if finite and not budget_pass:
            reasons.append("BINARY_MATERIAL_BUDGET_FAILURE")
        if finite and budget_pass and not primary_pass:
            reasons.append("PRIMARY_EFFECT_THRESHOLD_FAILURE")
        if finite and budget_pass and not noncollapse_pass:
            reasons.append("CATASTROPHIC_COLLAPSE")
        improvement = (TREE_PEAK_256 - peak_256) / TREE_PEAK_256 if finite else math.nan
        return cls(
            seed=seed,
            peak_256=float(peak_256),
            binary_fraction=float(binary_fraction),
            numerically_valid=finite,
            budget_pass=budget_pass,
            primary_pass=primary_pass,
            noncollapse_pass=noncollapse_pass,
            tree_improvement=improvement,
            reason_codes=tuple(reasons),
        )


@dataclass(frozen=True)
class NCA2CampaignVerdict:
    """Three-seed primary outcome plus complete descriptive statistics."""

    status: str
    passing_seed_count: int
    required_passing_seed_count: int
    mean_peak_256: float
    median_peak_256: float
    minimum_peak_256: float
    maximum_peak_256: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class NCA2SeedVerification:
    """Dual-grid physics, comparators and separate connectivity diagnostic."""

    seed: int
    verification_128: CandidateVerification
    verification_256: CandidateVerification
    relative_128_to_256_change: float
    connectivity: NCAConnectivityDiagnostic
    engineering_connectivity_pass: bool
    verdict: NCA2SeedVerdict
    previous_waveforge_relative_differences: dict[int, float]


def classify_nca2_campaign(
    seed_verdicts: tuple[NCA2SeedVerdict, ...],
) -> NCA2CampaignVerdict:
    """Apply invalidity, budget, effect and noncollapse precedence."""
    peaks = [verdict.peak_256 for verdict in seed_verdicts]
    finite_peaks = [peak for peak in peaks if math.isfinite(peak)]
    if finite_peaks:
        mean_peak = float(statistics.fmean(finite_peaks))
        median_peak = float(statistics.median(finite_peaks))
        minimum_peak = min(finite_peaks)
        maximum_peak = max(finite_peaks)
    else:
        mean_peak = median_peak = minimum_peak = maximum_peak = math.nan
    passing = sum(verdict.primary_pass for verdict in seed_verdicts)
    reasons: list[str] = []
    exact_registry = (
        tuple(verdict.seed for verdict in seed_verdicts) == PRODUCTION_SEEDS
    )
    if not exact_registry or any(
        not verdict.numerically_valid for verdict in seed_verdicts
    ):
        status = "NCA2_INVALID_RUN"
        reasons.append("INCOMPLETE_OR_NUMERICALLY_INVALID_REGISTRY")
    elif any(not verdict.budget_pass for verdict in seed_verdicts):
        status = "NCA2_NO_GO_EFFECT"
        reasons.append("BINARY_MATERIAL_BUDGET_FAILURE")
    elif any(not verdict.noncollapse_pass for verdict in seed_verdicts):
        status = "NCA2_NO_GO_EFFECT"
        reasons.append("CATASTROPHIC_COLLAPSE")
    elif passing < 2:
        status = "NCA2_NO_GO_EFFECT"
        reasons.append("FEWER_THAN_TWO_SEEDS_PASS_TREE_MARGIN")
    else:
        status = "NCA2_STABILITY_GO"
    return NCA2CampaignVerdict(
        status=status,
        passing_seed_count=passing,
        required_passing_seed_count=2,
        mean_peak_256=mean_peak,
        median_peak_256=median_peak,
        minimum_peak_256=minimum_peak,
        maximum_peak_256=maximum_peak,
        reason_codes=tuple(reasons),
    )


def verify_nca2_seed(
    *,
    seed: int,
    binary_design: NDArray[np.float64],
    continuous_design: NDArray[np.float64],
    expected_binary_content_hash: str,
    expected_continuous_content_hash: str,
    numerically_valid: bool = True,
) -> NCA2SeedVerification:
    """Verify one frozen seed with independent 128 and primary 256 SciPy solves."""
    if seed not in PRODUCTION_SEEDS:
        raise NCA2VerificationError(f"unregistered production seed: {seed}")
    raw_binary = np.asarray(binary_design)
    raw_continuous = np.asarray(continuous_design)
    if content_hash(raw_binary) != expected_binary_content_hash:
        raise NCA2VerificationError("binary production content hash mismatch")
    if content_hash(raw_continuous) != expected_continuous_content_hash:
        raise NCA2VerificationError("continuous production content hash mismatch")
    binary = np.asarray(raw_binary, dtype=np.float64)
    continuous = np.asarray(raw_continuous, dtype=np.float64)
    if binary.shape != (64, 64) or continuous.shape != (64, 64):
        raise NCA2VerificationError("frozen designs must be 64x64")
    if not np.isfinite(binary).all() or not np.isfinite(continuous).all():
        raise NCA2VerificationError("frozen designs must be finite")
    if not np.all((binary == 0.0) | (binary == 1.0)):
        raise NCA2VerificationError("binary design must be strict 0/1")
    if not np.array_equal(binary, (continuous >= 0.5).astype(np.float64)):
        raise NCA2VerificationError("binary design differs from strict D >= 0.5")
    design_hash = array_sha256(binary)
    verification_128 = verify_candidate(
        f"nca2_{seed}",
        binary,
        fidelity="reference_128",
        expected_design_hash=design_hash,
    )
    verification_256 = verify_candidate(
        f"nca2_{seed}",
        binary,
        fidelity="reference_256",
        expected_design_hash=design_hash,
    )
    expected_128 = array_sha256(replicate_design(binary, factor=2))
    expected_256 = array_sha256(replicate_design(binary, factor=4))
    if (
        verification_128.transferred_design_hash != expected_128
        or verification_256.transferred_design_hash != expected_256
    ):
        raise NCA2VerificationError("verification transfer is not exact replication")
    connectivity = connectivity_diagnostic(binary)
    engineering_pass = all(
        connectivity.sink_component_source_intersections.get(scenario, False)
        for scenario in ("A", "B", "C")
    )
    peak_256 = verification_256.worst_peak
    return NCA2SeedVerification(
        seed=seed,
        verification_128=verification_128,
        verification_256=verification_256,
        relative_128_to_256_change=(verification_128.worst_peak - peak_256)
        / max(abs(peak_256), 1.0e-12),
        connectivity=connectivity,
        engineering_connectivity_pass=engineering_pass,
        verdict=NCA2SeedVerdict.classify(
            seed=seed,
            peak_256=peak_256,
            binary_fraction=float(binary.mean()),
            numerically_valid=numerically_valid,
        ),
        previous_waveforge_relative_differences={
            previous_seed: (previous_peak - peak_256) / previous_peak
            for previous_seed, previous_peak in PREVIOUS_WAVEFORGE_PEAKS.items()
        },
    )
