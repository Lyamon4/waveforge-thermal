"""Independent unseen-layout verification and paired campaign statistics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.ml.multitask_protocol import PRODUCTION_SEEDS
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.grid import Grid2D
from waveforge.physics.steady_solver import solve_steady
from waveforge.verification.high_fidelity import array_sha256, replicate_design


class MultitaskCampaignStatus(StrEnum):
    """Primary prospective multi-task experiment outcomes."""

    MULTITASK_NCA_GO = "MULTITASK_NCA_GO"
    MULTITASK_NCA_NO_GO_EFFECT = "MULTITASK_NCA_NO_GO_EFFECT"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class BootstrapMedianInterval:
    """Paired nonparametric 95% interval for the median relative gap."""

    median: float
    lower: float
    upper: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class GeneralizationSeedSummary:
    """All registered ID metrics for one frozen production model."""

    seed: int
    valid: bool
    task_count: int
    median_gap: float
    p90_gap: float
    worst_gap: float
    win_count: int
    win_rate: float
    bootstrap_median_lower: float
    bootstrap_median_upper: float
    condition_matched_wins: int
    primary_seed_pass: bool
    better_tested_gradient: bool


@dataclass(frozen=True)
class MultitaskCampaignVerdict:
    """Two-of-three aggregate verdict without lucky-seed replacement."""

    status: MultitaskCampaignStatus
    passing_seed_count: int
    better_tested_gradient_seed_count: int
    seeds: tuple[GeneralizationSeedSummary, ...]


@dataclass(frozen=True)
class IndependentTaskVerification:
    """One strict-binary design verified by CPU SciPy at a fresh resolution."""

    task_id: str
    resolution: int
    design_hash_64: str
    transferred_design_hash: str
    material_fraction: float
    scenario_peaks: tuple[float, float, float]
    worst_peak: float
    maximum_normalized_residual: float
    wall_seconds: float


def relative_gap(*, nca_peak: float, gradient_peak: float) -> float:
    """Return `(NCA-gradient)/gradient`; negative means NCA is better."""
    if (
        not math.isfinite(nca_peak)
        or not math.isfinite(gradient_peak)
        or nca_peak <= 0.0
        or gradient_peak <= 0.0
    ):
        raise ValueError("paired peaks must be finite and positive")
    return (nca_peak - gradient_peak) / gradient_peak


def bootstrap_median_interval(
    gaps: tuple[float, ...],
    *,
    seed: int,
    resamples: int = 10_000,
) -> BootstrapMedianInterval:
    """Bootstrap paired task gaps with a locked NumPy PCG64 stream."""
    values = np.asarray(gaps, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap requires at least two finite paired gaps")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("bootstrap requires at least 100 resamples")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[indices], axis=1)
    return BootstrapMedianInterval(
        median=float(np.median(values)),
        lower=float(np.quantile(medians, 0.025)),
        upper=float(np.quantile(medians, 0.975)),
        resamples=resamples,
        seed=seed,
    )


def summarize_seed(
    *,
    seed: int,
    nca_peaks: tuple[float, ...],
    gradient_peaks: tuple[float, ...],
    bootstrap_seed: int,
    bootstrap_resamples: int,
    condition_matched_wins: int,
    valid: bool,
) -> GeneralizationSeedSummary:
    """Apply the locked median/p90/win/conditioning criteria to one seed."""
    if len(nca_peaks) != len(gradient_peaks) or len(nca_peaks) < 2:
        raise ValueError("seed summary requires equal paired task arrays")
    try:
        gaps = tuple(
            relative_gap(nca_peak=nca, gradient_peak=gradient)
            for nca, gradient in zip(nca_peaks, gradient_peaks, strict=True)
        )
    except ValueError:
        valid = False
        gaps = tuple(math.inf for _ in nca_peaks)
    if valid:
        interval = bootstrap_median_interval(
            gaps,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
        values = np.asarray(gaps, dtype=np.float64)
        median = float(np.median(values))
        p90 = float(np.quantile(values, 0.9))
        worst = float(np.max(values))
        wins = int(np.sum(values < 0.0))
    else:
        interval = BootstrapMedianInterval(
            median=math.inf,
            lower=math.inf,
            upper=math.inf,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        median = p90 = worst = math.inf
        wins = 0
    win_rate = wins / len(nca_peaks)
    primary_pass = bool(
        valid
        and median <= 0.03
        and p90 <= 0.10
        and win_rate >= 0.20
        and condition_matched_wins >= 23
    )
    better = bool(valid and median < 0.0 and interval.upper < 0.0 and win_rate > 0.50)
    return GeneralizationSeedSummary(
        seed=seed,
        valid=valid,
        task_count=len(nca_peaks),
        median_gap=median,
        p90_gap=p90,
        worst_gap=worst,
        win_count=wins,
        win_rate=win_rate,
        bootstrap_median_lower=interval.lower,
        bootstrap_median_upper=interval.upper,
        condition_matched_wins=condition_matched_wins,
        primary_seed_pass=primary_pass,
        better_tested_gradient=better,
    )


def classify_campaign(
    summaries: list[GeneralizationSeedSummary],
) -> MultitaskCampaignVerdict:
    """Require all exact registered seeds and two primary seed passes."""
    if tuple(item.seed for item in summaries) != PRODUCTION_SEEDS:
        raise ValueError("campaign summaries must use exact registered seed order")
    passing = sum(item.primary_seed_pass for item in summaries)
    better = sum(item.better_tested_gradient for item in summaries)
    if any(not item.valid for item in summaries):
        status = MultitaskCampaignStatus.INVALID_RUN
    elif passing >= 2:
        status = MultitaskCampaignStatus.MULTITASK_NCA_GO
    else:
        status = MultitaskCampaignStatus.MULTITASK_NCA_NO_GO_EFFECT
    return MultitaskCampaignVerdict(
        status=status,
        passing_seed_count=passing,
        better_tested_gradient_seed_count=better,
        seeds=tuple(summaries),
    )


def verify_binary_task(
    design_64: NDArray[np.float64],
    task: SourceLayoutTask,
    *,
    resolution: int = 256,
) -> IndependentTaskVerification:
    """Independently rerasterize a task and solve its frozen binary design."""
    design = np.asarray(design_64, dtype=np.float64)
    if design.shape != (64, 64) or not np.isin(design, (0.0, 1.0)).all():
        raise ValueError("independent verification requires a binary 64x64 design")
    if resolution not in (128, 256):
        raise ValueError("independent verification resolution must be 128 or 256")
    factor = resolution // 64
    transferred = replicate_design(design, factor=factor)
    grid = Grid2D(nx=resolution, ny=resolution)
    conductivity = 1.0 + 19.0 * transferred
    peaks: list[float] = []
    residuals: list[float] = []
    started = time.perf_counter()
    for bounds in task.bounds:
        source = area_overlap_rectangular_source(grid, bounds, 1.0)
        result = solve_steady(
            grid,
            conductivity,
            source,
            BoundaryConditions.production(),
        )
        peaks.append(float(np.max(result.temperature)))
        residuals.append(result.normalized_residual)
    return IndependentTaskVerification(
        task_id=task.task_id,
        resolution=resolution,
        design_hash_64=array_sha256(design),
        transferred_design_hash=array_sha256(transferred),
        material_fraction=float(np.mean(transferred)),
        scenario_peaks=(peaks[0], peaks[1], peaks[2]),
        worst_peak=max(peaks),
        maximum_normalized_residual=max(residuals),
        wall_seconds=time.perf_counter() - started,
    )
