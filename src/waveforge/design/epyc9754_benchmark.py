"""Prospective EPYC 9754-scale synthetic extreme-OOD benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D

BENCHMARK_LABEL = "EPYC_9754_SCALE_SYNTHETIC"
PACKAGE_WIDTH_MM = 75.4
PACKAGE_HEIGHT_MM = 72.0


@dataclass(frozen=True)
class SyntheticRegion:
    """One disclosed model region, not proprietary AMD die geometry."""

    region_id: str
    center_mm: tuple[float, float]
    size_mm: tuple[float, float]
    normalized_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class SyntheticWorkload:
    """One synthetic power allocation constrained to the 360 W envelope."""

    workload_id: str
    region_powers_watts: tuple[float, ...]


@dataclass(frozen=True)
class EPYC9754ScaleBenchmark:
    """Three robust scenarios for frozen-model secondary evaluation only."""

    task_id: str
    label: str
    benchmark_role: str
    package_size_mm: tuple[float, float]
    regions: tuple[SyntheticRegion, ...]
    workloads: tuple[SyntheticWorkload, ...]
    sources: NDArray[np.float64]
    may_select_checkpoint: bool
    may_update_weights: bool


def _normalized_bounds(
    center_mm: tuple[float, float],
    size_mm: tuple[float, float],
) -> tuple[float, float, float, float]:
    x, y = center_mm
    width, height = size_mm
    return (
        (x - 0.5 * width) / PACKAGE_WIDTH_MM,
        (x + 0.5 * width) / PACKAGE_WIDTH_MM,
        (y - 0.5 * height) / PACKAGE_HEIGHT_MM,
        (y + 0.5 * height) / PACKAGE_HEIGHT_MM,
    )


def _region(
    region_id: str,
    center_mm: tuple[float, float],
    size_mm: tuple[float, float],
) -> SyntheticRegion:
    return SyntheticRegion(
        region_id=region_id,
        center_mm=center_mm,
        size_mm=size_mm,
        normalized_bounds=_normalized_bounds(center_mm, size_mm),
    )


def _regions() -> tuple[SyntheticRegion, ...]:
    x_centers = (11.0, 28.8, 46.6, 64.4)
    ccd_size = (10.0, 7.27)
    top = tuple(
        _region(f"CCD_{index + 1}", (x, 53.0), ccd_size)
        for index, x in enumerate(x_centers)
    )
    bottom = tuple(
        _region(f"CCD_{index + 5}", (x, 19.0), ccd_size)
        for index, x in enumerate(x_centers)
    )
    io_die = _region("IO_DIE", (PACKAGE_WIDTH_MM / 2.0, 36.0), (25.0, 16.0))
    return top + bottom + (io_die,)


def _workloads() -> tuple[SyntheticWorkload, ...]:
    return (
        SyntheticWorkload(
            "uniform_compute",
            (35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 35.0, 80.0),
        ),
        SyntheticWorkload(
            "localized_compute",
            (55.0, 50.0, 25.0, 20.0, 60.0, 55.0, 25.0, 20.0, 50.0),
        ),
        SyntheticWorkload(
            "asymmetric_rows",
            (45.0, 45.0, 45.0, 45.0, 20.0, 20.0, 20.0, 20.0, 100.0),
        ),
    )


def _task_id(
    regions: tuple[SyntheticRegion, ...],
    workloads: tuple[SyntheticWorkload, ...],
    sources: NDArray[np.float64],
) -> str:
    metadata = {
        "label": BENCHMARK_LABEL,
        "package_size_mm": [PACKAGE_WIDTH_MM, PACKAGE_HEIGHT_MM],
        "regions": [
            {
                "region_id": region.region_id,
                "center_mm": list(region.center_mm),
                "size_mm": list(region.size_mm),
            }
            for region in regions
        ],
        "workloads": [
            {
                "workload_id": workload.workload_id,
                "region_powers_watts": list(workload.region_powers_watts),
            }
            for workload in workloads
        ],
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(sources).tobytes())
    return digest.hexdigest()


def build_epyc9754_scale_benchmark(
    *,
    resolution: int,
) -> EPYC9754ScaleBenchmark:
    """Build three independently rasterized 360 W synthetic workload maps."""
    if resolution not in (64, 128, 256):
        raise ValueError("EPYC-scale benchmark resolution must be 64, 128, or 256")
    regions = _regions()
    workloads = _workloads()
    grid = Grid2D(nx=resolution, ny=resolution)
    source_maps: list[NDArray[np.float64]] = []
    for workload in workloads:
        source = np.zeros(grid.shape, dtype=np.float64)
        for region, power in zip(regions, workload.region_powers_watts, strict=True):
            source += area_overlap_rectangular_source(
                grid,
                region.normalized_bounds,
                power,
            )
        source_maps.append(source)
    sources = np.stack(source_maps).astype(np.float64, copy=False)
    return EPYC9754ScaleBenchmark(
        task_id=_task_id(regions, workloads, sources),
        label=BENCHMARK_LABEL,
        benchmark_role="secondary_extreme_ood",
        package_size_mm=(PACKAGE_WIDTH_MM, PACKAGE_HEIGHT_MM),
        regions=regions,
        workloads=workloads,
        sources=sources,
        may_select_checkpoint=False,
        may_update_weights=False,
    )
