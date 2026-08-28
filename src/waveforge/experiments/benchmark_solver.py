"""Warm/cold benchmark for validated steady and transient physics."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.conductivity import interpolate_conductivity
from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import normalized_rectangular_source
from waveforge.physics.steady_solver import (
    AssembledSystem,
    assemble_steady_system,
    factorize_system,
)
from waveforge.physics.transient_solver import (
    PreparedTransientSystem,
    TransientConfig,
    assemble_transient_system,
    factorize_transient_system,
    prepare_transient_system,
    solve_transient_prepared,
)
from waveforge.reproducibility import content_hash


@dataclass(frozen=True)
class TimingSummary:
    """Summary statistics for one measured phase."""

    runs: int
    mean: float
    median: float
    p90: float
    std: float


@dataclass(frozen=True)
class BenchmarkRecord:
    """One benchmark phase with all registered summary statistics."""

    solver: str
    resolution: int
    time_steps: int
    scenarios: int
    mode: str
    phase: str
    runs: int
    mean: float
    median: float
    p90: float
    std: float
    conductivity_family_hash: str
    unit: str = "seconds"


def summarize_timings(samples: list[float]) -> TimingSummary:
    """Вычислить mean, median, p90 и sample standard deviation."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timing samples must be a non-empty one-dimensional list")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("timing samples must be finite and non-negative")
    return TimingSummary(
        runs=int(values.size),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p90=float(np.quantile(values, 0.9)),
        std=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    )


def generate_conductivity_maps(
    grid: Grid2D,
    *,
    count: int,
    seed: int,
) -> tuple[NDArray[np.float64], ...]:
    """Precompute distinct deterministic maps outside timed regions."""
    if count <= 0:
        raise ValueError("count must be positive")
    x, y = grid.mesh
    seed_phase = (seed % 997) / 997.0
    maps: list[NDArray[np.float64]] = []
    for index in range(count):
        phase = 2.0 * np.pi * (seed_phase + index * 0.17320508075688773)
        design = 0.5 + 0.3 * np.sin(2.0 * np.pi * x + phase) * np.cos(
            2.0 * np.pi * y - 0.7 * phase
        )
        maps.append(interpolate_conductivity(np.clip(design, 0.0, 1.0)))
    return tuple(maps)


def _scenario_sources(
    grid: Grid2D,
    scenario_count: int,
) -> tuple[NDArray[np.float64], ...]:
    if scenario_count <= 0:
        raise ValueError("scenario_count must be positive")
    centers = ((0.50, 0.72), (0.28, 0.72), (0.72, 0.72))
    sources: list[NDArray[np.float64]] = []
    for index in range(scenario_count):
        center_x, center_y = centers[index % len(centers)]
        sources.append(
            normalized_rectangular_source(
                grid,
                max(0.0, center_x - 0.1),
                min(1.0, center_x + 0.1),
                max(0.0, center_y - 0.1),
                min(1.0, center_y + 0.1),
            )
        )
    return tuple(sources)


def _seconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) * 1e-9


def _conductivity_family_hash(
    conductivity_maps: tuple[NDArray[np.float64], ...],
) -> str:
    member_hashes = "|".join(content_hash(array) for array in conductivity_maps)
    return hashlib.sha256(member_hashes.encode()).hexdigest()


def _record(
    *,
    solver: str,
    resolution: int,
    time_steps: int,
    scenarios: int,
    mode: str,
    phase: str,
    samples: list[float],
    conductivity_family_hash: str,
) -> BenchmarkRecord:
    summary = summarize_timings(samples)
    return BenchmarkRecord(
        solver=solver,
        resolution=resolution,
        time_steps=time_steps,
        scenarios=scenarios,
        mode=mode,
        phase=phase,
        conductivity_family_hash=conductivity_family_hash,
        **asdict(summary),
    )


def _solve_steady_scenarios(
    system: AssembledSystem,
    factorization: object,
    sources: tuple[NDArray[np.float64], ...],
) -> float:
    worst_peak = -np.inf
    for source in sources:
        rhs = source.ravel() + system.dirichlet_rhs
        temperature = factorization.solve(rhs)
        worst_peak = max(worst_peak, float(np.max(temperature)))
    return worst_peak


def benchmark_steady_case(
    *,
    resolution: int,
    warmup_runs: int,
    measured_runs: int,
    scenario_count: int,
    seed: int,
) -> tuple[BenchmarkRecord, ...]:
    """Measure warm reused solve and cold changing-design evaluation."""
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    grid = Grid2D(resolution, resolution)
    bcs = BoundaryConditions.production()
    sources = _scenario_sources(grid, scenario_count)
    conductivity_maps = generate_conductivity_maps(
        grid, count=warmup_runs + measured_runs + 1, seed=seed
    )
    family_hash = _conductivity_family_hash(conductivity_maps)

    warm_system = assemble_steady_system(grid, conductivity_maps[0], sources[0], bcs)
    warm_factorization = factorize_system(warm_system)
    warm_samples: list[float] = []
    for run_index in range(warmup_runs + measured_runs):
        start = perf_counter_ns()
        peak = _solve_steady_scenarios(warm_system, warm_factorization, sources)
        end = perf_counter_ns()
        if not np.isfinite(peak):
            raise FloatingPointError("steady benchmark produced non-finite peak")
        if run_index >= warmup_runs:
            warm_samples.append(_seconds(start, end))

    cold_samples: dict[str, list[float]] = {
        "assembly": [],
        "factorization": [],
        "solve": [],
        "total_evaluation": [],
    }
    for run_index, conductivity in enumerate(conductivity_maps[1:]):
        total_start = perf_counter_ns()
        system = assemble_steady_system(grid, conductivity, sources[0], bcs)
        assembly_end = perf_counter_ns()
        factorization = factorize_system(system)
        factorization_end = perf_counter_ns()
        peak = _solve_steady_scenarios(system, factorization, sources)
        solve_end = perf_counter_ns()
        if not np.isfinite(peak):
            raise FloatingPointError("steady benchmark produced non-finite peak")
        if run_index >= warmup_runs:
            cold_samples["assembly"].append(_seconds(total_start, assembly_end))
            cold_samples["factorization"].append(
                _seconds(assembly_end, factorization_end)
            )
            cold_samples["solve"].append(_seconds(factorization_end, solve_end))
            cold_samples["total_evaluation"].append(_seconds(total_start, solve_end))

    records = [
        _record(
            solver="steady",
            resolution=resolution,
            time_steps=0,
            scenarios=scenario_count,
            mode="warm_reused",
            phase="solve",
            samples=warm_samples,
            conductivity_family_hash=family_hash,
        )
    ]
    records.extend(
        _record(
            solver="steady",
            resolution=resolution,
            time_steps=0,
            scenarios=scenario_count,
            mode="cold_design",
            phase=phase,
            samples=samples,
            conductivity_family_hash=family_hash,
        )
        for phase, samples in cold_samples.items()
    )
    return tuple(records)


def _solve_transient_scenarios(
    prepared: PreparedTransientSystem,
    sources: tuple[NDArray[np.float64], ...],
    initial: NDArray[np.float64],
) -> float:
    worst_peak = -np.inf
    for source in sources:
        result = solve_transient_prepared(prepared, source, initial)
        worst_peak = max(worst_peak, float(np.max(result.temperatures[-1])))
    return worst_peak


def benchmark_transient_case(
    *,
    resolution: int,
    time_steps: int,
    warmup_runs: int,
    measured_runs: int,
    scenario_count: int,
    seed: int,
) -> tuple[BenchmarkRecord, ...]:
    """Measure reused and changing-design implicit transient trajectories."""
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measured_runs positive")
    grid = Grid2D(resolution, resolution)
    bcs = BoundaryConditions.production()
    sources = _scenario_sources(grid, scenario_count)
    initial = np.zeros(grid.shape, dtype=np.float64)
    config = TransientConfig(
        dt=0.005,
        n_steps=time_steps,
        rho_c=1.0,
        store_every=time_steps,
    )
    conductivity_maps = generate_conductivity_maps(
        grid, count=warmup_runs + measured_runs + 1, seed=seed
    )
    family_hash = _conductivity_family_hash(conductivity_maps)

    warm_prepared = prepare_transient_system(grid, conductivity_maps[0], bcs, config)
    warm_samples: list[float] = []
    for run_index in range(warmup_runs + measured_runs):
        start = perf_counter_ns()
        peak = _solve_transient_scenarios(warm_prepared, sources, initial)
        end = perf_counter_ns()
        if not np.isfinite(peak):
            raise FloatingPointError("transient benchmark produced non-finite peak")
        if run_index >= warmup_runs:
            warm_samples.append(_seconds(start, end))

    cold_samples: dict[str, list[float]] = {
        "assembly": [],
        "factorization": [],
        "trajectory": [],
        "total_evaluation": [],
    }
    for run_index, conductivity in enumerate(conductivity_maps[1:]):
        total_start = perf_counter_ns()
        linear_system = assemble_transient_system(grid, conductivity, bcs, config)
        assembly_end = perf_counter_ns()
        prepared = factorize_transient_system(linear_system)
        factorization_end = perf_counter_ns()
        peak = _solve_transient_scenarios(prepared, sources, initial)
        trajectory_end = perf_counter_ns()
        if not np.isfinite(peak):
            raise FloatingPointError("transient benchmark produced non-finite peak")
        if run_index >= warmup_runs:
            cold_samples["assembly"].append(_seconds(total_start, assembly_end))
            cold_samples["factorization"].append(
                _seconds(assembly_end, factorization_end)
            )
            cold_samples["trajectory"].append(
                _seconds(factorization_end, trajectory_end)
            )
            cold_samples["total_evaluation"].append(
                _seconds(total_start, trajectory_end)
            )

    records = [
        _record(
            solver="transient",
            resolution=resolution,
            time_steps=time_steps,
            scenarios=scenario_count,
            mode="warm_reused",
            phase="trajectory",
            samples=warm_samples,
            conductivity_family_hash=family_hash,
        )
    ]
    records.extend(
        _record(
            solver="transient",
            resolution=resolution,
            time_steps=time_steps,
            scenarios=scenario_count,
            mode="cold_design",
            phase=phase,
            samples=samples,
            conductivity_family_hash=family_hash,
        )
        for phase, samples in cold_samples.items()
    )
    return tuple(records)


def run_benchmark(
    *,
    warmup_runs: int,
    measured_runs: int,
    seed: int = 20260828,
) -> tuple[BenchmarkRecord, ...]:
    """Run complete pre-registered steady/transient benchmark matrix."""
    records: list[BenchmarkRecord] = []
    for resolution in (32, 64, 128, 256):
        print(f"Benchmark steady {resolution}x{resolution}", flush=True)
        records.extend(
            benchmark_steady_case(
                resolution=resolution,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                scenario_count=3,
                seed=seed,
            )
        )
    for resolution, time_steps in ((64, 100), (128, 100), (128, 300)):
        print(
            f"Benchmark transient {resolution}x{resolution}, {time_steps} steps",
            flush=True,
        )
        records.extend(
            benchmark_transient_case(
                resolution=resolution,
                time_steps=time_steps,
                warmup_runs=warmup_runs,
                measured_runs=measured_runs,
                scenario_count=3,
                seed=seed,
            )
        )
    return tuple(records)


def write_benchmark_csv(
    records: tuple[BenchmarkRecord, ...],
    output_path: Path,
) -> Path:
    """Write results only after every timed region has completed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(asdict(record) for record in records)
    frame.to_csv(output_path, index=False, float_format="%.17g")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark WaveForge reference solvers"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/solver_benchmark.csv")
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260828)
    arguments = parser.parse_args()
    records = run_benchmark(
        warmup_runs=arguments.warmups,
        measured_runs=arguments.runs,
        seed=arguments.seed,
    )
    write_benchmark_csv(records, arguments.output)
    print(f"Saved {len(records)} benchmark rows to {arguments.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
