"""Independent numerical and physical validation metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from waveforge.physics.boundary_conditions import BoundaryConditions
from waveforge.physics.conductivity import harmonic_mean, validate_conductivity
from waveforge.physics.grid import Grid2D
from waveforge.physics.manufactured_solutions import (
    normalized_rectangular_source,
    sine_manufactured_fixture,
    two_layer_fixture,
)
from waveforge.physics.steady_solver import assemble_steady_system, solve_steady
from waveforge.physics.transient_solver import TransientConfig, solve_transient
from waveforge.reproducibility import content_hash, set_deterministic_seed


def relative_l2(
    predicted: NDArray[np.float64],
    exact: NDArray[np.float64],
) -> float:
    """Вычислить `||predicted-exact|| / max(||exact||, 1e-12)`."""
    predicted_array = np.asarray(predicted, dtype=np.float64)
    exact_array = np.asarray(exact, dtype=np.float64)
    if predicted_array.shape != exact_array.shape:
        raise ValueError("predicted and exact fields must have identical shapes")
    if not np.all(np.isfinite(predicted_array)) or not np.all(np.isfinite(exact_array)):
        raise ValueError("relative L2 inputs must be finite")
    return float(
        np.linalg.norm(predicted_array - exact_array)
        / max(np.linalg.norm(exact_array), 1e-12)
    )


def symmetry_defect(field: NDArray[np.float64]) -> float:
    """Вычислить normalized left-right symmetry defect."""
    field_array = np.asarray(field, dtype=np.float64)
    if field_array.ndim != 2 or not np.all(np.isfinite(field_array)):
        raise ValueError("symmetry field must be a finite two-dimensional array")
    return float(
        np.max(np.abs(field_array - np.flip(field_array, axis=1)))
        / max(np.max(np.abs(field_array)), 1e-12)
    )


def two_layer_interface_flux(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    temperature: NDArray[np.float64],
    *,
    harmonic_epsilon: float = 1e-12,
) -> NDArray[np.float64]:
    """Вычислить magnitude discrete flux на aligned interface `x=0.5`."""
    if grid.nx % 2 != 0:
        raise ValueError("two-layer interface requires an even nx")
    conductivity_array = np.asarray(conductivity, dtype=np.float64)
    temperature_array = np.asarray(temperature, dtype=np.float64)
    validate_conductivity(conductivity_array, grid.shape)
    if temperature_array.shape != grid.shape or not np.all(
        np.isfinite(temperature_array)
    ):
        raise ValueError("temperature must be finite and match grid shape")

    right_column = grid.nx // 2
    left_column = right_column - 1
    face_conductivity = harmonic_mean(
        conductivity_array[:, left_column],
        conductivity_array[:, right_column],
        harmonic_epsilon,
    )
    gradient = (
        temperature_array[:, right_column] - temperature_array[:, left_column]
    ) / grid.dx
    return face_conductivity * gradient


def dirichlet_outward_flux(
    grid: Grid2D,
    conductivity: NDArray[np.float64],
    temperature: NDArray[np.float64],
    bcs: BoundaryConditions,
) -> float:
    """Независимо суммировать outward flux через Dirichlet faces."""
    conductivity_array = np.asarray(conductivity, dtype=np.float64)
    temperature_array = np.asarray(temperature, dtype=np.float64)
    validate_conductivity(conductivity_array, grid.shape)
    if temperature_array.shape != grid.shape or not np.all(
        np.isfinite(temperature_array)
    ):
        raise ValueError("temperature must be finite and match grid shape")

    outward = 0.0
    if bcs.left.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[:, 0]
                * (temperature_array[:, 0] - bcs.left.value)
                / grid.dx
                * grid.dy
            )
        )
    if bcs.right.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[:, -1]
                * (temperature_array[:, -1] - bcs.right.value)
                / grid.dx
                * grid.dy
            )
        )
    if bcs.bottom.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[0, :]
                * (temperature_array[0, :] - bcs.bottom.value)
                / grid.dy
                * grid.dx
            )
        )
    if bcs.top.kind == "dirichlet":
        outward += float(
            np.sum(
                2.0
                * conductivity_array[-1, :]
                * (temperature_array[-1, :] - bcs.top.value)
                / grid.dy
                * grid.dx
            )
        )
    return outward


@dataclass(frozen=True)
class ValidationMetric:
    """Одна измеренная Gate 1 metric и её pre-registered criterion."""

    category: str
    name: str
    grid: str
    value: float
    threshold: float | None
    comparison: str
    passed: bool


@dataclass(frozen=True)
class Gate1ValidationBundle:
    """Metrics и immutable snapshots до rendering artifacts."""

    metrics: tuple[ValidationMetric, ...]
    fields: dict[str, NDArray[np.float64]]
    transient_times: NDArray[np.float64]
    transient_temperatures: NDArray[np.float64]
    config_hash: str
    input_hashes: dict[str, str]
    passed: bool


def _metric_max(
    category: str,
    name: str,
    grid: str,
    value: float,
    threshold: float,
) -> ValidationMetric:
    return ValidationMetric(
        category=category,
        name=name,
        grid=grid,
        value=float(value),
        threshold=float(threshold),
        comparison="<=",
        passed=bool(value <= threshold),
    )


def _metric_min(
    category: str,
    name: str,
    grid: str,
    value: float,
    threshold: float,
    *,
    strict: bool = False,
) -> ValidationMetric:
    return ValidationMetric(
        category=category,
        name=name,
        grid=grid,
        value=float(value),
        threshold=float(threshold),
        comparison=">" if strict else ">=",
        passed=bool(value > threshold if strict else value >= threshold),
    )


def _metric_info(
    category: str,
    name: str,
    grid: str,
    value: float,
) -> ValidationMetric:
    return ValidationMetric(
        category=category,
        name=name,
        grid=grid,
        value=float(value),
        threshold=None,
        comparison="informational",
        passed=True,
    )


def _canonical_config_hash(
    steady_config: dict[str, Any],
    transient_config: dict[str, Any],
) -> str:
    payload = json.dumps(
        {"steady": steady_config, "transient": transient_config},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_gate1_validation(
    steady_config: dict[str, Any],
    transient_config: dict[str, Any],
) -> Gate1ValidationBundle:
    """Вычислить все Gate 1 metrics до plotting и file I/O."""
    seed = int(steady_config["seed"])
    set_deterministic_seed(seed)
    tolerances = steady_config["tolerances"]
    resolutions = tuple(int(value) for value in steady_config["grids"]["validation"])
    metrics: list[ValidationMetric] = []
    input_hashes: dict[str, str] = {}

    constant_grid = Grid2D(nx=8, ny=6)
    constant_k = np.full(constant_grid.shape, 3.0)
    constant_source = np.zeros(constant_grid.shape)
    constant = solve_steady(
        constant_grid,
        constant_k,
        constant_source,
        BoundaryConditions.all_dirichlet(2.5),
    )
    constant_error = float(np.max(np.abs(constant.temperature - 2.5)))
    metrics.append(
        _metric_max(
            "analytical",
            "constant_absolute_error",
            "8x6",
            constant_error,
            float(tolerances["constant_absolute"]),
        )
    )

    linear_grid = Grid2D(nx=64, ny=32)
    linear_k = np.ones(linear_grid.shape)
    linear_source = np.zeros(linear_grid.shape)
    linear_exact = np.broadcast_to(linear_grid.x_centers, linear_grid.shape).copy()
    linear = solve_steady(
        linear_grid,
        linear_k,
        linear_source,
        BoundaryConditions.left_right(0.0, 1.0),
    )
    metrics.extend(
        (
            _metric_max(
                "analytical",
                "linear_relative_l2",
                "64x32",
                relative_l2(linear.temperature, linear_exact),
                float(tolerances["linear_relative_l2"]),
            ),
            _metric_max(
                "linear_system",
                "linear_normalized_residual",
                "64x32",
                linear.normalized_residual,
                float(tolerances["linear_relative_l2"]),
            ),
        )
    )
    input_hashes["linear_exact"] = content_hash(linear_exact)

    manufactured_errors: list[float] = []
    manufactured_finest = None
    manufactured_finest_result = None
    for resolution in resolutions:
        fixture = sine_manufactured_fixture(Grid2D(resolution, resolution))
        result = solve_steady(*fixture.solver_arguments())
        error = relative_l2(result.temperature, fixture.exact)
        manufactured_errors.append(error)
        metrics.append(
            _metric_info(
                "manufactured",
                f"manufactured_relative_l2_n{resolution}",
                f"{resolution}x{resolution}",
                error,
            )
        )
        input_hashes[f"manufactured_source_n{resolution}"] = content_hash(
            fixture.source
        )
        if resolution == resolutions[-1]:
            manufactured_finest = fixture
            manufactured_finest_result = result

    for coarse_index in range(len(manufactured_errors) - 1):
        reduction = (
            manufactured_errors[coarse_index] / manufactured_errors[coarse_index + 1]
        )
        metrics.append(
            _metric_min(
                "manufactured",
                f"manufactured_reduction_{resolutions[coarse_index]}_to_"
                f"{resolutions[coarse_index + 1]}",
                f"{resolutions[coarse_index]}->{resolutions[coarse_index + 1]}",
                reduction,
                float(tolerances["manufactured_min_reduction"]),
            )
        )
    manufactured_order = np.log(
        manufactured_errors[0] / manufactured_errors[-1]
    ) / np.log(resolutions[-1] / resolutions[0])
    metrics.append(
        _metric_min(
            "manufactured",
            "manufactured_empirical_order",
            f"{resolutions[0]}->{resolutions[-1]}",
            float(manufactured_order),
            float(tolerances["manufactured_min_order"]),
        )
    )

    harmonic_exact = 40.0 / 21.0
    harmonic_value = float(harmonic_mean(np.array([1.0]), np.array([20.0]))[0])
    harmonic_error = abs(harmonic_value - harmonic_exact) / harmonic_exact
    metrics.append(
        _metric_max(
            "heterogeneous",
            "harmonic_face_relative_error",
            "face",
            harmonic_error,
            float(tolerances["harmonic_relative"]),
        )
    )

    for resolution in resolutions:
        fixture = two_layer_fixture(Grid2D(resolution, resolution))
        result = solve_steady(*fixture.solver_arguments())
        flux = two_layer_interface_flux(
            fixture.grid, fixture.conductivity, result.temperature
        )
        l2_error = relative_l2(result.temperature, fixture.exact)
        flux_mean_error = abs(float(np.mean(flux)) - harmonic_exact) / harmonic_exact
        flux_variation = float(np.ptp(flux) / harmonic_exact)
        metrics.extend(
            (
                _metric_max(
                    "heterogeneous",
                    f"two_layer_relative_l2_n{resolution}",
                    f"{resolution}x{resolution}",
                    l2_error,
                    float(tolerances["two_layer_relative_l2"]),
                ),
                _metric_max(
                    "heterogeneous",
                    f"two_layer_flux_mean_relative_error_n{resolution}",
                    f"{resolution}x{resolution}",
                    flux_mean_error,
                    float(tolerances["two_layer_flux_relative"]),
                ),
                _metric_max(
                    "heterogeneous",
                    f"two_layer_flux_variation_n{resolution}",
                    f"{resolution}x{resolution}",
                    flux_variation,
                    float(tolerances["two_layer_flux_relative"]),
                ),
            )
        )
        input_hashes[f"two_layer_k_n{resolution}"] = content_hash(fixture.conductivity)

    energy_grid = Grid2D(nx=48, ny=40)
    energy_x, _ = energy_grid.mesh
    energy_k = np.where(energy_x < 0.5, 1.0, 20.0)
    energy_source = normalized_rectangular_source(energy_grid, 0.35, 0.65, 0.6, 0.8)
    production_bcs = BoundaryConditions.production()
    energy_result = solve_steady(energy_grid, energy_k, energy_source, production_bcs)
    generated = float(energy_source.sum() * energy_grid.dx * energy_grid.dy)
    outward = dirichlet_outward_flux(
        energy_grid, energy_k, energy_result.temperature, production_bcs
    )
    imbalance = abs(generated - outward) / max(abs(generated), abs(outward), 1e-12)
    metrics.append(
        _metric_max(
            "conservation",
            "global_energy_relative_imbalance",
            "48x40",
            imbalance,
            float(tolerances["energy_relative_imbalance"]),
        )
    )
    input_hashes["energy_source"] = content_hash(energy_source)
    input_hashes["energy_conductivity"] = content_hash(energy_k)

    symmetry_grid = Grid2D(nx=32, ny=32)
    symmetry_source = normalized_rectangular_source(symmetry_grid, 0.4, 0.6, 0.65, 0.85)
    symmetry_temperature = solve_steady(
        symmetry_grid,
        np.ones(symmetry_grid.shape),
        symmetry_source,
        production_bcs,
    ).temperature
    metrics.append(
        _metric_max(
            "physical",
            "symmetry_defect",
            "32x32",
            symmetry_defect(symmetry_temperature),
            float(tolerances["symmetry_defect"]),
        )
    )
    low_peak = float(np.max(symmetry_temperature))
    high_peak = float(
        np.max(
            solve_steady(
                symmetry_grid,
                np.full(symmetry_grid.shape, 20.0),
                symmetry_source,
                production_bcs,
            ).temperature
        )
    )
    metrics.extend(
        (
            _metric_info("physical", "peak_uniform_k1", "32x32", low_peak),
            _metric_info("physical", "peak_uniform_k20", "32x32", high_peak),
            _metric_max(
                "physical",
                "conductivity_monotonicity_delta",
                "32x32",
                high_peak - low_peak,
                float(tolerances["monotonicity_slack"]),
            ),
        )
    )

    operator_grid = Grid2D(nx=5, ny=4)
    operator_k = np.linspace(1.0, 20.0, 20).reshape(operator_grid.shape)
    operator_system = assemble_steady_system(
        operator_grid,
        operator_k,
        np.zeros(operator_grid.shape),
        production_bcs,
    )
    dense_operator = operator_system.matrix.toarray()
    diagonal = np.diag(dense_operator)
    off_diagonal = dense_operator - np.diag(diagonal)
    metrics.extend(
        (
            _metric_max(
                "operator",
                "matrix_symmetry_max_abs",
                "5x4",
                float(np.max(np.abs(dense_operator - dense_operator.T))),
                float(tolerances["matrix_symmetry_absolute"]),
            ),
            _metric_min(
                "operator",
                "matrix_min_diagonal",
                "5x4",
                float(np.min(diagonal)),
                0.0,
                strict=True,
            ),
            _metric_max(
                "operator",
                "matrix_max_off_diagonal",
                "5x4",
                float(np.max(off_diagonal)),
                float(tolerances["matrix_off_diagonal_max"]),
            ),
            _metric_min(
                "operator",
                "matrix_min_eigenvalue",
                "5x4",
                float(np.linalg.eigvalsh(dense_operator).min()),
                float(tolerances["matrix_min_eigenvalue"]),
                strict=True,
            ),
        )
    )

    steady_limit = transient_config["steady_limit"]
    transient_grid = Grid2D(nx=int(steady_limit["grid"]), ny=int(steady_limit["grid"]))
    transient_k = np.full(transient_grid.shape, float(steady_limit["conductivity"]))
    source_bounds = tuple(float(value) for value in steady_limit["source_rectangle"])
    transient_source = normalized_rectangular_source(transient_grid, *source_bounds)
    transient_initial = np.full(
        transient_grid.shape, float(steady_limit["initial_temperature"])
    )
    transient_config_object = TransientConfig(
        dt=float(steady_limit["dt"]),
        n_steps=int(steady_limit["maximum_steps"]),
        rho_c=float(steady_limit["rho_c"]),
    )
    transient = solve_transient(
        grid=transient_grid,
        conductivity=transient_k,
        source=transient_source,
        bcs=production_bcs,
        initial_temperature=transient_initial,
        config=transient_config_object,
    )
    steady_reference = solve_steady(
        transient_grid, transient_k, transient_source, production_bcs
    )
    transient_final = transient.temperatures[-1]
    transient_system = steady_reference.system
    transient_residual = float(
        np.linalg.norm(
            transient_system.matrix @ transient_final.ravel() - transient_system.rhs
        )
        / max(np.linalg.norm(transient_system.rhs), 1.0)
    )
    metrics.extend(
        (
            _metric_max(
                "transient",
                "steady_limit_relative_l2",
                "32x32",
                relative_l2(transient_final, steady_reference.temperature),
                float(steady_limit["relative_l2_tolerance"]),
            ),
            _metric_max(
                "transient",
                "steady_limit_residual",
                "32x32",
                transient_residual,
                float(steady_limit["residual_tolerance"]),
            ),
            _metric_max(
                "transient",
                "steady_limit_final_time_error",
                "32x32",
                abs(float(transient.times[-1]) - float(steady_limit["t_final"])),
                1e-14,
            ),
        )
    )
    input_hashes["transient_source"] = content_hash(transient_source)
    input_hashes["transient_initial"] = content_hash(transient_initial)

    timestep = transient_config["timestep_convergence"]
    timestep_grid = Grid2D(nx=int(timestep["grid"]), ny=int(timestep["grid"]))
    _, timestep_y = timestep_grid.mesh
    timestep_initial = np.sin(np.pi * timestep_y / 2.0)
    timestep_common = {
        "grid": timestep_grid,
        "conductivity": np.full(timestep_grid.shape, float(timestep["conductivity"])),
        "source": np.zeros(timestep_grid.shape),
        "bcs": production_bcs,
        "initial_temperature": timestep_initial,
    }

    def run_timestep(dt_key: str) -> Any:
        dt = float(timestep[dt_key])
        comparison_time = float(timestep["comparison_time"])
        steps = round(comparison_time / dt)
        if not np.isclose(steps * dt, comparison_time, rtol=0.0, atol=1e-14):
            raise ValueError(f"{dt_key} does not land on comparison_time")
        return solve_transient(
            **timestep_common,
            config=TransientConfig(
                dt=dt, n_steps=steps, rho_c=float(timestep["rho_c"])
            ),
        )

    coarse = run_timestep("dt_coarse")
    half = run_timestep("dt_half")
    reference = run_timestep("dt_reference")
    coarse_error = relative_l2(coarse.temperatures[-1], reference.temperatures[-1])
    half_error = relative_l2(half.temperatures[-1], reference.temperatures[-1])
    error_ratio = half_error / max(coarse_error, 1e-30)
    comparison_time = float(timestep["comparison_time"])
    time_error = max(
        abs(float(result.times[-1]) - comparison_time)
        for result in (coarse, half, reference)
    )
    metrics.extend(
        (
            _metric_info(
                "transient", "timestep_coarse_relative_l2", "32x32", coarse_error
            ),
            _metric_info("transient", "timestep_half_relative_l2", "32x32", half_error),
            _metric_max(
                "transient",
                "timestep_half_to_coarse_error_ratio",
                "32x32",
                error_ratio,
                float(timestep["half_to_coarse_error_max_ratio"]),
            ),
            _metric_max(
                "transient",
                "timestep_common_time_error",
                "32x32",
                time_error,
                1e-14,
            ),
        )
    )
    input_hashes["timestep_initial"] = content_hash(timestep_initial)

    if manufactured_finest is None or manufactured_finest_result is None:
        raise ValueError("manufactured validation requires at least one resolution")
    fields = {
        "linear_solution": linear.temperature.copy(),
        "manufactured_exact": manufactured_finest.exact.copy(),
        "manufactured_predicted": manufactured_finest_result.temperature.copy(),
        "manufactured_error": np.abs(
            manufactured_finest_result.temperature - manufactured_finest.exact
        ),
    }
    metric_tuple = tuple(metrics)
    return Gate1ValidationBundle(
        metrics=metric_tuple,
        fields=fields,
        transient_times=transient.times.copy(),
        transient_temperatures=transient.temperatures.copy(),
        config_hash=_canonical_config_hash(steady_config, transient_config),
        input_hashes=input_hashes,
        passed=all(metric.passed for metric in metric_tuple),
    )


def run_gate1_validation(config_dir: Path, artifact_dir: Path) -> bool:
    """Вычислить metrics, затем отдельно записать Gate 1 artifacts."""
    from waveforge.reporting.figures import (
        save_convergence_plot,
        save_field_figure,
        save_transient_convergence_gif,
    )
    from waveforge.reporting.summary import write_gate1_report
    from waveforge.reporting.tables import write_validation_metrics

    with (config_dir / "steady_validation.yaml").open(encoding="utf-8") as stream:
        steady_config = yaml.safe_load(stream)
    with (config_dir / "transient_validation.yaml").open(encoding="utf-8") as stream:
        transient_config = yaml.safe_load(stream)

    bundle = compute_gate1_validation(steady_config, transient_config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    solver_config = {
        "steady_validation": steady_config,
        "transient_validation": transient_config,
        "config_hash": bundle.config_hash,
        "input_hashes": bundle.input_hashes,
    }
    (artifact_dir / "solver_config.json").write_text(
        json.dumps(solver_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_validation_metrics(bundle.metrics, artifact_dir / "validation_metrics.csv")

    resolutions = np.asarray(steady_config["grids"]["validation"], dtype=np.int64)
    metric_lookup = {metric.name: metric.value for metric in bundle.metrics}
    manufactured_errors = np.asarray(
        [
            metric_lookup[f"manufactured_relative_l2_n{int(resolution)}"]
            for resolution in resolutions
        ],
        dtype=np.float64,
    )
    save_convergence_plot(
        resolutions,
        manufactured_errors,
        artifact_dir / "convergence_plot.png",
    )
    save_field_figure(
        bundle.fields["linear_solution"],
        artifact_dir / "linear_solution.png",
        title="Linear analytical solution: predicted",
    )
    save_field_figure(
        bundle.fields["manufactured_exact"],
        artifact_dir / "manufactured_solution_exact.png",
        title="Manufactured solution: exact",
    )
    save_field_figure(
        bundle.fields["manufactured_predicted"],
        artifact_dir / "manufactured_solution_predicted.png",
        title="Manufactured solution: predicted",
    )
    save_field_figure(
        bundle.fields["manufactured_error"],
        artifact_dir / "manufactured_solution_error.png",
        title="Manufactured solution: absolute error",
        cmap="viridis",
        colorbar_label="Absolute error",
    )
    save_transient_convergence_gif(
        bundle.transient_temperatures,
        bundle.transient_times,
        artifact_dir / "transient_convergence.gif",
    )
    write_gate1_report(
        bundle.metrics,
        bundle.passed,
        artifact_dir / "gate1_report.md",
        config_hash=bundle.config_hash,
        benchmark_frame=(
            pd.read_csv(artifact_dir.parent / "solver_benchmark.csv")
            if (artifact_dir.parent / "solver_benchmark.csv").is_file()
            else None
        ),
    )
    return bundle.passed


def main() -> int:
    """CLI entry point for Gate 1 validation artifacts."""
    parser = argparse.ArgumentParser(description="Run WaveForge Gate 1 validation")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/gate1_physics"),
    )
    arguments = parser.parse_args()
    passed = run_gate1_validation(arguments.config_dir, arguments.artifact_dir)
    print(f"Gate 1 numerical validation: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
