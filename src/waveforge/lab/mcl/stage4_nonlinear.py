"""Uncalibrated reduced-model drag/viscosity sensitivity, NOT Brinkman CFD."""

import numpy as np

from .contracts import MCLProblem
from .high_fidelity_eval import (
    _factor,
    base_resistance,
    cell_velocities,
    diagnostics,
    solve_flow,
    thermal_system,
)


def water_viscosity(temperature_c):
    temperature_c = np.asarray(temperature_c, dtype=float)
    if not np.isfinite(temperature_c).all() or np.any(temperature_c <= -133.15):
        raise ValueError("invalid temperature for approximate viscosity law")
    return 2.414e-5 * 10 ** (247.8 / (temperature_c + 133.15))


def forchheimer_beta(alpha_reference, mu_reference, c_f):
    if c_f < 0 or mu_reference <= 0 or np.any(alpha_reference <= 0):
        raise ValueError("positive mobility/resistance required")
    # alpha [Pa s/m2]; intrinsic permeability kappa=mu/alpha [m2].
    return c_f / np.sqrt(mu_reference / alpha_reference)


def solve_nonlinear(
    design,
    problem,
    *,
    h_air,
    c_f=0.55,
    variable_viscosity=True,
    resistance_ratio=1e7,
    max_iterations=120,
    tolerance=1e-10,
):
    d = problem.checked_design(design)
    if not np.all((d == 0) | (d == 1)) or not np.any(d):
        raise ValueError("open binary design required")
    s = problem.spec
    base = base_resistance(d, s, resistance_ratio)
    beta = forchheimer_beta(base, s.mu_pa_s, c_f)
    opened = d == 1
    temperatures, flows, scenario_diags = [], [], []
    for scenario in problem.power_w:
        single = MCLProblem(s, problem.chip_mask, scenario[None])
        alpha = base.copy()
        previous = None
        iteration = 0
        for _ in range(max_iterations):
            iteration += 1
            flow = solve_flow(alpha, s, s.pressure_drop_pa)
            matrix, rhs = thermal_system(d, single, flow, h_air)
            lu = _factor(matrix)
            temperature = lu.solve(rhs).T.reshape(1, 4, s.nx, s.ny)
            speed, _, _ = cell_velocities(flow, s)
            mu = np.full(d.shape, s.mu_pa_s)
            if variable_viscosity:
                # No coolant temperature/property is assigned to copper walls.
                mu[opened] = (
                    s.mu_pa_s
                    * water_viscosity(25 + temperature[0, 3, opened])
                    / water_viscosity(25)
                )
            target = base * mu / s.mu_pa_s + beta * s.density_kg_m3 * speed
            # Relative local mismatch, not a global norm dominated by solid walls.
            alpha_error = float(
                np.max(np.abs(target - alpha) / np.maximum(np.abs(target), 1e-30))
            )
            thermal_error = (
                float("inf")
                if previous is None
                else float(
                    np.max(np.abs(temperature - previous))
                    / max(np.max(np.abs(temperature)), 1e-30)
                )
            )
            if alpha_error < tolerance and (
                thermal_error < tolerance or previous is None
            ):
                break
            previous = temperature.copy()
            alpha = 0.5 * alpha + 0.5 * target
        diag = diagnostics(
            d, single, flow, temperature, s.pressure_drop_pa, h_air, matrix, rhs, lu
        )
        diameter = 2 * s.mobility_length_m * s.dz[3] / (s.mobility_length_m + s.dz[3])
        # Cell-based Re uses the local viscosity; constant-mu face-based estimate
        # remains separately in diagnostics, so it is not silently replaced.
        diag.update(
            nonlinear_converged=alpha_error < tolerance and thermal_error < tolerance
            if previous is not None
            else alpha_error < tolerance,
            nonlinear_iterations=iteration,
            alpha_fixed_point_residual=alpha_error,
            thermal_iteration_change=thermal_error
            if np.isfinite(thermal_error)
            else 0.0,
            local_cell_re_variable_mu=float(
                np.max(s.density_kg_m3 * speed * diameter / mu)
            ),
            viscosity_temperature_range_25_100c=bool(
                np.all(
                    (temperature[0, 3, opened] >= 0) & (temperature[0, 3, opened] <= 75)
                )
            ),
            c_f=c_f,
            resistance_ratio=resistance_ratio,
            variable_viscosity=variable_viscosity,
            physical_validation=False,
            flow_model="local nonlinear Darcy drag; NO Brinkman Laplacian",
        )
        temperatures.append(temperature[0])
        flows.append(float(flow.qin.sum() * 1e6))
        scenario_diags.append(diag)
    return dict(
        temperature=np.asarray(temperatures),
        scenario_flow_ml_s=flows,
        scenario_diagnostics=scenario_diags,
        converged=all(x["nonlinear_converged"] for x in scenario_diags),
    )
