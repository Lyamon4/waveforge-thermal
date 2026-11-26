"""Independent NumPy/SciPy finite-volume assembly of the USER-SUPPLIED model.

This is Darcy/Hele-Shaw plus empirical heat exchange, NOT resolved Brinkman CFD.
The floor permeability and air sink deliberately reproduce the supplied equations.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .contracts import MCLProblem
from .factorization import Factorization


@dataclass
class ReferenceResult:
    pressure: np.ndarray
    temperature: np.ndarray  # [scenario,z,x,y], temperature rise K
    flow_ml_s: float
    scenario_peaks: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    qin: np.ndarray
    qout: np.ndarray
    diagnostics: dict


def _relative_residual(matrix, solution, rhs):
    residual = matrix @ solution - rhs
    return np.linalg.norm(residual, axis=0) / np.maximum(
        np.linalg.norm(rhs, axis=0), 1e-30
    )


def solve_reference(design, problem: MCLProblem, *, _frozen_flows=None):
    """Real evaluation, or local analytic continuation for complex-step ONLY.

    The private continuation freezes upwind branches at a real base state; no
    physical diagnostics or maxima are computed from complex temperatures.
    """
    if _frozen_flows is None:
        if np.iscomplexobj(design):
            raise ValueError("complex design requires frozen real upwind branches")
        d = problem.checked_design(design)
    else:
        problem.checked_design(np.real(design))
        d = np.asarray(design, dtype=np.complex128)
        if not np.isfinite(d).all():
            raise ValueError("nonfinite complex design")
    s = problem.spec
    nx, ny, dz = s.nx, s.ny, s.dz
    n = nx * ny
    ids = np.arange(n).reshape(nx, ny)
    port = slice(s.port_start, s.port_stop)
    inlet, outlet = ids[0, port], ids[-1, port]
    chip = problem.chip_mask
    alpha_min = 12 * s.mu_pa_s / s.mobility_length_m**2
    alpha_max = s.resistance_ratio * alpha_min
    # Algebraically identical BP law; avoids subtracting ~1e9 to recover ~1e4
    # at D=1, where the literal expression loses endpoint accuracy.
    alpha = alpha_min + (alpha_max - alpha_min) * s.q_bp * (1 - d) / (d + s.q_bp)
    kh = 1 / alpha
    rows, cols, values = [], [], []

    def edge(u, v, conductance):
        u, v, g = np.ravel(u), np.ravel(v), np.ravel(conductance)
        rows.extend((u, v, u, v))
        cols.extend((u, v, v, u))
        values.extend((g, g, -g, -g))

    def diagonal(indices, conductance):
        rows.append(np.ravel(indices))
        cols.append(np.ravel(indices))
        values.append(np.ravel(conductance))

    def matrix(size):
        return sp.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(size, size),
        ).tocsc()

    kx = 2 * kh[:-1] * kh[1:] / (kh[:-1] + kh[1:])
    ky = 2 * kh[:, :-1] * kh[:, 1:] / (kh[:, :-1] + kh[:, 1:])
    gx, gy = kx * s.dy * dz[3] / s.dx, ky * s.dx * dz[3] / s.dy
    edge(ids[:-1], ids[1:], gx)
    edge(ids[:, :-1], ids[:, 1:], gy)
    gin = 2 * kh[0, port] * s.dy * dz[3] / s.dx
    gout = 2 * kh[-1, port] * s.dy * dz[3] / s.dx
    diagonal(inlet, gin)
    diagonal(outlet, gout)
    kp = matrix(n)
    rhs_p = np.zeros(n, dtype=d.dtype)
    rhs_p[inlet] = gin * s.pressure_drop_pa
    pressure_factor = Factorization(
        kp, refine=s.scaled_refinement, compensated_refinement=s.compensated_refinement
    )
    pressure_flat = pressure_factor.solve(rhs_p)
    pressure = pressure_flat.reshape(nx, ny)
    qx = gx * (pressure[:-1] - pressure[1:])
    qy = gy * (pressure[:, :-1] - pressure[:, 1:])
    qin = gin * (s.pressure_drop_pa - pressure[0, port])
    qout = gout * pressure[-1, port]

    rows, cols, values = [], [], []
    ktop = 0.6 + 399.4 * (1 - d) ** 3
    conductivities = (
        np.where(chip, 130.0, 0.0),
        np.where(chip, 5.0, 0.0),
        np.full((nx, ny), 400.0),
        ktop,
    )
    for z, k in enumerate(conductivities):
        k = np.asarray(k, dtype=d.dtype)
        kx = np.divide(
            2 * k[:-1] * k[1:],
            k[:-1] + k[1:],
            out=np.zeros((nx - 1, ny), dtype=d.dtype),
            where=k[:-1] + k[1:] != 0,
        )
        ky = np.divide(
            2 * k[:, :-1] * k[:, 1:],
            k[:, :-1] + k[:, 1:],
            out=np.zeros((nx, ny - 1), dtype=d.dtype),
            where=k[:, :-1] + k[:, 1:] != 0,
        )
        edge(z * n + ids[:-1], z * n + ids[1:], kx * s.dy * dz[z] / s.dx)
        edge(z * n + ids[:, :-1], z * n + ids[:, 1:], ky * s.dx * dz[z] / s.dy)

    r01 = dz[0] / 260 + 2e-6 + dz[1] / 10
    r12 = dz[1] / 10 + 2e-6 + dz[2] / 800
    h_local = s.h_fluid * d**2 + (1 - d**2) * ktop / (dz[3] / 2)
    r23 = dz[2] / 800 + 1 / h_local
    edge(ids, n + ids, np.where(chip, s.area / r01, 0))
    edge(n + ids, 2 * n + ids, np.where(chip, s.area / r12, 0))
    edge(2 * n + ids, 3 * n + ids, s.area / r23)

    def positive_part(flow, branch_index):
        if _frozen_flows is None:
            return np.maximum(flow, 0), np.maximum(-flow, 0)
        base_flow = _frozen_flows[branch_index]
        return np.where(base_flow > 0, flow, 0), np.where(base_flow < 0, -flow, 0)

    for branch_index, (u, v, flow) in enumerate(
        (
            (3 * n + ids[:-1], 3 * n + ids[1:], qx),
            (3 * n + ids[:, :-1], 3 * n + ids[:, 1:], qy),
        )
    ):
        u, v = u.ravel(), v.ravel()
        qp, qm = positive_part(flow, branch_index)
        positive = (s.rho_cp * qp).ravel()
        negative = (s.rho_cp * qm).ravel()
        rows.extend((u, u, v, v))
        cols.extend((u, v, u, v))
        values.extend((positive, -negative, -positive, negative))
    diagonal(3 * n + outlet, s.rho_cp * positive_part(qout, 2)[0])
    diagonal(3 * n + ids, np.full((nx, ny), s.h_air * s.area))
    for z in (0, 1):
        diagonal(z * n + ids[~chip], np.ones(np.count_nonzero(~chip)))
    kt = matrix(4 * n)
    rhs = np.zeros((4 * n, len(problem.power_w)), dtype=d.dtype)
    rhs[:n] = problem.power_w.reshape(-1, n).T
    thermal_factor = Factorization(
        kt, refine=s.scaled_refinement, compensated_refinement=s.compensated_refinement
    )
    thermal_flat = thermal_factor.solve(rhs)
    temperature = thermal_flat.T.reshape(-1, 4, nx, ny)
    if _frozen_flows is not None:
        return ReferenceResult(
            pressure,
            temperature,
            qin.sum() * 1e6,
            np.empty(0),
            qx,
            qy,
            qin,
            qout,
            {
                "pressure_residual": float(
                    _relative_residual(kp, pressure_flat, rhs_p)
                ),
                "thermal_residual": _relative_residual(kt, thermal_flat, rhs).tolist(),
                "compensated_pressure_residual": float(
                    pressure_factor.relative_residual(pressure_flat, rhs_p)
                ),
                "compensated_thermal_residual": thermal_factor.relative_residual(
                    thermal_flat, rhs
                ).tolist(),
            },
        )
    peaks = temperature[:, 0, chip].max(axis=1)
    incoming_power = problem.power_w.sum((1, 2))
    outlet_power = s.rho_cp * (temperature[:, 3, -1, port] * np.maximum(qout, 0)).sum(1)
    air_power = s.h_air * s.area * temperature[:, 3].sum((1, 2))
    energy = np.abs(outlet_power + air_power - incoming_power) / np.maximum(
        incoming_power, 1e-30
    )
    width = (s.port_stop - s.port_start) * s.dy
    velocity = float(qin.sum()) / (width * dz[3])
    hydraulic_width = width if s.channel_width_m is None else s.channel_width_m
    diameter = 2 * hydraulic_width * dz[3] / (hydraulic_width + dz[3])
    reynolds = s.density_kg_m3 * abs(velocity) * diameter / s.mu_pa_s
    # Face and centered cell estimates; homogenized velocities assume porosity=1.
    ux = np.zeros((nx + 1, ny))
    uy = np.zeros((nx, ny + 1))
    ux[1:-1] = qx / (s.dy * dz[3])
    ux[0, port] = qin / (s.dy * dz[3])
    ux[-1, port] = qout / (s.dy * dz[3])
    uy[:, 1:-1] = qy / (s.dx * dz[3])
    speed = np.hypot((ux[:-1] + ux[1:]) / 2, (uy[:, :-1] + uy[:, 1:]) / 2)
    local_speed = float(max(np.max(np.abs(ux)), np.max(np.abs(uy)), speed.max()))
    local_re = s.density_kg_m3 * local_speed * diameter / s.mu_pa_s
    warnings = ["reduced_model_not_physically_validated", "finite_solid_permeability"]
    if reynolds > 2300:
        warnings.append("high_reynolds")
    open_water = d == 1
    water_peak_c = (
        float(temperature[:, 3, open_water].max() + 25) if np.any(open_water) else None
    )
    if water_peak_c is not None and water_peak_c > 100:
        warnings.append("check_water_phase_and_temperature_dependent_properties")
    if np.any((d > 0) & (d < 1)):
        warnings.append("intermediate_material_has_no_resolved_fluid_temperature")
    diagnostics = {
        "pressure_residual": float(_relative_residual(kp, pressure_flat, rhs_p)),
        "thermal_residual": _relative_residual(kt, thermal_flat, rhs).tolist(),
        "compensated_pressure_residual": float(
            pressure_factor.relative_residual(pressure_flat, rhs_p)
        ),
        "compensated_thermal_residual": thermal_factor.relative_residual(
            thermal_flat, rhs
        ).tolist(),
        "mass_relative_error": float(
            abs(qin.sum() - qout.sum()) / max(abs(qin.sum()), 1e-30)
        ),
        "energy_relative_error": energy.tolist(),
        "outlet_heat_w": outlet_power.tolist(),
        "air_heat_w": air_power.tolist(),
        "air_heat_fraction": np.divide(
            air_power,
            incoming_power,
            out=np.zeros_like(air_power),
            where=incoming_power > 0,
        ).tolist(),
        "reduced_order_homogenized_model": s.reduced_order_homogenized_model,
        "housing_sink_empirically_validated": False,
        "inlet_velocity_m_s": velocity,
        "inlet_reynolds_estimate": reynolds,
        "max_local_reynolds_estimate": local_re,
        "max_local_speed_m_s": local_speed,
        "hydraulic_diameter_m": diameter,
        "local_re_is_sampled_surrogate_not_universal_bound": True,
        "refinement_steps": thermal_factor.refinement_steps,
        "pump_hydraulic_power_w": float(s.pressure_drop_pa * qin.sum()),
        "physical_validation": False,
        "warnings": warnings,
        "water_fraction": float(d.mean()),
        "peak_over_all_layers_k": float(temperature.max()),
        "peak_open_water_temperature_c": water_peak_c,
    }
    return ReferenceResult(
        pressure,
        temperature,
        float(qin.sum() * 1e6),
        peaks,
        qx,
        qy,
        qin,
        qout,
        diagnostics,
    )
