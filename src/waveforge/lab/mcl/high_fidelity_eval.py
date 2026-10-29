"""Frozen-design sensitivity evaluator, NOT resolved/high-fidelity CFD.

Independent finite-volume assembly, no changes to the sealed training operator.
Hydraulic mobility is m2/(Pa s); intrinsic permeability is mu_ref * mobility.
"""

from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from .contracts import MCLProblem
from .factorization import Factorization
from .reference import ReferenceResult


def hydraulic_targets(flow_ml_s, pressure_pa, target_ml_s, power_w):
    values = np.asarray([flow_ml_s, pressure_pa, target_ml_s, power_w], float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("positive finite hydraulic targets required")
    conductance = flow_ml_s * 1e-6 / pressure_pa
    dp = target_ml_s * 1e-6 / conductance
    return dict(
        conductance_m3_s_pa=conductance,
        iso_flow_pressure_pa=dp,
        iso_flow_power_w=dp * target_ml_s * 1e-6,
        iso_power_pressure_pa=float(np.sqrt(power_w / conductance)),
    )


def refine_problem(problem, design, factor=2):
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError("positive integer refinement factor required")
    d = problem.checked_design(design)
    s = problem.spec
    finer = replace(
        s,
        nx=s.nx * factor,
        ny=s.ny * factor,
        port_start=s.port_start * factor,
        port_stop=s.port_stop * factor,
    )

    def tile(a):
        return np.repeat(np.repeat(a, factor, axis=-2), factor, axis=-1)

    return MCLProblem(
        finer, tile(problem.chip_mask), tile(problem.power_w) / factor**2
    ), tile(d)


class _Assembly:
    def __init__(self, size):
        self.size = size
        self.rows = []
        self.cols = []
        self.values = []

    def add(self, row, col, value):
        self.rows.append(np.ravel(row))
        self.cols.append(np.ravel(col))
        self.values.append(np.broadcast_to(value, np.shape(row)).ravel())

    def edge(self, u, v, g):
        self.add(u, u, g)
        self.add(v, v, g)
        self.add(u, v, -g)
        self.add(v, u, -g)

    def matrix(self):
        return sp.coo_matrix(
            (
                np.concatenate(self.values),
                (np.concatenate(self.rows), np.concatenate(self.cols)),
            ),
            shape=(self.size, self.size),
        ).tocsc()


def _harmonic(a, b):
    return np.divide(2 * a * b, a + b, out=np.zeros_like(a), where=(a + b) != 0)


def base_resistance(design, spec, resistance_ratio=None):
    ratio = spec.resistance_ratio if resistance_ratio is None else resistance_ratio
    if not np.isfinite(ratio) or ratio < 1:
        raise ValueError("resistance ratio >=1 required")
    minimum = 12 * spec.mu_pa_s / spec.mobility_length_m**2
    return minimum * (1 + (ratio - 1) * spec.q_bp * (1 - design) / (design + spec.q_bp))


def _factor(matrix):
    return Factorization(matrix, refine=True, compensated_refinement=True)


def _residual(matrix, x, rhs):
    return np.linalg.norm(matrix @ x - rhs, axis=0) / np.maximum(
        np.linalg.norm(rhs, axis=0), 1e-30
    )


@dataclass
class Flow:
    pressure: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    qin: np.ndarray
    qout: np.ndarray
    raw_residual: float
    compensated_residual: float


def solve_flow(alpha, s, pressure_pa):
    n = s.nx * s.ny
    ids = np.arange(n).reshape(s.nx, s.ny)
    port = slice(s.port_start, s.port_stop)
    kh = 1 / alpha
    hx, hy = s.dy * s.dz[3] / s.dx, s.dx * s.dz[3] / s.dy
    gx = _harmonic(kh[:-1], kh[1:]) * hx
    gy = _harmonic(kh[:, :-1], kh[:, 1:]) * hy
    gin = 2 * kh[0, port] * hx
    gout = 2 * kh[-1, port] * hx
    a = _Assembly(n)
    a.edge(ids[:-1], ids[1:], gx)
    a.edge(ids[:, :-1], ids[:, 1:], gy)
    a.add(ids[0, port], ids[0, port], gin)
    a.add(ids[-1, port], ids[-1, port], gout)
    matrix = a.matrix()
    rhs = np.zeros(n)
    rhs[ids[0, port]] = gin * pressure_pa
    lu = _factor(matrix)
    p = lu.solve(rhs)
    pressure = p.reshape(s.nx, s.ny)
    return Flow(
        pressure,
        gx * (pressure[:-1] - pressure[1:]),
        gy * (pressure[:, :-1] - pressure[:, 1:]),
        gin * (pressure_pa - pressure[0, port]),
        gout * pressure[-1, port],
        float(_residual(matrix, p, rhs)),
        float(lu.relative_residual(p, rhs)),
    )


def cell_velocities(flow, s):
    ux = np.zeros((s.nx + 1, s.ny))
    uy = np.zeros((s.nx, s.ny + 1))
    port = slice(s.port_start, s.port_stop)
    ux[1:-1] = flow.qx / (s.dy * s.dz[3])
    uy[:, 1:-1] = flow.qy / (s.dx * s.dz[3])
    ux[0, port] = flow.qin / (s.dy * s.dz[3])
    ux[-1, port] = flow.qout / (s.dy * s.dz[3])
    speed = np.hypot((ux[:-1] + ux[1:]) / 2, (uy[:, :-1] + uy[:, 1:]) / 2)
    return speed, ux, uy


def thermal_system(d, problem, flow, h_air):
    s = problem.spec
    n = s.nx * s.ny
    ids = np.arange(n).reshape(s.nx, s.ny)
    chip = problem.chip_mask
    a = _Assembly(4 * n)
    top = 0.6 + 399.4 * (1 - d) ** 3
    conductivities = [
        np.where(chip, 130.0, 0.0),
        np.where(chip, 5.0, 0.0),
        np.full(d.shape, 400.0),
        top,
    ]
    for z, k in enumerate(conductivities):
        a.edge(
            z * n + ids[:-1],
            z * n + ids[1:],
            _harmonic(k[:-1], k[1:]) * s.dy * s.dz[z] / s.dx,
        )
        a.edge(
            z * n + ids[:, :-1],
            z * n + ids[:, 1:],
            _harmonic(k[:, :-1], k[:, 1:]) * s.dx * s.dz[z] / s.dy,
        )
    r01 = s.dz[0] / 260 + 2e-6 + s.dz[1] / 10
    r12 = s.dz[1] / 10 + 2e-6 + s.dz[2] / 800
    hlocal = s.h_fluid * d * d + (1 - d * d) * top / (s.dz[3] / 2)
    a.edge(ids, n + ids, np.where(chip, s.area / r01, 0))
    a.edge(n + ids, 2 * n + ids, np.where(chip, s.area / r12, 0))
    a.edge(2 * n + ids, 3 * n + ids, s.area / (s.dz[2] / 800 + 1 / hlocal))
    for u, v, q in (
        (3 * n + ids[:-1], 3 * n + ids[1:], flow.qx),
        (3 * n + ids[:, :-1], 3 * n + ids[:, 1:], flow.qy),
    ):
        pos = s.rho_cp * np.maximum(q, 0)
        neg = s.rho_cp * np.maximum(-q, 0)
        a.add(u, u, pos)
        a.add(u, v, -neg)
        a.add(v, u, -pos)
        a.add(v, v, neg)
    outlet = 3 * n + ids[-1, s.port_start : s.port_stop]
    a.add(outlet, outlet, s.rho_cp * np.maximum(flow.qout, 0))
    a.add(3 * n + ids, 3 * n + ids, h_air * s.area)
    for z in (0, 1):
        a.add(z * n + ids[~chip], z * n + ids[~chip], 1.0)
    rhs = np.zeros((4 * n, len(problem.power_w)))
    rhs[:n] = problem.power_w.reshape(-1, n).T
    return a.matrix(), rhs


def diagnostics(d, problem, flow, temperature, dp, h_air, matrix, rhs, lu):
    s = problem.spec
    port = slice(s.port_start, s.port_stop)
    n = s.nx * s.ny
    flat = temperature.reshape(-1, 4 * n).T
    incoming = problem.power_w.sum((1, 2))
    outlet = s.rho_cp * (temperature[:, 3, -1, port] * np.maximum(flow.qout, 0)).sum(1)
    air = h_air * s.area * temperature[:, 3].sum((1, 2))
    speed, ux, uy = cell_velocities(flow, s)
    opened = d == 1
    diameter = 2 * s.mobility_length_m * s.dz[3] / (s.mobility_length_m + s.dz[3])
    local_speed = max(np.abs(ux).max(), np.abs(uy).max(), speed.max())
    total = max(abs(flow.qin.sum()), 1e-30)
    walls_x = ~(opened[:-1] & opened[1:])
    walls_y = ~(opened[:, :-1] & opened[:, 1:])
    wall_flux = np.abs(flow.qx[walls_x]).sum() + np.abs(flow.qy[walls_y]).sum()
    all_flux = np.abs(flow.qx).sum() + np.abs(flow.qy).sum()
    return dict(
        pressure_residual=flow.raw_residual,
        compensated_pressure_residual=flow.compensated_residual,
        thermal_residual=_residual(matrix, flat, rhs).tolist(),
        compensated_thermal_residual=lu.relative_residual(flat, rhs).tolist(),
        mass_relative_error=float(abs(flow.qin.sum() - flow.qout.sum()) / total),
        energy_relative_error=(
            np.abs(outlet + air - incoming) / np.maximum(incoming, 1e-30)
        ).tolist(),
        outlet_heat_w=outlet.tolist(),
        air_heat_w=air.tolist(),
        water_heat_fraction=(outlet / np.maximum(incoming, 1e-30)).tolist(),
        air_heat_fraction=(air / np.maximum(incoming, 1e-30)).tolist(),
        peak_silicon_temperature_c=float(
            temperature[:, 0, problem.chip_mask].max() + 25
        ),
        peak_open_water_temperature_c=float(temperature[:, 3, opened].max() + 25)
        if opened.any()
        else None,
        max_local_reynolds_estimate=float(
            s.density_kg_m3 * local_speed * diameter / s.mu_pa_s
        ),
        max_local_speed_m_s=float(local_speed),
        flowing_open_fraction=float(np.mean(speed[opened] > 0.1 * speed[opened].mean()))
        if opened.any()
        else 0.0,
        wall_touching_internal_abs_face_flux_fraction=float(
            wall_flux / max(all_flux, 1e-30)
        ),
        wall_outlet_flow_fraction=float(
            np.abs(flow.qout[~opened[-1, port]]).sum() / total
        ),
        pump_hydraulic_power_w=float(dp * flow.qin.sum()),
        physical_validation=False,
        flow_model="depth-averaged Darcy; empirical thermal exchange",
        h_air=h_air,
        pressure_drop_pa=dp,
    )


def solve_audit(
    design,
    problem,
    *,
    h_air,
    pressure_drop_pa=None,
    scheme="upwind",
    resistance_ratio=None,
):
    s = problem.spec
    d = problem.checked_design(design)
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("audit requires frozen binary geometry")
    dp = s.pressure_drop_pa if pressure_drop_pa is None else pressure_drop_pa
    if not np.isfinite([h_air, dp]).all() or min(h_air, dp) < 0:
        raise ValueError("invalid sink or pressure")
    if h_air == 0 and (dp == 0 or not np.any(d)):
        raise ValueError("no open water heat sink")
    if h_air == 0:
        from scipy.ndimage import label

        components, _ = label(d == 1)
        port = slice(s.port_start, s.port_stop)
        common = np.intersect1d(components[0, port], components[-1, port])
        if not np.any(common > 0):
            raise ValueError(
                "no open inlet-to-outlet path; wall leakage is not a water sink"
            )
    if scheme not in ("upwind", "vanleer"):
        raise ValueError("unsupported transport scheme")
    flow = solve_flow(base_resistance(d, s, resistance_ratio), s, dp)
    matrix, rhs = thermal_system(d, problem, flow, h_air)
    lu = _factor(matrix)
    theta = lu.solve(rhs).T.reshape(-1, 4, s.nx, s.ny)
    transport = {}
    if scheme == "vanleer":
        from .stage4_transport import solve_limited

        theta, transport = solve_limited(lu, matrix, rhs, theta, d, flow, s)
    diag = diagnostics(d, problem, flow, theta, dp, h_air, matrix, rhs, lu)
    if transport:
        # Residual against the upwind matrix alone is not the limited equation.
        diag["upwind_only_residual_for_limited_field"] = diag.pop("thermal_residual")
        diag["upwind_only_compensated_residual_for_limited_field"] = diag.pop(
            "compensated_thermal_residual"
        )
        diag.update(transport)
        diag["thermal_residual"] = transport["inner_linear_residual"]
        diag["compensated_thermal_residual"] = transport["inner_compensated_residual"]
    return ReferenceResult(
        flow.pressure,
        theta,
        float(flow.qin.sum() * 1e6),
        theta[:, 0, problem.chip_mask].max(1),
        flow.qx,
        flow.qy,
        flow.qin,
        flow.qout,
        diag,
    )
