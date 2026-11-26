"""Proposed Stage5 forward diagnostics, not a qualified training/CFD operator.

Keep all old operators sealed. Apply only the explicitly requested TIM
conductivity change to the independently assembled Stage4 matrix.
"""

import numpy as np

from .high_fidelity_eval import (
    _Assembly,
    _factor,
    base_resistance,
    diagnostics,
    solve_flow,
    thermal_system,
)
from .reference import ReferenceResult


def thermal_with_tim(design, problem, flow, *, k_tim):
    if not np.isfinite(k_tim) or k_tim <= 0:
        raise ValueError("positive finite TIM conductivity required")
    s = problem.spec
    matrix, rhs = thermal_system(design, problem, flow, s.h_air)
    if k_tim == 5.0:
        return matrix, rhs
    n = s.nx * s.ny
    ids = np.arange(n).reshape(s.nx, s.ny)
    chip = problem.chip_mask
    change = _Assembly(4 * n)
    # Both active sides have the same k: harmonic conductivity is exactly k.
    change.edge(
        n + ids[:-1],
        n + ids[1:],
        (chip[:-1] & chip[1:]) * (k_tim - 5) * s.dy * s.dz[1] / s.dx,
    )
    change.edge(
        n + ids[:, :-1],
        n + ids[:, 1:],
        (chip[:, :-1] & chip[:, 1:]) * (k_tim - 5) * s.dx * s.dz[1] / s.dy,
    )
    # Do NOT delete either explicit contact resistance.
    for lower, fixed in ((0, s.dz[0] / 260 + 2e-6), (1, s.dz[2] / 800 + 2e-6)):
        old = s.area / (fixed + s.dz[1] / 10)
        new = s.area / (fixed + s.dz[1] / (2 * k_tim))
        change.edge(lower * n + ids, (lower + 1) * n + ids, chip * (new - old))
    return (matrix + change.matrix()).tocsc(), rhs


def solve_profile(design, problem, *, k_tim, pressure_drop_pa=None):
    s = problem.spec
    d = problem.checked_design(design)
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("binary geometry required")
    dp = s.pressure_drop_pa if pressure_drop_pa is None else pressure_drop_pa
    if not np.isfinite(dp) or dp <= 0:
        raise ValueError("positive finite pressure required")
    flow = solve_flow(base_resistance(d, s), s, dp)
    return profile_from_flow(d, problem, flow, k_tim=k_tim, pressure_drop_pa=dp)


def profile_from_flow(design, problem, flow, *, k_tim, pressure_drop_pa):
    """Same thermal assembly and diagnostics for an already qualified flow."""
    s = problem.spec
    d = problem.checked_design(design)
    dp = pressure_drop_pa
    matrix, rhs = thermal_with_tim(d, problem, flow, k_tim=k_tim)
    lu = _factor(matrix)
    theta = lu.solve(rhs).T.reshape(-1, 4, s.nx, s.ny)
    diag = diagnostics(d, problem, flow, theta, dp, s.h_air, matrix, rhs, lu)
    diag.update(
        tim_conductivity_w_m_k=float(k_tim),
        tim_contact_resistance_each_m2_k_w=2e-6,
        training_operator_qualified=False,
    )
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
