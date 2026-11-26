"""Exact pressure scaling between regimes; unchanged thermal solves and gates."""

import numpy as np
from scipy.ndimage import label

from .high_fidelity_eval import Flow, base_resistance, solve_flow
from .stage5_admission import admission
from .stage5_preflight import profile_from_flow, solve_profile


def record_result(design, problem, result, pressure=None):
    """The inherited geometry and admission rules, without repeating a solve."""
    s = problem.spec
    labels, count = label(design)
    common = np.intersect1d(
        labels[0, s.port_start : s.port_stop], labels[-1, s.port_start : s.port_stop]
    )
    geometry = count == 1 and np.any(common > 0) and design.sum() == s.nx * s.ny // 4
    row = dict(
        accepted=bool(geometry),
        peak_rise_k=float(result.scenario_peaks.max()),
        silicon_peak_c=float(result.scenario_peaks.max() + 25),
        flow_ml_s=result.flow_ml_s,
        pressure_pa=pressure or s.pressure_drop_pa,
        diagnostics=result.diagnostics,
    )
    row["admission"] = admission(row)
    return row


def solve_dual_profile(design, problem, *, pump_power_w, k_tim=40.0):
    if not np.isfinite(pump_power_w) or pump_power_w <= 0:
        raise ValueError("positive finite hydraulic pump power required")
    d = problem.checked_design(design)
    if not np.all((d == 0) | (d == 1)):
        raise ValueError("binary geometry required")
    s = problem.spec
    flow, matrix, rhs, factor = solve_flow(
        base_resistance(d, s), s, s.pressure_drop_pa, return_linear_system=True
    )
    nominal = profile_from_flow(
        d, problem, flow, k_tim=k_tim, pressure_drop_pa=s.pressure_drop_pa
    )
    dp = np.sqrt(pump_power_w * s.pressure_drop_pa / (nominal.flow_ml_s * 1e-6))
    scale = dp / s.pressure_drop_pa
    pressure = flow.pressure * scale
    scaled_rhs = rhs * scale
    denominator = max(np.linalg.norm(scaled_rhs), 1e-30)
    raw = float(np.linalg.norm(matrix @ pressure.ravel() - scaled_rhs) / denominator)
    accurate = float(factor.relative_residual(pressure.ravel(), scaled_rhs))
    if not np.isfinite(raw + accurate) or max(raw, accurate) >= 1e-12:
        # A numerical fallback retains the original independent solve and gates.
        powered = solve_profile(d, problem, k_tim=k_tim, pressure_drop_pa=dp)
    else:
        powered_flow = Flow(
            pressure,
            flow.qx * scale,
            flow.qy * scale,
            flow.qin * scale,
            flow.qout * scale,
            raw,
            accurate,
        )
        powered = profile_from_flow(
            d, problem, powered_flow, k_tim=k_tim, pressure_drop_pa=dp
        )
    return nominal, powered
