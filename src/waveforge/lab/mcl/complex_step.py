"""Independent complex-step verification on a fixed smooth upwind branch."""

import numpy as np

from .reference import solve_reference


def complex_step_check(
    design, problem, directions, thermal_gradient, flow_gradient, *, epsilon=1e-20
):
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive finite complex step required")
    design = problem.checked_design(design)
    directions = np.asarray(directions, dtype=np.float64)
    if (
        directions.ndim != 3
        or directions.shape[1:] != design.shape
        or not len(directions)
        or not np.isfinite(directions).all()
    ):
        raise ValueError("nonempty finite directions [direction,x,y] required")
    base = solve_reference(design, problem)
    branches = (base.qx, base.qy, base.qout)
    # At a zero flow the upwind operator is not generally differentiable.
    if any(np.any(flow == 0) for flow in branches):
        raise ValueError("ambiguous zero-flow upwind branch; no derivative certificate")
    shift = float((5 * base.temperature[:, 0, problem.chip_mask]).max())
    results, residuals = [], []
    for direction in directions:
        perturbed = solve_reference(
            design + 1j * epsilon * direction, problem, _frozen_flows=branches
        )
        active = perturbed.temperature[:, 0, problem.chip_mask]
        objective = (np.log(np.exp(5 * active - shift).sum()) + shift) / 5
        gt = float(objective.imag / epsilon)
        gp = float(perturbed.flow_ml_s.imag / epsilon)
        at = float(np.sum(thermal_gradient * direction))
        ap = float(np.sum(flow_gradient * direction))
        results.append(
            {
                "thermal_adjoint": at,
                "thermal_complex_step": gt,
                "thermal_relative_error": abs(at - gt) / max(abs(at), abs(gt), 1e-30),
                "flow_adjoint": ap,
                "flow_complex_step": gp,
                "flow_relative_error": abs(ap - gp) / max(abs(ap), abs(gp), 1e-30),
            }
        )
        residuals.extend(
            [
                perturbed.diagnostics["pressure_residual"],
                *perturbed.diagnostics["thermal_residual"],
            ]
        )
        residuals.extend(
            [
                perturbed.diagnostics["compensated_pressure_residual"],
                *perturbed.diagnostics["compensated_thermal_residual"],
            ]
        )
    return {
        "epsilon": epsilon,
        "directions": results,
        "all_upwind_branches_frozen_at_real_base": True,
        "minimum_absolute_base_face_flow_m3_s": min(
            float(abs(q).min()) for q in branches
        ),
        "max_thermal_relative_error": max(r["thermal_relative_error"] for r in results),
        "max_flow_relative_error": max(r["flow_relative_error"] for r in results),
        "max_complex_solve_relative_residual": max(residuals),
        "base_diagnostics": base.diagnostics,
    }
