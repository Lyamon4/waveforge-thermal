"""Conservative limited interior reconstruction; boundary faces stay upwind.

This is a spatial-sensitivity probe, not a claim of global second-order
convergence in the pixelated multiscale geometry.
"""

import numpy as np


def limited_slope(left, right):
    same = left * right > 0
    return np.divide(
        2 * left * right, left + right, out=np.zeros_like(left, dtype=float), where=same
    )


def correction_divergence(temperature, design, flow, spec):
    water = temperature[:, 3]
    opened = design == 1
    sx = np.zeros_like(water)
    sy = np.zeros_like(water)
    sx[:, 1:-1] = limited_slope(
        water[:, 1:-1] - water[:, :-2], water[:, 2:] - water[:, 1:-1]
    )
    sy[:, :, 1:-1] = limited_slope(
        water[:, :, 1:-1] - water[:, :, :-2], water[:, :, 2:] - water[:, :, 1:-1]
    )
    sx[:, 1:-1] *= opened[:-2] & opened[1:-1] & opened[2:]
    sy[:, :, 1:-1] *= opened[:, :-2] & opened[:, 1:-1] & opened[:, 2:]
    fx = spec.rho_cp * flow.qx * 0.5 * np.where(flow.qx >= 0, sx[:, :-1], -sx[:, 1:])
    fy = (
        spec.rho_cp
        * flow.qy
        * 0.5
        * np.where(flow.qy >= 0, sy[:, :, :-1], -sy[:, :, 1:])
    )
    fx *= opened[:-1] & opened[1:]
    fy *= opened[:, :-1] & opened[:, 1:]
    result = np.zeros_like(temperature)
    result[:, 3, :-1] += fx
    result[:, 3, 1:] -= fx
    result[:, 3, :, :-1] += fy
    result[:, 3, :, 1:] -= fy
    return result


def solve_limited(
    lu, matrix, rhs, initial, design, flow, spec, max_iterations=1000, tolerance=1e-10
):
    temperature = initial.copy()
    shape = temperature.shape
    denom = np.maximum(np.linalg.norm(rhs, axis=0), 1e-30)
    residual = float("inf")
    for iteration in range(max_iterations + 1):
        corr = (
            correction_divergence(temperature, design, flow, spec)
            .reshape(shape[0], -1)
            .T
        )
        flat = temperature.reshape(shape[0], -1).T
        residual = float(
            np.max(np.linalg.norm(matrix @ flat + corr - rhs, axis=0) / denom)
        )
        if residual < tolerance:
            break
        if iteration == max_iterations:
            break
        # Fixed-point iteration does not need compensated corrections at every
        # intermediate step. Reuse the row-scaled factor and certify the final
        # linear solve with compensated residual below. Equations unchanged.
        proposal = lu._solve(rhs - corr, transpose=False).T.reshape(shape)
        temperature = 0.35 * temperature + 0.65 * proposal
    # An extra linear solution measures the inner solver, not the nonlinear
    # transport residual. The reported field itself remains the converged iterate.
    effective_rhs = rhs - corr
    inner = lu.solve(effective_rhs)
    diag = dict(
        transport_converged=residual < tolerance,
        transport_iterations=iteration,
        transport_equation_residual=residual,
        transport_minimum_temperature_rise=float(temperature.min()),
        transport_scheme="Van Leer limited interior; upwind boundaries",
        inner_linear_residual=(
            np.linalg.norm(matrix @ inner - effective_rhs, axis=0)
            / np.maximum(np.linalg.norm(effective_rhs, axis=0), 1e-30)
        ).tolist(),
        inner_compensated_residual=lu.relative_residual(inner, effective_rhs).tolist(),
    )
    return temperature, diag
