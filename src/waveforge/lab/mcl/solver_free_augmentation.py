"""Offline superposition for one fixed geometry/design, never a global Green map."""

import numpy as np

from .contracts import MCLProblem, problem_from_task
from .dual_profile import record_result, solve_dual_profile
from .high_fidelity_eval import refine_problem
from .stage5_evaluation import PUMP_W
from .stage5_physics import stage5_spec


def response_labels(task, design, powers, *, grids=(64, 128)):
    """Exact linear-model targets; independently checked outputs keep full gates."""
    powers = np.asarray(powers, dtype=np.float64)
    if powers.ndim != 3 or powers.shape[1:] != (3, len(task.chiplets)):
        raise ValueError("[workload,3,chiplet] powers required")
    base = problem_from_task(task, stage5_spec())
    basis = np.zeros((len(task.chiplets), 64, 64))
    for i, chip in enumerate(task.chiplets):
        basis[i, chip.x : chip.x + chip.width, chip.y : chip.y + chip.height] = 1 / (
            chip.width * chip.height
        )
    problem = MCLProblem(base.spec, base.chip_mask, basis)
    flat = powers.reshape(-1, len(task.chiplets))
    silicon, water, reynolds = {}, [], []
    numerical = True
    for grid in grids:
        p, d = refine_problem(problem, design, grid // 64)
        results = solve_dual_profile(d, p, pump_power_w=PUMP_W)
        for mode, result in zip(("nominal", "power"), results, strict=True):
            row = record_result(d, p, result)
            numerical &= (
                row["accepted"]
                and row["admission"]["numerical"]
                and row["admission"]["conservation"]
            )
            t = result.temperature
            si = (flat @ t[:, 0, p.chip_mask]).max(1).reshape(len(powers), 3).max(1)
            wa = (flat @ t[:, 3, d.astype(bool)]).max(1).reshape(len(powers), 3).max(1)
            silicon[(grid, mode)] = si
            water.append(wa)
            reynolds.append(result.diagnostics["max_local_reynolds_estimate"])
    primary = max(grids)
    joint_water = np.max(water, axis=0)
    joint_re = max(reynolds)
    metrics = np.stack(
        (
            silicon[(primary, "nominal")] / 50,
            silicon[(primary, "power")] / 50,
            np.full(len(powers), joint_re / 1200),
            joint_water / 35,
        ),
        1,
    )
    valid = (
        (numerical and joint_re < 1200)
        & (np.max(list(silicon.values()), axis=0) <= 70)
        & (joint_water < 35)
    )
    return metrics, np.asarray(valid, dtype=bool)
