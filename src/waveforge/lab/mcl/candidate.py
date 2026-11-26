"""Common numerical and local-Re rejection contract for EVERY candidate method."""

import numpy as np


def candidate_status(diagnostics):
    re = diagnostics["max_local_reynolds_estimate"]
    reynolds_failed = not np.isfinite(re) or re < 0 or re >= 500
    residuals = [diagnostics["pressure_residual"], *diagnostics["thermal_residual"]]
    complete = (
        "compensated_pressure_residual" in diagnostics
        and "compensated_thermal_residual" in diagnostics
        and len(diagnostics["thermal_residual"]) > 0
        and len(diagnostics["thermal_residual"])
        == len(diagnostics["compensated_thermal_residual"])
        and len(diagnostics["energy_relative_error"])
        == len(diagnostics["thermal_residual"])
    )
    if complete:
        residuals += [
            diagnostics["compensated_pressure_residual"],
            *diagnostics["compensated_thermal_residual"],
        ]
    numerical = complete and all(np.isfinite(r) and 0 <= r < 1e-12 for r in residuals)
    balances = [
        diagnostics["mass_relative_error"],
        *diagnostics["energy_relative_error"],
    ]
    conservation = all(np.isfinite(r) and 0 <= r < 1e-10 for r in balances)
    return {
        "accepted": bool(numerical and conservation and not reynolds_failed),
        "reynolds_contract_violated": bool(reynolds_failed),
        "numerically_accepted": bool(numerical),
        "conservation_accepted": bool(conservation),
        "physical_validation": False,
    }
