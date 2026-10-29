"""Full inherited Stage5 gates; reduced-model admission is NOT CFD validation."""

import numpy as np


def admission(row):
    d = row.get("diagnostics", {})
    reasons = []
    try:
        residuals = [
            d["pressure_residual"],
            d["compensated_pressure_residual"],
            *d["thermal_residual"],
            *d["compensated_thermal_residual"],
        ]
        balances = [d["mass_relative_error"], *d["energy_relative_error"]]
        complete = len(d["thermal_residual"]) > 0 and len(d["thermal_residual"]) == len(
            d["compensated_thermal_residual"]
        ) == len(d["energy_relative_error"])
        numeric = complete and all(np.isfinite(v) and 0 <= v < 1e-12 for v in residuals)
        conservation = all(np.isfinite(v) and 0 <= v < 1e-10 for v in balances)
    except (KeyError, TypeError):
        numeric = conservation = False
    if not row.get("accepted", False) or not numeric:
        reasons.append("numerical_or_geometry_gate")
    if not conservation:
        reasons.append("mass_or_energy_gate")
    for name, value, limit, inclusive in (
        ("local_re", d.get("max_local_reynolds_estimate"), 1200.0, False),
        ("water_c", d.get("peak_open_water_temperature_c"), 60.0, False),
        ("silicon_c", row.get("silicon_peak_c"), 95.0, True),
    ):
        good = (
            isinstance(value, (int, float, np.number))
            and np.isfinite(value)
            and (value <= limit if inclusive else value < limit)
        )
        if name == "local_re":
            good = good and value >= 0
        if not good:
            reasons.append(name)
    return dict(
        admissible=not reasons,
        rejection_reasons=reasons,
        numerical=numeric,
        conservation=conservation,
        physical_validation=False,
    )


def select(rows):
    valid = [r for r in rows if admission(r)["admissible"]]
    return (
        min(valid, key=lambda r: (r["peak_rise_k"], r.get("step", 0)))
        if valid
        else None
    )


def system(raw, uniform):
    if not admission(uniform)["admissible"]:
        raise RuntimeError("Uniform fallback violates inherited Stage5 gates")
    status = admission(raw)
    fallback = not status["admissible"] or raw["peak_rise_k"] >= uniform["peak_rise_k"]
    value = uniform["peak_rise_k"] if fallback else raw["peak_rise_k"]
    return dict(
        **status,
        fallback_used=fallback,
        system_peak_rise_k=value,
        raw_gain_percent=100
        * (uniform["peak_rise_k"] - raw["peak_rise_k"])
        / uniform["peak_rise_k"]
        if status["admissible"]
        else None,
        system_gain_percent=100
        * (uniform["peak_rise_k"] - value)
        / uniform["peak_rise_k"],
    )
