"""Independent coefficient checks for proposed-profile diagnostics only."""

import numpy as np
import pytest

from waveforge.lab.mcl.contracts import MCLProblem, stage12_spec, straight_channel
from waveforge.lab.mcl.high_fidelity_eval import (
    base_resistance,
    solve_audit,
    solve_flow,
)


def fixture():
    s = stage12_spec(nx=8, ny=8)
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    power = np.stack([mask * watts / 16 for watts in (10, 20, 30)])
    return MCLProblem(s, mask, power)


def test_variable_tim_default_replays_sealed_evaluator():
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    p = fixture()
    d = straight_channel(p.spec)
    got = solve_profile(d, p, k_tim=5)
    oracle = solve_audit(d, p, h_air=500)
    np.testing.assert_allclose(
        got.temperature, oracle.temperature, atol=1e-9, rtol=1e-11
    )
    assert got.flow_ml_s == pytest.approx(oracle.flow_ml_s, rel=1e-12)


def test_tim_changes_lateral_and_both_interface_coefficients():
    from waveforge.lab.mcl.stage5_preflight import thermal_with_tim

    p = fixture()
    s = p.spec
    d = straight_channel(s)
    flow = solve_flow(base_resistance(d, s), s, s.pressure_drop_pa)
    matrix, _ = thermal_with_tim(d, p, flow, k_tim=40)
    n, i = 64, 3 * 8 + 3
    assert matrix[n + i, n + i + 8] == pytest.approx(-40 * s.dy * s.dz[1] / s.dx)
    assert matrix[n + i, n + i + 1] == pytest.approx(-40 * s.dx * s.dz[1] / s.dy)
    assert matrix[i, n + i] == pytest.approx(
        -s.area / (0.00025 / 130 + 2e-6 + 0.00005 / 40)
    )
    assert matrix[n + i, 2 * n + i] == pytest.approx(
        -s.area / (0.00005 / 40 + 2e-6 + 0.0005 / 400)
    )
    assert matrix[n, n] == 1  # Inactive TIM cells stay pinned.


def test_changed_tim_preserves_energy_and_hydraulics():
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    p = fixture()
    d = straight_channel(p.spec)
    first = solve_profile(d, p, k_tim=5)
    second = solve_profile(d, p, k_tim=40)
    assert second.flow_ml_s == first.flow_ml_s
    assert second.scenario_peaks.max() < first.scenario_peaks.max()
    assert max(second.diagnostics["energy_relative_error"]) < 1e-10
    assert max(second.diagnostics["compensated_thermal_residual"]) < 1e-12
    assert second.diagnostics["physical_validation"] is False


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
def test_invalid_tim_rejected(bad):
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    p = fixture()
    with pytest.raises(ValueError):
        solve_profile(straight_channel(p.spec), p, k_tim=bad)
