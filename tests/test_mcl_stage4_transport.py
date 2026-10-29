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
    mask = np.ones((8, 8), bool)
    p = MCLProblem(s, mask, np.ones((1, 8, 8)) / 64 * 10)
    return p, straight_channel(s)


def test_vanleer_limiter_linear_constant_and_extremum():
    from waveforge.lab.mcl.stage4_transport import limited_slope

    assert limited_slope(
        np.array([1.0, 0, 1, 2]), np.array([1.0, 2, -1, 1])
    ).tolist() == [1.0, 0.0, 0.0, pytest.approx(4 / 3)]


def test_tvd_correction_is_conservative_and_zero_for_constant_temperature():
    from waveforge.lab.mcl.stage4_transport import correction_divergence

    p, d = fixture()
    s = p.spec
    f = solve_flow(base_resistance(d, s), s, 2500)
    const = np.ones((1, 4, 8, 8)) * 10
    np.testing.assert_array_equal(
        correction_divergence(const, d, f, s), np.zeros_like(const)
    )
    field = const.copy()
    field[:, 3] = np.arange(8)[:, None]
    correction = correction_divergence(field, d, f, s)
    assert np.abs(correction).max() > 0
    assert abs(correction.sum()) < 1e-12
    assert np.count_nonzero(correction[:, :3]) == 0


def test_tvd_solution_converges_and_conserves_heat_without_air():
    p, d = fixture()
    r = solve_audit(d, p, h_air=0, scheme="vanleer")
    assert r.diagnostics["transport_converged"]
    assert r.temperature.min() >= -1e-9
    assert r.diagnostics["transport_equation_residual"] < 1e-10
    assert max(r.diagnostics["energy_relative_error"]) < 1e-10


def test_mobility_to_permeability_and_viscosity_units():
    from waveforge.lab.mcl.stage4_nonlinear import forchheimer_beta, water_viscosity

    assert water_viscosity(25) == pytest.approx(0.0008904389816146538, rel=1e-12)
    assert water_viscosity(85) < water_viscosity(25) / 2.5
    # Intrinsic kappa=3.333e-9 m2, sqrt=5.7735e-5 m. beta is 1/m.
    assert forchheimer_beta(np.array([267000.0]), 8.9e-4, 0.55)[0] == pytest.approx(
        9526.2794416
    )


def test_nonlinear_zero_correction_reproduces_linear_operator():
    from waveforge.lab.mcl.stage4_nonlinear import solve_nonlinear

    p, d = fixture()
    base = solve_audit(d, p, h_air=10)
    got = solve_nonlinear(
        d, p, h_air=10, c_f=0, variable_viscosity=False, resistance_ratio=1e4
    )
    assert got["converged"]
    np.testing.assert_allclose(
        got["temperature"], base.temperature, rtol=1e-10, atol=1e-8
    )
    assert got["scenario_flow_ml_s"][0] == pytest.approx(base.flow_ml_s, rel=1e-10)
