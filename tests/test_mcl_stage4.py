"""Physical audit tests with analytic balances and independent old-solver oracle."""

import numpy as np
import pytest

from waveforge.lab.mcl.contracts import MCLProblem, stage12_spec, straight_channel
from waveforge.lab.mcl.reference import solve_reference


def problem():
    spec = stage12_spec(nx=8, ny=8)
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    power = np.stack([mask * x / mask.sum() for x in (10, 20, 30)])
    return MCLProblem(spec, mask, power)


def test_audit_linear_assembly_matches_independent_stage12_oracle():
    from waveforge.lab.mcl.high_fidelity_eval import solve_audit

    p = problem()
    d = straight_channel(p.spec)
    oracle = solve_reference(d, p)
    got = solve_audit(d, p, h_air=500)
    np.testing.assert_allclose(
        got.temperature, oracle.temperature, rtol=1e-11, atol=1e-9
    )
    np.testing.assert_allclose(got.pressure, oracle.pressure, rtol=1e-11, atol=1e-9)
    assert got.flow_ml_s == pytest.approx(oracle.flow_ml_s, rel=1e-11)


def test_zero_air_has_no_hidden_sink_and_all_power_leaves_in_water():
    from waveforge.lab.mcl.high_fidelity_eval import solve_audit

    p = problem()
    r = solve_audit(straight_channel(p.spec), p, h_air=0)
    np.testing.assert_allclose(r.diagnostics["outlet_heat_w"], [10, 20, 30], rtol=1e-10)
    assert r.diagnostics["air_heat_w"] == [0.0, 0.0, 0.0]
    assert max(r.diagnostics["thermal_residual"]) < 1e-12
    assert max(r.diagnostics["energy_relative_error"]) < 1e-10


def test_refinement_preserves_watts_ports_dimensions_and_footprints():
    from waveforge.lab.mcl.high_fidelity_eval import refine_problem

    p = problem()
    d = straight_channel(p.spec)
    q, e = refine_problem(p, d, 2)
    assert e.shape == (16, 16) and e.sum() == 64 and e.mean() == 0.25
    assert q.spec.dx == p.spec.dx / 2 and q.spec.dy == p.spec.dy / 2
    assert q.spec.port_start == 6 and q.spec.port_stop == 10
    np.testing.assert_array_equal(q.power_w.sum((1, 2)), [10.0, 20.0, 30.0])
    np.testing.assert_allclose(q.power_w.reshape(3, 8, 2, 8, 2).sum((2, 4)), p.power_w)


def test_hydraulic_scaling_uses_m3_per_second_and_square_root():
    from waveforge.lab.mcl.high_fidelity_eval import hydraulic_targets

    got = hydraulic_targets(
        flow_ml_s=2.5, pressure_pa=2500, target_ml_s=15, power_w=0.005
    )
    assert got["conductance_m3_s_pa"] == pytest.approx(1e-9)
    assert got["iso_flow_pressure_pa"] == pytest.approx(15000)
    assert got["iso_flow_power_w"] == pytest.approx(0.225)
    assert got["iso_power_pressure_pa"] == pytest.approx(2236.06797749979)
    with pytest.raises(ValueError):
        hydraulic_targets(0, 2500, 15, 0.005)


def test_rejects_zero_sink_without_outflow_and_nonfinite_inputs():
    from waveforge.lab.mcl.high_fidelity_eval import solve_audit

    p = problem()
    d = straight_channel(p.spec)
    with pytest.raises(ValueError):
        solve_audit(d, p, h_air=0, pressure_drop_pa=0)
    with pytest.raises(ValueError):
        solve_audit(d, p, h_air=float("nan"))
    with pytest.raises(ValueError):
        solve_audit(d, p, h_air=-1)


def test_zero_air_rejects_geometry_with_only_wall_leakage_sink():
    from waveforge.lab.mcl.high_fidelity_eval import solve_audit

    p = problem()
    d = np.zeros((8, 8))
    d[3:5, 3:5] = 1
    with pytest.raises(ValueError):
        solve_audit(d, p, h_air=0)
