"""Pressure-only evaluation must reproduce the existing model and adjoint."""

import importlib.util

import numpy as np
import pytest
import torch

from waveforge.lab.mcl.contracts import MCLProblem, straight_channel
from waveforge.lab.mcl.stage5_physics import stage5_spec
from waveforge.lab.mcl.stage5_reference import solve_reference
from waveforge.lab.mcl.stage51 import reynolds_samples


def module():
    name = "waveforge.lab.hydraulic_probe"
    assert importlib.util.find_spec(name) is not None, "pressure-only probe missing"
    return __import__(name, fromlist=["*"])


def test_pressure_only_batch_matches_independent_reference_and_uses_no_thermal_solve():
    m = module()
    s = stage5_spec(nx=8, ny=8)
    rng = np.random.default_rng(2026102901)
    designs = np.stack([straight_channel(s), rng.uniform(0.2, 0.8, (8, 8))])
    x = torch.tensor(designs, dtype=torch.float64, requires_grad=True)
    solver = m.HydraulicProbe(s)
    result = solver(x)
    p = MCLProblem(s, np.ones((8, 8), bool), np.full((1, 8, 8), 1 / 64))
    for i, d in enumerate(designs):
        ref = solve_reference(d, p)
        np.testing.assert_allclose(
            result.pressure[i].detach(), ref.pressure, rtol=1e-11, atol=1e-9
        )
        np.testing.assert_allclose(
            result.qx[i].detach(), ref.qx, rtol=1e-11, atol=1e-18
        )
        assert result.flow_ml_s[i].item() == pytest.approx(ref.flow_ml_s, rel=1e-11)
        assert reynolds_samples(result, s)[i].max().item() == pytest.approx(
            ref.diagnostics["max_local_reynolds_estimate"], rel=1e-11
        )
    result.flow_ml_s.log().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert len(solver.linear_diagnostics) == 4
    assert all(
        r["size"] == 64 and r["rhs"] == 1 and r["accepted"]
        for r in solver.linear_diagnostics
    )


@pytest.mark.parametrize("factor", [1, 2])
def test_pump_power_chain_rule_matches_directional_finite_differences(factor):
    m = module()
    s = stage5_spec(nx=8, ny=8)
    fine = m.refined_spec(s, factor)
    assert fine.width_m == s.width_m and fine.port_start == s.port_start * factor
    rng = np.random.default_rng(2026102902)
    x = torch.tensor(rng.uniform(0.2, 0.8, (1, 8, 8)), requires_grad=True)
    solver = m.HydraulicProbe(fine)

    def objective(v):
        r = solver(m.replicated_design(v, factor)).at_pump_power(0.11090112476467685)
        torch.testing.assert_close(
            r.hydraulic_power_w,
            torch.full((1,), 0.11090112476467685, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-14,
        )
        return r.flow_ml_s.log().sum() + reynolds_samples(r, fine).square().mean() / 1e6

    grad = torch.autograd.grad(objective(x), x)[0]
    for _ in range(4):
        direction = torch.tensor(rng.normal(size=x.shape))
        direction /= direction.norm()
        h = 1e-6
        fd = (
            (
                objective(x.detach() + h * direction)
                - objective(x.detach() - h * direction)
            )
            / (2 * h)
        ).item()
        ad = (grad * direction).sum().item()
        assert abs(fd - ad) / max(abs(fd), abs(ad), 1e-12) < 1e-6


def test_closed_plate_pressure_is_finite_and_conserves_tiny_nonzero_darcy_leakage():
    m = module()
    s = stage5_spec(nx=8, ny=8)
    result = m.HydraulicProbe(s)(torch.zeros((1, 8, 8), dtype=torch.float64))
    assert torch.isfinite(result.pressure).all() and result.flow_ml_s.item() > 0
    relative = (result.qin.sum() - result.qout.sum()).abs() / result.qin.sum()
    assert relative.item() < 1e-10


def test_invalid_inputs_and_targets_are_rejected():
    m = module()
    s = stage5_spec(nx=8, ny=8)
    solver = m.HydraulicProbe(s)
    for bad in (
        torch.zeros((1, 8, 8), dtype=torch.float32),
        torch.zeros((0, 8, 8), dtype=torch.float64),
        torch.full((1, 8, 8), float("nan"), dtype=torch.float64),
        torch.ones((1, 7, 8), dtype=torch.float64),
        torch.full((1, 8, 8), 1.01, dtype=torch.float64),
    ):
        with pytest.raises(ValueError):
            solver(bad)
    result = solver(torch.ones((1, 8, 8), dtype=torch.float64))
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            result.at_pump_power(bad)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            m.refined_spec(s, bad)
        with pytest.raises(ValueError):
            m.replicated_design(torch.ones(1, 8, 8), bad)
