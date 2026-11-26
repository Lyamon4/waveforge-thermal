"""Stage5 physical and geometric contracts, independent of old result claims."""

import numpy as np
import pytest
import torch

from waveforge.lab.mcl.contracts import MCLProblem, straight_channel


def problem():
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    s = stage5_spec(nx=8, ny=8)
    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    return MCLProblem(s, mask, np.stack([mask * w / 16 for w in (10, 20, 30)]))


def test_profile_and_independent_forward():
    from waveforge.lab.mcl.stage5_physics import K_TIM, Stage5Solver
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    p = problem()
    assert K_TIM == 40 and p.spec.channel_width_m == 280e-6
    assert p.spec.pressure_drop_pa == 10000 and p.spec.h_air == 10
    d = straight_channel(p.spec)
    got = Stage5Solver([p])(torch.tensor(d[None], dtype=torch.float64))
    oracle = solve_profile(d, p, k_tim=40)
    np.testing.assert_allclose(
        got.temperature[0].numpy(), oracle.temperature, rtol=1e-11, atol=1e-9
    )
    assert got.flow_ml_s.item() == pytest.approx(oracle.flow_ml_s, rel=1e-12)


def test_coupled_gradient_complex_step():
    from waveforge.lab.mcl.stage5_physics import Stage5Solver
    from waveforge.lab.mcl.stage5_reference import solve_reference

    p = problem()
    rng = np.random.default_rng(2026101805)
    d = rng.uniform(0.2, 0.8, (8, 8))
    x = torch.tensor(d[None], dtype=torch.float64, requires_grad=True)
    solver = Stage5Solver([p])
    got = solver(x)
    active = got.temperature[0, :, 0, solver.chip[0]]
    loss = torch.logsumexp(5 * active.flatten(), 0) / 5
    grad = torch.autograd.grad(loss, x)[0][0].numpy()
    base = solve_reference(d, p)
    shift = float((5 * base.temperature[:, 0, p.chip_mask]).max())
    for _ in range(5):
        v = rng.normal(size=d.shape)
        v /= np.linalg.norm(v)
        cs = solve_reference(
            d + 1j * 1e-20 * v, p, _frozen_flows=(base.qx, base.qy, base.qout)
        )
        value = (
            np.log(np.exp(5 * cs.temperature[:, 0, p.chip_mask] - shift).sum()) + shift
        ) / 5
        actual = value.imag / 1e-20
        expect = (grad * v).sum()
        assert abs(actual - expect) / max(abs(actual), abs(expect), 1e-30) < 1e-6


def test_batch_matches_sequential():
    from waveforge.lab.mcl.stage5_physics import Stage5Solver

    p = problem()
    d = torch.tensor(
        np.random.default_rng(5).uniform(0.2, 0.8, (2, 8, 8)), requires_grad=True
    )
    batched = Stage5Solver([p, p])(d).temperature
    gb = torch.autograd.grad(batched.square().mean(), d)[0]
    for i in range(2):
        single = d[i : i + 1].detach().requires_grad_()
        one = Stage5Solver([p])(single).temperature
        gs = torch.autograd.grad(one.square().mean() / 2, single)[0]
        torch.testing.assert_close(batched[i], one[0], rtol=1e-10, atol=1e-9)
        torch.testing.assert_close(gb[i], gs[0], rtol=1e-9, atol=1e-8)


def test_core_budget_and_all_water_connected():
    from waveforge.lab.mcl.stage5_decoder import binary, fields, initial_parameters
    from waveforge.lab.mcl.vascular import connectivity

    rng = np.random.default_rng(505)
    cases = [initial_parameters()]
    for _ in range(30):
        p = rng.normal(0, 5, 4121)
        cases.append(p)
    p = np.zeros(4121)
    p[:20] = np.tile([-20, 20, -20, 20], 5)
    p[20:25] = [-20, 20, -20, 20, -20]
    cases.append(p)
    for p in cases:
        _score, core, _ = fields(torch.tensor(p[None], dtype=torch.float64))
        assert core.sum() <= 760
        assert core[0, :4, 7:57].all() and core[0, 60:, 7:57].all()
        d = binary(torch.tensor(p[None], dtype=torch.float64))[0].numpy()
        assert d.sum() == 1024 and np.all(d[core[0].numpy()])
        assert connectivity(d)["all_cells_port_connected"]


def test_smooth_auxiliary_gradient_and_volume():
    from waveforge.lab.mcl.stage5_decoder import fields, initial_parameters, project_all

    rng = np.random.default_rng(124)
    p = torch.tensor(
        (initial_parameters() + rng.normal(0, 0.03, 4121))[None], requires_grad=True
    )
    v = torch.tensor(rng.normal(size=p.shape))
    v /= v.norm()
    weights = torch.tensor(rng.normal(size=(1, 64, 64)))

    def f(x):
        score, _, _ = fields(x)
        d = project_all(score, 2.0)
        assert abs(d.sum().item() - 1024) < 1e-8
        return (d * weights).sum()

    g = torch.autograd.grad(f(p), p)[0]
    fd = (f(p.detach() + 1e-5 * v) - f(p.detach() - 1e-5 * v)) / 2e-5
    assert abs(fd.item() - (g * v).sum().item()) / max(abs(fd.item()), 1e-8) < 1e-5


def test_model_four_heads_and_hard_forward():
    from waveforge.lab.mcl.stage5_decoder import ManifoldUNet, density

    model = ManifoldUNet()
    p = model(torch.zeros(2, 13, 64, 64))
    assert p.shape == (2, 4, 4121)
    d, _soft = density(p.flatten(0, 1).double(), beta=8.0, hard=True)
    assert torch.all((d == 0) | (d == 1))
    assert torch.all(d.sum((-2, -1)) == 1024)
    weights = torch.linspace(0, 1, 64)[None, None, :]
    (d * weights).sum().backward()
    assert torch.isfinite(model.geometry.weight.grad).all()


def test_operating_points_and_fallback_are_separate():
    from waveforge.lab.mcl.stage5_evaluation import pressure_for, with_fallback

    q = 11.090112476467686
    assert pressure_for(q, "nominal") == 10000
    assert pressure_for(q, "flow15") == pytest.approx(10000 * 15 / q)
    assert pressure_for(q, "flow16.7") == pytest.approx(10000 * 16.7 / q)
    assert pressure_for(q, "power") == pytest.approx(10000)
    assert pressure_for(4 * q, "power") == pytest.approx(5000)
    raw = {"peak_rise_k": 62.0, "accepted": True}
    base = {"peak_rise_k": 56.0, "accepted": True}
    report = with_fallback(raw, base)
    assert report["raw_peak_rise_k"] == 62
    assert report["system_peak_rise_k"] == 56 and report["fallback_used"]
    assert raw["peak_rise_k"] == 62


def test_smooth_training_uses_actual_fixed_core_gradient():
    from waveforge.lab.mcl.stage5_decoder import density, fields, initial_parameters

    rng = np.random.default_rng(2026101811)
    p = torch.tensor(
        (initial_parameters() + rng.normal(0, 0.03, 4121))[None], requires_grad=True
    )
    v = torch.tensor(rng.normal(size=p.shape))
    v /= v.norm()
    weights = torch.tensor(rng.normal(size=(1, 64, 64)))

    def f(x):
        return (density(x, beta=2.0, hard=False)[0] * weights).sum()

    _, cp, _ = fields(p.detach() + 1e-5 * v)
    _, cm, _ = fields(p.detach() - 1e-5 * v)
    assert torch.equal(cp, cm), "test must stay on one fixed-core branch"
    g = torch.autograd.grad(f(p), p)[0]
    fd = (f(p.detach() + 1e-5 * v) - f(p.detach() - 1e-5 * v)) / 2e-5
    ad = (g * v).sum()
    assert abs(fd.item() - ad.item()) / max(abs(fd.item()), 1e-8) < 1e-5


def test_optimizer_binary_scoring_budget_is_matched(tmp_path):
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.stage5_evaluation import optimize

    task = sample_epyc_task(np.random.default_rng(401018))
    adam = optimize(task, "Manifold-Adam", tmp_path / "adam", steps=3)
    mma = optimize(task, "Manifold-MMA", tmp_path / "mma", steps=3)
    assert adam["evaluations"] == mma["evaluations"] == 3
    assert (
        adam["independent_binary_evaluations"]
        == mma["independent_binary_evaluations"]
        == 3
    )
