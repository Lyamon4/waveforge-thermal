"""Stage-one numerical qualification; this is NOT a CFD validation certificate."""

import importlib

import numpy as np
import pytest
import torch


def api():
    try:
        return importlib.import_module("waveforge.lab.mcl")
    except ModuleNotFoundError as exc:
        pytest.fail(f"MCL stage-one implementation is absent: {exc}")


def small_problem(nx=6, ny=8, scenarios=2, **kwargs):
    m = api()
    spec = m.MCLSpec(nx=nx, ny=ny, **kwargs)
    chip = np.ones((nx, ny), dtype=bool)
    power = np.zeros((scenarios, nx, ny), dtype=np.float64)
    power[:, 1:3, 2:4] = np.arange(1, scenarios + 1)[:, None, None] * 3.0
    return m.MCLProblem(spec, chip, power)


def test_uniform_full_port_pressure_and_poiseuille_flow():
    """Wrong port half-cell resistance changes this analytic solution."""
    m = api()
    p = small_problem(nx=8, ny=10, port_start=0, port_stop=10)
    r = m.solve_reference(np.ones((8, 10)), p)
    expected = 15000 * (1 - (np.arange(8) + 0.5) / 8)
    np.testing.assert_allclose(
        r.pressure, np.broadcast_to(expected[:, None], (8, 10)), rtol=1e-12, atol=1e-8
    )
    expected_flow = (1e-6 / (12 * 8.9e-4)) * (0.072 * 0.001) / 0.0754 * 15000
    assert r.flow_ml_s == pytest.approx(expected_flow * 1e6, rel=1e-12)
    assert r.diagnostics["pressure_residual"] < 1e-12
    assert r.diagnostics["mass_relative_error"] < 1e-12


def test_reference_energy_and_power_scaling():
    """Incorrect upwind signs or missing outlet loss break the balance."""
    m = api()
    p = small_problem()
    d = np.linspace(0.12, 0.89, 48).reshape(6, 8)
    r = m.solve_reference(d, p)
    assert r.temperature.shape == (2, 4, 6, 8)
    assert np.min(r.temperature) >= -1e-9
    np.testing.assert_allclose(
        r.temperature[1], 2 * r.temperature[0], rtol=2e-12, atol=1e-9
    )
    assert max(r.diagnostics["energy_relative_error"]) < 1e-12
    assert max(r.diagnostics["thermal_residual"]) < 1e-12


def test_zero_power_is_zero_temperature_even_with_flow():
    m = api()
    p = small_problem()
    p = m.MCLProblem(p.spec, p.chip_mask, np.zeros_like(p.power_w))
    r = m.solve_reference(np.full((6, 8), 0.25), p)
    assert np.count_nonzero(r.temperature) == 0
    assert r.flow_ml_s > 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_invalid_design_rejected(bad):
    m = api()
    d = np.full((6, 8), 0.25)
    d[2, 3] = bad
    with pytest.raises(ValueError, match="design"):
        m.solve_reference(d, small_problem())


def test_task_adapter_preserves_xy_and_all_nine_powers():
    from waveforge.lab.contracts import Chiplet, ChipletTask

    m = api()
    chips = (Chiplet(3, 9, 8, 7), Chiplet(26, 26, 12, 12))
    task = ChipletTask(chips, ((60.0, 50.0), (30.0, 50.0)))
    p = m.problem_from_task(task)
    assert p.chip_mask[3, 9] and p.chip_mask[10, 15]
    assert not p.chip_mask[9, 3]
    assert np.count_nonzero(p.chip_mask) == 56 + 144
    np.testing.assert_allclose(p.power_w.sum((1, 2)), [110, 80], rtol=0, atol=1e-12)
    assert p.power_w[0, 3, 9] == pytest.approx(60 / 56)


def test_straight_control_has_exact_budget_and_connected_ports():
    m = api()
    d = m.straight_channel(m.MCLSpec())
    assert d.shape == (64, 64)
    assert np.sum(d) == 1024
    assert np.all(d[:, 24:40] == 1)
    assert np.all(d[:, :24] == 0)


def test_reduced_model_warns_for_high_reynolds_and_solid_leakage():
    m = api()
    p = small_problem(nx=8, ny=8, port_start=0, port_stop=8)
    open_result = m.solve_reference(np.ones((8, 8)), p)
    closed_result = m.solve_reference(np.zeros((8, 8)), p)
    assert open_result.diagnostics["inlet_reynolds_estimate"] > 10000
    assert open_result.diagnostics["physical_validation"] is False
    assert "high_reynolds" in open_result.diagnostics["warnings"]
    assert closed_result.flow_ml_s > 0
    assert closed_result.flow_ml_s / open_result.flow_ml_s == pytest.approx(
        1e-5, rel=1e-10
    )


def test_nonsymmetric_sparse_adjoint_matches_dense_autograd():
    """Using A instead of A.T in backward fails this non-symmetric fixture."""
    api()
    from waveforge.lab.mcl.linear import sparse_solve

    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    values = torch.tensor(
        [[3.0, -1.0, -2.0, 4.0], [2.0, -0.4, -0.7, 3.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    rhs = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0]], [[2.0, 1.0], [4.0, 6.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    got = sparse_solve(values, rhs, rows, cols, 2)
    expected = torch.linalg.solve(values.reshape(2, 2, 2), rhs)
    weights = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[-2.0, 1.0], [0.2, 4.0]]], dtype=torch.float64
    )
    actual_grad = torch.autograd.grad(
        (got * weights).sum(), (values, rhs), retain_graph=True
    )
    expect_grad = torch.autograd.grad((expected * weights).sum(), (values, rhs))
    torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-13)
    for actual, expect in zip(actual_grad, expect_grad, strict=True):
        torch.testing.assert_close(actual, expect, rtol=1e-12, atol=1e-12)


def test_torch_reference_and_coupled_gradient():
    """Detaching pressure or omitting flow dependence breaks finite differences."""
    m = api()
    p = small_problem()
    d = (
        torch.linspace(0.15, 0.85, 48, dtype=torch.float64)
        .reshape(1, 6, 8)
        .requires_grad_()
    )
    solver = m.MCLSolver([p])
    out = solver(d)
    ref = m.solve_reference(d.detach().numpy()[0], p)
    np.testing.assert_allclose(
        out.temperature.detach().numpy()[0], ref.temperature, rtol=1e-10, atol=1e-8
    )
    np.testing.assert_allclose(
        out.pressure.detach().numpy()[0], ref.pressure, rtol=1e-11, atol=1e-8
    )
    loss = torch.logsumexp(out.temperature[:, :, 0].flatten(), dim=0)
    (gradient,) = torch.autograd.grad(loss, d)
    direction = torch.cos(torch.arange(48, dtype=torch.float64)).reshape_as(d)
    direction /= direction.norm()
    exact = (gradient * direction).sum().item()
    eps = 1e-6

    def reference_loss(array):
        from scipy.special import logsumexp

        return logsumexp(m.solve_reference(array, p).temperature[:, 0])

    centered = (
        reference_loss((d + eps * direction).detach().numpy()[0])
        - reference_loss((d - eps * direction).detach().numpy()[0])
    ) / (2 * eps)
    assert exact == pytest.approx(centered, rel=1e-6, abs=1e-7)
    assert {r["phase"] for r in solver.linear_diagnostics} == {"forward", "adjoint"}
    assert len(solver.linear_diagnostics) == 4


def test_batch_matches_single_forward_and_gradient():
    m = api()
    p1 = small_problem()
    p2 = m.MCLProblem(p1.spec, p1.chip_mask, np.flip(p1.power_w, axis=2).copy())
    d = (
        torch.linspace(0.11, 0.91, 96, dtype=torch.float64)
        .reshape(2, 6, 8)
        .requires_grad_()
    )
    batched = m.MCLSolver([p1, p2])(d)
    (gb,) = torch.autograd.grad(batched.temperature.square().mean(), d)
    for i, p in enumerate([p1, p2]):
        single_d = d[i : i + 1].detach().clone().requires_grad_()
        single = m.MCLSolver([p])(single_d)
        (gs,) = torch.autograd.grad(single.temperature.square().mean() / 2, single_d)
        torch.testing.assert_close(
            batched.temperature[i : i + 1], single.temperature, rtol=1e-11, atol=1e-8
        )
        torch.testing.assert_close(gb[i : i + 1], gs, rtol=1e-10, atol=1e-8)


def test_gradcheck_small_interior_design():
    m = api()
    p = small_problem(nx=4, ny=4, scenarios=1)
    d = (
        torch.linspace(0.2, 0.8, 16, dtype=torch.float64)
        .reshape(1, 4, 4)
        .requires_grad_()
    )
    solver = m.MCLSolver([p])
    assert torch.autograd.gradcheck(
        lambda x: solver(x).temperature[:, :, 0],
        (d,),
        eps=1e-6,
        atol=2e-5,
        rtol=1e-6,
        fast_mode=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local CUDA unavailable")
def test_cuda_tensor_bridge_is_explicit_and_matches_cpu():
    m = api()
    p = small_problem(nx=4, ny=4, scenarios=1)
    d = torch.linspace(0.17, 0.83, 16, dtype=torch.float64).reshape(1, 4, 4)
    cpu_d = d.clone().requires_grad_()
    gpu_d = d.cuda().requires_grad_()
    cpu = m.MCLSolver([p])(cpu_d)
    gpu_solver = m.MCLSolver([p], device="cuda")
    gpu = gpu_solver(gpu_d)
    assert gpu_solver.backend == "scipy_lu"
    torch.testing.assert_close(
        gpu.temperature.cpu(), cpu.temperature, rtol=1e-10, atol=1e-8
    )
    (gc,) = torch.autograd.grad(cpu.temperature.sum(), cpu_d)
    (gg,) = torch.autograd.grad(gpu.temperature.sum(), gpu_d)
    torch.testing.assert_close(gg.cpu(), gc, rtol=1e-9, atol=1e-7)


def test_full_grid_matches_submitted_reference():
    """Compare against the supplied independent code, not our own assembly helpers."""
    import runpy
    from pathlib import Path

    from waveforge.lab.contracts import Chiplet, ChipletTask

    fixture = Path(__file__).parent / "fixtures" / "mcl_submitted_reference.py"
    assert fixture.is_file(), "verbatim submitted numerical reference is missing"
    supplied = runpy.run_path(str(fixture))["solve_mcl_bp"]
    coords = [(x, y) for y in (10, 47) for x in (4, 20, 36, 52)]
    chips = (*(Chiplet(x, y, 8, 7) for x, y in coords), Chiplet(26, 26, 12, 12))
    modes = [[38.75] * 8, [58.125] * 4 + [19.375] * 4]
    task = ChipletTask(chips, tuple(tuple([*row, 50.0]) for row in modes))
    m = api()
    p = m.problem_from_task(task)
    d = m.straight_channel(p.spec)
    expected = supplied(d, coords, modes)
    actual = m.solve_reference(d, p)
    np.testing.assert_allclose(actual.pressure, expected[3], rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(
        actual.temperature, expected[4].T.reshape(2, 4, 64, 64), rtol=1e-9, atol=1e-8
    )
    assert actual.flow_ml_s == pytest.approx(expected[2], rel=1e-9)


@pytest.mark.parametrize("gradient_error,expected", [(1e-8, True), (1e-3, False)])
def test_qualification_does_not_claim_physical_validation(gradient_error, expected):
    from waveforge.experiments.qualify_mcl_stage1 import qualification_verdict

    result = qualification_verdict(
        1e-14, 1e-10, gradient_error, max_mass_error=1e-12, max_energy_error=1e-12
    )
    assert result["numerically_qualified"] is expected
    assert result["physically_validated"] is False
    assert result["training_authorized_by_this_gate"] is False


def test_control_numerical_failure_is_recorded_not_hidden(monkeypatch):
    from waveforge.experiments import qualify_mcl_stage1 as q
    from waveforge.lab.mcl.linear import NumericalQualificationError

    records = [{"phase": "forward", "relative_residual": 5.312345e-12}]

    def reject(*args, **kwargs):
        raise NumericalQualificationError("forward", 5.312345e-12, records)

    monkeypatch.setattr(q, "_timed_torch", reject)
    arrays, seconds, records, error = q.check_control(None, None)
    assert arrays is None and seconds is None
    assert records[-1]["relative_residual"] == 5.312345e-12
    assert "unqualified" in error

    def unrelated(*args, **kwargs):
        raise RuntimeError("unrelated implementation error")

    monkeypatch.setattr(q, "_timed_torch", unrelated)
    with pytest.raises(RuntimeError, match="unrelated"):
        q.check_control(None, None)


@pytest.mark.parametrize("mass,energy", [(1e-3, 1e-12), (1e-12, 1e-3)])
def test_qualification_rejects_failed_conservation(mass, energy):
    from waveforge.experiments.qualify_mcl_stage1 import qualification_verdict

    result = qualification_verdict(
        1e-14, 1e-10, 1e-8, max_mass_error=mass, max_energy_error=energy
    )
    assert result["numerically_qualified"] is False


def test_failed_sparse_solve_preserves_full_precision_records(monkeypatch):
    from waveforge.lab.mcl import linear

    monkeypatch.setattr(linear, "_residual", lambda *args: 2.123456789e-12)
    records = []
    with pytest.raises(linear.NumericalQualificationError) as caught:
        linear.sparse_solve(
            torch.ones((1, 2), dtype=torch.float64),
            torch.ones((1, 2, 1), dtype=torch.float64),
            np.arange(2),
            np.arange(2),
            2,
            records,
        )
    assert records[0]["relative_residual"] == 2.123456789e-12
    assert caught.value.records[0] == records[0]


def test_pressure_branch_flow_gradient_independently():
    m = api()
    p = small_problem()
    d = torch.linspace(0.17, 0.83, 48, dtype=torch.float64).reshape(1, 6, 8)
    d.requires_grad_()
    out = m.MCLSolver([p])(d)
    gradient = torch.autograd.grad(out.flow_ml_s.sum(), d)[0]
    direction = torch.sin(torch.arange(48, dtype=torch.float64)).reshape_as(d)
    direction /= direction.norm()
    eps = 1e-6
    plus = m.solve_reference((d + eps * direction).detach().numpy()[0], p)
    minus = m.solve_reference((d - eps * direction).detach().numpy()[0], p)
    fd = (plus.flow_ml_s - minus.flow_ml_s) / (2 * eps)
    assert float((gradient * direction).sum()) == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_hot_copper_is_not_misreported_as_hot_water():
    m = api()
    p = small_problem(nx=8, ny=8)
    p = m.MCLProblem(p.spec, p.chip_mask, 100 * p.power_w)
    design = m.straight_channel(p.spec)
    result = m.solve_reference(design, p)
    assert result.temperature[:, 3, design == 0].max() + 25 > 100
    assert result.temperature[:, 3, design == 1].max() + 25 < 100
    assert (
        "check_water_phase_and_temperature_dependent_properties"
        not in result.diagnostics["warnings"]
    )
