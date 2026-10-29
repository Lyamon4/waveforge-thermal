"""Stage 1.1 patch tests. Passing software tests is NOT a qualification verdict."""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")


def micro_problem(n=6):
    from waveforge.lab.mcl import MCLProblem
    from waveforge.lab.mcl.contracts import stage11_spec

    spec = stage11_spec(nx=n, ny=n, port_start=0, port_stop=n)
    power = np.zeros((2, n, n))
    power[:, 1:3, 1:3] = np.array([2.0, 3.0])[:, None, None]
    return MCLProblem(spec, np.ones((n, n), bool), power)


def test_profile_is_explicit_and_preserves_stage1():
    from waveforge.lab.mcl.contracts import MCLSpec, stage11_spec

    old, new = MCLSpec(), stage11_spec()
    assert old.pressure_drop_pa == 15000 and old.h_fluid == 15000
    assert old.channel_width_m is None and not old.scaled_refinement
    assert new.channel_width_m == 2e-4 and new.pressure_drop_pa == 10000
    assert new.h_fluid == 25830 and new.scaled_refinement
    assert new.mobility_length_m == 2e-4


def test_scaled_refined_adjoint_nonsymmetric_matches_dense():
    from waveforge.lab.mcl.linear import sparse_solve

    rows = np.repeat(np.arange(3), 3)
    cols = np.tile(np.arange(3), 3)
    a = torch.tensor(
        [[[3e-3, -1e-3, 0], [-2, 4, -1], [0, -1e3, 5e3]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    b = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64, requires_grad=True
    )
    records = []
    x = sparse_solve(a.flatten(1), b, rows, cols, 3, records, scaled_refinement=True)
    expected = torch.linalg.solve(a, b)
    actual_grad = torch.autograd.grad(x.square().sum(), (a, b), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.square().sum(), (a, b))
    torch.testing.assert_close(x, expected, rtol=1e-13, atol=1e-12)
    for got, want in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-10)
    assert len(records) == 2
    assert all(r["refinement_steps"] == 1 for r in records)
    assert all(r["residual_system"] == "original_unscaled" for r in records)


def test_new_profile_flow_scaling_and_reference_agreement():
    from waveforge.lab.mcl import MCLSolver, solve_reference

    p = micro_problem()
    design = np.ones((6, 6))
    ref = solve_reference(design, p)
    expected = (2e-4) ** 2 / (12 * 8.9e-4) * 0.072 * 0.001 / 0.0754 * 10000 * 1e6
    assert ref.flow_ml_s == pytest.approx(expected, rel=1e-11)
    assert ref.diagnostics["inlet_reynolds_estimate"] < 500
    result = MCLSolver([p])(torch.tensor(design[None]))
    np.testing.assert_allclose(
        result.temperature.numpy()[0], ref.temperature, rtol=1e-11, atol=1e-8
    )


def test_complex_step_coupled_and_flow_gradients_all_directions():
    from waveforge.lab.mcl import MCLSolver
    from waveforge.lab.mcl.complex_step import complex_step_check

    p = micro_problem(4)
    d = torch.linspace(0.2, 0.7, 16, dtype=torch.float64).reshape(1, 4, 4)
    d.requires_grad_()
    out = MCLSolver([p])(d)
    loss = torch.logsumexp(5 * out.temperature[0, :, 0].flatten(), 0) / 5
    gt = torch.autograd.grad(loss, d, retain_graph=True)[0].numpy()[0]
    gp = torch.autograd.grad(out.flow_ml_s.sum(), d)[0].numpy()[0]
    # ALL coordinate directions on this small case, not only high-signal ones.
    directions = np.eye(16).reshape(16, 4, 4)
    check = complex_step_check(d.detach().numpy()[0], p, directions, gt, gp)
    assert len(check["directions"]) == 16
    assert check["max_thermal_relative_error"] < 1e-6
    assert check["max_flow_relative_error"] < 1e-6


def test_segment_distance_is_euclidean_and_has_gradients():
    from waveforge.lab.mcl.vascular import segment_distances

    y = torch.full((1, 1, 6), 31.5, dtype=torch.float64, requires_grad=True)
    d = segment_distances(y)
    expected = (torch.arange(64, dtype=torch.float64) - 31.5).abs()
    torch.testing.assert_close(d[0, 0, 17], expected)
    g = torch.autograd.grad(d[0, 0, 17, 34], y)[0]
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_decoder_exact_budget_and_audit_does_not_hide_islands():
    from waveforge.lab.mcl.vascular import binary_topk, connectivity, vascular_scores

    y = torch.full((1, 6, 4), 31.5, dtype=torch.float64)
    w = torch.tensor([[4.0, 1, 1, 1, 1, 1]], dtype=torch.float64)
    scores, distances = vascular_scores(y, w)
    binary = binary_topk(scores)[0].numpy()
    assert binary.sum() == 1024
    audit = connectivity(binary, distances[0].numpy())
    assert audit["inlet_outlet_connected"]
    assert audit["all_cells_port_connected"]
    island = np.zeros((64, 64), bool)
    island[:, 31] = True
    island[10, 10] = True
    audit = connectivity(island)
    assert audit["inlet_outlet_connected"]
    assert not audit["all_cells_port_connected"]


def test_qualification_rejects_every_failed_gate():
    from waveforge.experiments.qualify_mcl_stage11 import verdict

    good = dict(numerical=True, gradients=True, connectivity=True, hydraulic=True)
    assert verdict(**good)["stage11_qualified"]
    for field in good:
        assert not verdict(**(good | {field: False}))["stage11_qualified"]
    assert not verdict(**good)["physically_validated"]


def test_diagnostic_continuation_records_failure_without_relaxing_gate(monkeypatch):
    from waveforge.lab.mcl import linear

    monkeypatch.setattr(linear, "_residual", lambda *args: 4e-12)
    records = []
    values = torch.ones((1, 2), dtype=torch.float64, requires_grad=True)
    rhs = torch.ones((1, 2, 1), dtype=torch.float64)
    with pytest.raises(linear.NumericalQualificationError):
        linear.sparse_solve(values, rhs, np.arange(2), np.arange(2), 2)
    result = linear.sparse_solve(
        values, rhs, np.arange(2), np.arange(2), 2, records, diagnostic_only=True
    )
    torch.autograd.grad(result.sum(), values)
    assert len(records) == 2
    assert all(not r["accepted"] and r["diagnostic_only"] for r in records)
    assert all(r["relative_residual"] == 4e-12 for r in records)


def test_core_size_bound_counterexample_is_not_hidden():
    from waveforge.lab.mcl.vascular import binary_topk, connectivity, vascular_scores

    y = torch.tensor(
        [
            [
                [
                    25.968905308575632,
                    5.391093599200173,
                    15.660385617690881,
                    38.901520597725145,
                ],
                [
                    50.18139477062199,
                    25.023868334482685,
                    5.753639125682014,
                    19.743914339913026,
                ],
                [
                    25.62784376189087,
                    47.76873609187301,
                    14.838378926278818,
                    52.213372427403925,
                ],
                [
                    11.14620493578705,
                    30.816249845544874,
                    54.72452544217196,
                    11.347144641184688,
                ],
                [
                    56.85939755228229,
                    4.277826493461916,
                    46.990503191340274,
                    33.2421127298677,
                ],
                [
                    35.002684413568545,
                    58.05007997579336,
                    11.05324259355822,
                    54.814831421978475,
                ],
            ]
        ],
        dtype=torch.float64,
    )
    widths = torch.tensor(
        [
            [
                2.509663848638075,
                2.7918820361121495,
                3.864253715538921,
                3.2698096525625786,
                2.662144832822581,
                2.012415902489156,
            ]
        ],
        dtype=torch.float64,
    )
    scores, ds = vascular_scores(y, widths)
    audit = connectivity(binary_topk(scores)[0].numpy(), ds[0].numpy())
    assert audit["core_cells"] == 1027 and audit["core_exceeds_budget"]
    assert audit["omitted_core_cells"] >= 3
    assert audit["all_cells_port_connected"]  # Omission != proven disconnection.


def test_connected_neck_breaks_universal_re500_claim_at_exact_budget():
    from waveforge.lab.mcl import MCLProblem, solve_reference, straight_channel
    from waveforge.lab.mcl.contracts import stage11_spec
    from waveforge.lab.mcl.vascular import connectivity

    spec = stage11_spec()
    design = straight_channel(spec)
    design[32, 24:40] = 0
    design[32, 31] = 1
    design[10:25, 23] = 1
    p = MCLProblem(spec, np.ones((64, 64), bool), np.zeros((1, 64, 64)))
    result = solve_reference(design, p)
    assert design.sum() == 1024 and connectivity(design)["all_cells_port_connected"]
    assert result.diagnostics["inlet_reynolds_estimate"] < 500
    assert result.diagnostics["max_local_reynolds_estimate"] > 1500


def test_stage11_diagnostic_batch_and_cuda_bridge_match_sequential():
    from waveforge.lab.mcl import MCLSolver

    p = micro_problem(4)
    d = (
        torch.linspace(0.22, 0.71, 32, dtype=torch.float64)
        .reshape(2, 4, 4)
        .requires_grad_()
    )
    solver = MCLSolver([p, p], diagnostic_only=True)
    batched = solver(d)
    gb = torch.autograd.grad(batched.temperature.square().sum(), d)[0]
    for i in range(2):
        ds = d[i : i + 1].detach().clone().requires_grad_()
        single = MCLSolver([p], diagnostic_only=True)(ds)
        gs = torch.autograd.grad(single.temperature.square().sum(), ds)[0]
        torch.testing.assert_close(batched.temperature[i : i + 1], single.temperature)
        torch.testing.assert_close(gb[i : i + 1], gs)
    if torch.cuda.is_available():
        dg = d.detach().cuda().requires_grad_()
        gpu = MCLSolver([p, p], device="cuda", diagnostic_only=True)(dg)
        gg = torch.autograd.grad(gpu.temperature.square().sum(), dg)[0]
        torch.testing.assert_close(
            gpu.temperature.cpu(), batched.temperature, rtol=1e-10, atol=1e-8
        )
        torch.testing.assert_close(gg.cpu(), gb, rtol=1e-9, atol=1e-7)
    # This is a batch equivalence test, NOT a qualification of these adjoints.
    assert all(
        r["accepted"] == (r["relative_residual"] < 1e-12)
        for r in solver.linear_diagnostics
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_reject_bad_microchannel_width(bad):
    from waveforge.lab.mcl.contracts import MCLSpec

    with pytest.raises(ValueError, match="channel_width"):
        MCLSpec(channel_width_m=bad)


def test_segment_diagonal_distance_matches_independent_numpy_and_gradient():
    from waveforge.lab.mcl.vascular import segment_distances

    y = torch.tensor(
        [[[31.5, 8.0, 52.0, 13.0, 50.0, 31.5]]], dtype=torch.float64, requires_grad=True
    )
    got = segment_distances(y)
    x = np.array([0.0, 12, 25, 38, 51, 63])
    xx, yy = np.indices((64, 64), dtype=float)
    expected = np.full((64, 64), np.inf)
    yn = y.detach().numpy()[0, 0]
    for k in range(5):
        vx, vy = x[k + 1] - x[k], yn[k + 1] - yn[k]
        t = np.clip(((xx - x[k]) * vx + (yy - yn[k]) * vy) / (vx * vx + vy * vy), 0, 1)
        expected = np.minimum(
            expected, np.hypot(xx - x[k] - t * vx, yy - yn[k] - t * vy)
        )
    np.testing.assert_allclose(
        got.detach().numpy()[0, 0], expected, rtol=1e-13, atol=1e-13
    )
    assert torch.autograd.gradcheck(
        lambda v: segment_distances(v)[0, 0, 17, 34],
        (y,),
        eps=1e-6,
        atol=1e-7,
        rtol=1e-6,
    )


def test_reference_diagnostic_records_cover_every_rhs_and_rejection():
    from types import SimpleNamespace

    from waveforge.experiments.qualify_mcl_stage11 import reference_records

    result = SimpleNamespace(
        diagnostics={
            "pressure_residual": 2e-15,
            "thermal_residual": [3e-13, 2e-12, 4e-13],
        }
    )
    records = reference_records(result, "direction_plus")
    assert len(records) == 4
    assert [r["relative_residual"] for r in records] == [2e-15, 3e-13, 2e-12, 4e-13]
    assert [r["accepted"] for r in records] == [True, True, False, True]
