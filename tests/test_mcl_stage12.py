"""Stage1.2 tests; independent arithmetic oracles and explicit qualification gates."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
import scipy.sparse as sp
import torch


def test_compensated_residual_retains_cancelled_unit_and_tiny_complex_part():
    from waveforge.lab.mcl.compensated import CompensatedCSR

    matrix = sp.csr_matrix([[1e16, 1.0, -1e16], [2.0, -3.0, 4.0]])
    x = np.ones((3, 1))
    rhs = np.zeros((2, 1))
    got = CompensatedCSR(matrix).residual(x, rhs)
    np.testing.assert_array_equal(got, np.array([[-1.0], [-3.0]]))
    xc = x.astype(complex) * (1 + 1e-20j)
    complex_got = CompensatedCSR(matrix.astype(complex)).residual(
        xc, rhs.astype(complex)
    )
    np.testing.assert_allclose(complex_got.real, got, rtol=0, atol=0)
    np.testing.assert_allclose(complex_got.imag, got * 1e-20, rtol=1e-15, atol=0)


def test_two_prod_error_matches_decimal_not_just_rounded_product():
    from waveforge.lab.mcl.compensated import two_prod, two_sum

    a, b = 1.0 + 2**-27, 1.0 - 2**-27
    high, low = two_prod(np.array(a), np.array(b))
    with localcontext() as ctx:
        ctx.prec = 100
        assert Decimal(float(high)) + Decimal(float(low)) == Decimal(a) * Decimal(b)
    high, low = two_sum(np.array(1e16), np.array(1.0))
    assert high == 1e16 and low == 1.0


def test_complex_sparse_residual_against_independent_decimal_oracle():
    from waveforge.lab.mcl.compensated import CompensatedCSR

    rng = np.random.default_rng(127)
    a = rng.normal(size=(5, 7)) + 1j * rng.normal(size=(5, 7))
    x = rng.normal(size=(7, 2)) + 1j * rng.normal(size=(7, 2)) * 1e-20
    b = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    got = CompensatedCSR(sp.csr_matrix(a)).residual(x, b)
    expected = np.empty_like(got)
    with localcontext() as ctx:
        ctx.prec = 120
        for i in range(5):
            for j in range(2):
                real = Decimal(float(b[i, j].real))
                imag = Decimal(float(b[i, j].imag))
                for k in range(7):
                    ar, ai = Decimal(float(a[i, k].real)), Decimal(float(a[i, k].imag))
                    xr, xi = Decimal(float(x[k, j].real)), Decimal(float(x[k, j].imag))
                    real -= ar * xr - ai * xi
                    imag -= ar * xi + ai * xr
                expected[i, j] = complex(float(real), float(imag))
    np.testing.assert_array_equal(got, expected)


def test_double_double_residual_random_sparse_real_and_complex_against_decimal():
    from waveforge.lab.mcl.compensated import CompensatedCSR

    rng = np.random.default_rng(25)
    matrix = rng.normal(size=(5, 7)) * 10 ** rng.integers(-6, 7, size=(5, 7)).astype(
        float
    )
    matrix[matrix < 0] = 0
    x = rng.normal(size=(7, 2))
    b = rng.normal(size=(5, 2))
    got = CompensatedCSR(sp.csr_matrix(matrix)).residual(x, b)
    with localcontext() as ctx:
        ctx.prec = 100
        expected = np.array(
            [
                [
                    float(
                        Decimal(b[i, j])
                        - sum(
                            Decimal(matrix[i, k]) * Decimal(x[k, j]) for k in range(7)
                        )
                    )
                    for j in range(2)
                ]
                for i in range(5)
            ]
        )
    np.testing.assert_array_equal(got, expected)


def test_stage12_refinement_transpose_and_complex_equivalence():
    from waveforge.lab.mcl.factorization import Factorization

    matrix = sp.csc_matrix([[3e-3, -1e-3, 0], [-2, 4, -1], [0, -1000, 5000]])
    rhs = np.array([[1.0, 2.0], [3, 4], [5, 6]])
    lu = Factorization(matrix, refine=True, compensated_refinement=True)
    assert lu.refinement_steps == 2
    for trans in (False, True):
        a = matrix.T if trans else matrix
        np.testing.assert_allclose(
            lu.solve(rhs, transpose=trans),
            np.linalg.solve(a.toarray(), rhs),
            rtol=1e-13,
            atol=1e-12,
        )


def test_stage12_profile_does_not_change_old_physics():
    from waveforge.lab.mcl.contracts import stage11_spec, stage12_spec

    old, new = stage11_spec(), stage12_spec()
    assert (
        old.h_air == 20
        and old.pressure_drop_pa == 10000
        and old.resistance_ratio == 1e5
    )
    assert (
        new.h_air == 500
        and new.pressure_drop_pa == 2500
        and new.resistance_ratio == 1e4
    )
    assert new.compensated_refinement and new.reduced_order_homogenized_model


def test_bounded_waypoints_include_last_segment():
    from waveforge.lab.mcl.vascular_stage12 import limited_waypoints

    raw = torch.tensor(
        [
            [
                [100.0, 100.0, 100.0, 100.0],
                [-100.0, -100.0, -100.0, -100.0],
                [100.0, -100.0, 100.0, -100.0],
                [-100.0, 100.0, -100.0, 100.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    y = limited_waypoints(raw)
    ends = torch.full_like(y[..., :1], 31.5)
    knots = torch.cat([ends, y, ends], -1)
    assert ((knots[..., 1:] - knots[..., :-1]).abs() <= 16).all()
    assert ((y >= 6) & (y <= 57)).all()
    assert torch.isfinite(torch.autograd.grad(y.sum(), raw)[0]).all()


def test_reservation_keeps_every_core_cell_even_under_adverse_scores():
    from waveforge.lab.mcl.vascular_stage12 import reserve_core

    scores = torch.full((1, 64, 64), 25.0, dtype=torch.float64, requires_grad=True)
    core = torch.zeros_like(scores, dtype=torch.bool)
    core[:, :, 31:33] = True
    adjusted = torch.where(core, -70.0, scores)
    binary, rank = reserve_core(adjusted, core)
    assert binary.sum() == 1024 and binary[core].all()
    g = torch.autograd.grad(rank.sum(), scores)[0]
    assert (g[~core] == 1).all()
    too_big = torch.ones_like(core)
    with pytest.raises(ValueError, match="budget"):
        reserve_core(scores, too_big)


def test_candidate_gate_rejects_reynolds_500_and_unqualified_residual():
    from waveforge.lab.mcl.candidate import candidate_status

    good = dict(
        max_local_reynolds_estimate=499.0,
        pressure_residual=1e-15,
        thermal_residual=[2e-14],
        compensated_pressure_residual=1e-15,
        compensated_thermal_residual=[2e-14],
        mass_relative_error=1e-12,
        energy_relative_error=[1e-12],
    )
    assert candidate_status(good)["accepted"]
    high = candidate_status(good | {"max_local_reynolds_estimate": 500.0})
    assert not high["accepted"] and high["reynolds_contract_violated"]
    assert not candidate_status(good | {"thermal_residual": [2e-12]})["accepted"]
    incomplete = {k: v for k, v in good.items() if not k.startswith("compensated")}
    assert not candidate_status(incomplete)["accepted"]
    assert not candidate_status(good | {"compensated_thermal_residual": []})["accepted"]


def test_spatial_pixel_counterexample_is_rejected_not_repaired():
    from waveforge.lab.mcl.vascular_stage12 import decode

    rng = np.random.default_rng(460)
    for _ in range(74):
        raw = rng.uniform(-5, 5, (1, 4, 4))
        widths = rng.uniform(1, 4, (1, 4))
        pixel = rng.choice([-20.0, 20.0], (1, 64, 64))
    with pytest.raises(ValueError, match="disconnected"):
        decode(torch.tensor(raw), torch.tensor(widths), torch.tensor(pixel))


def test_stage12_torch_and_reference_use_two_compensated_steps():
    from waveforge.lab.mcl import MCLProblem, MCLSolver, solve_reference
    from waveforge.lab.mcl.contracts import stage12_spec

    spec = stage12_spec(nx=6, ny=6)
    power = np.zeros((2, 6, 6))
    power[:, 1:3, 1:3] = 2.0
    problem = MCLProblem(spec, np.ones((6, 6), bool), power)
    design = (
        torch.linspace(0.2, 0.7, 36, dtype=torch.float64)
        .reshape(1, 6, 6)
        .requires_grad_()
    )
    solver = MCLSolver([problem])
    result = solver(design)
    torch.autograd.grad(result.temperature.square().sum(), design)
    assert all(r["refinement_steps"] == 2 for r in solver.linear_diagnostics)
    assert all(
        r["compensated_relative_residual"] < 1e-12 for r in solver.linear_diagnostics
    )
    ref = solve_reference(design.detach().numpy()[0], problem)
    assert max(ref.diagnostics["compensated_thermal_residual"]) < 1e-12
    np.testing.assert_allclose(
        result.temperature.detach().numpy()[0], ref.temperature, rtol=1e-11, atol=1e-8
    )


def test_reference_records_do_not_hide_compensated_failures():
    from types import SimpleNamespace

    from waveforge.experiments.qualify_mcl_stage11 import reference_records

    result = SimpleNamespace(
        diagnostics=dict(
            pressure_residual=1e-15,
            thermal_residual=[1e-15],
            compensated_pressure_residual=1e-15,
            compensated_thermal_residual=[2e-12],
        )
    )
    records = reference_records(result, "probe")
    assert records[0]["accepted"]
    assert not records[1]["accepted"]
    assert records[1]["compensated_relative_residual"] == 2e-12


def test_stage12_geometry_audit_uses_new_radius_and_preserves_fixtures(tmp_path):
    from pathlib import Path

    from waveforge.experiments.qualify_mcl_stage12 import check_geometry

    root = Path(__file__).resolve().parents[1]
    result = check_geometry(root, tmp_path)
    assert result["primary"]["evaluations"] == 772
    assert result["primary"]["max_core_cells"] <= 768
    assert result["primary"]["failures_all_core_retained"] == 0
    assert {r["name"] for r in result["external_replays"]["records"]} >= {
        "uniform_108",
        "uniform_14",
    }
    assert result["external_replays"]["parameters_unchanged"]
    assert result["passed"]


def test_decoder_rejects_integer_parameters_and_mismatched_pixel_dtype():
    from waveforge.lab.mcl.vascular_stage12 import limited_waypoints, smooth_scores

    with pytest.raises(ValueError):
        limited_waypoints(torch.zeros((1, 4, 4), dtype=torch.int64))
    with pytest.raises(ValueError):
        smooth_scores(
            torch.full((1, 4, 4), 31.5, dtype=torch.float64),
            torch.ones((1, 4), dtype=torch.float64),
            torch.zeros((1, 64, 64), dtype=torch.float32),
        )
