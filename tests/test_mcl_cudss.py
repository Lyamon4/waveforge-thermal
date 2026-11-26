"""Optional GPU LU must retain exact forward/transpose behavior and live leases."""

from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pytest
import scipy.sparse as sp
import torch


def require_cudss():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    major = torch.version.cuda.split(".")[0]
    try:
        version(f"nvidia-cudss-cu{major}")
    except PackageNotFoundError:
        pytest.skip("optional matching cuDSS package unavailable")


def test_cudss_two_live_factors_cannot_overwrite_each_other():
    require_cudss()
    from waveforge.lab.mcl.cudss_backend import CudssLU, clear_pool, pool_info

    clear_pool()
    a = sp.csr_matrix([[4.0, -1.0, 0.0], [-2.0, 5.0, -1.0], [0.0, -1.0, 3.0]])
    b = np.array([[1.0, 3.0], [2.0, 1.0], [4.0, 5.0]])
    first, second = CudssLU(a), CudssLU(a + sp.eye(3, format="csr"))
    try:
        x = first.solve(b)
        y = second.solve(b)
        np.testing.assert_allclose(a @ x, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose((a + sp.eye(3)) @ y, b, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(first.solve(b), x, rtol=1e-12, atol=1e-12)
        t = first.solve(b, trans="T")
        np.testing.assert_allclose(a.T @ t, b, rtol=1e-12, atol=1e-12)
    finally:
        first.close()
        second.close()
    before = pool_info()
    third = CudssLU(a * 1.2)
    try:
        np.testing.assert_allclose(a * 1.2 @ third.solve(b), b, rtol=1e-12, atol=1e-12)
        assert pool_info()["analysis_reuses"] == before["analysis_reuses"] + 1
    finally:
        third.close()
        clear_pool()


def test_cudss_backend_satisfies_original_fp64_compensated_gates(monkeypatch):
    require_cudss()
    from waveforge.lab.mcl.factorization import Factorization

    matrix = sp.csc_matrix([[3e-3, -1e-3, 0], [-2.0, 4.0, -1.0], [0.0, -100.0, 500.0]])
    rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    monkeypatch.setenv("WAVEFORGE_SPARSE_BACKEND", "cudss")
    monkeypatch.setenv("WAVEFORGE_CUDSS_MIN_SIZE", "0")
    factor = Factorization(matrix, refine=True, compensated_refinement=True)
    assert factor.backend == "nvidia_cudss"
    for transpose in (False, True):
        actual = factor.solve(rhs, transpose=transpose)
        expected = np.linalg.solve((matrix.T if transpose else matrix).toarray(), rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
        assert (
            np.max(factor.relative_residual(actual, rhs, transpose=transpose)) < 1e-12
        )


def test_complex_step_stays_on_scipy_even_when_gpu_requested(monkeypatch):
    from waveforge.lab.mcl.factorization import Factorization

    monkeypatch.setenv("WAVEFORGE_SPARSE_BACKEND", "cudss")
    monkeypatch.setenv("WAVEFORGE_CUDSS_MIN_SIZE", "0")
    matrix = sp.eye(3, format="csc", dtype=np.complex128) * (2 + 1e-20j)
    factor = Factorization(matrix)
    assert factor.backend == "scipy_lu"
    np.testing.assert_allclose(factor.solve(np.ones(3)), 1 / (2 + 1e-20j))


def test_gpu_multiple_rhs_shapes_and_returned_arrays_remain_independent():
    require_cudss()
    from waveforge.lab.mcl.cudss_backend import CudssLU, clear_pool

    matrix = sp.csr_matrix([[4.0, -1.0, 0.0], [-2.0, 5.0, -1.0], [0.0, -1.0, 3.0]])
    factor = CudssLU(matrix)
    try:
        rhs = np.array([1.0, 2.0, 3.0])
        first = factor.solve(rhs)
        saved = first.copy()
        np.testing.assert_allclose(matrix @ factor.solve(2 * rhs), 2 * rhs)
        np.testing.assert_array_equal(first, saved)
        many = np.stack((rhs, 3 * rhs), axis=1)
        np.testing.assert_allclose(matrix @ factor.solve(many), many)
        np.testing.assert_allclose(matrix.T @ factor.solve(rhs, trans="T"), rhs)
    finally:
        factor.close()
        clear_pool()


def test_gpu_implicit_adjoint_matches_scipy(monkeypatch):
    require_cudss()
    from waveforge.lab.mcl.linear import sparse_solve

    rows = np.array([0, 0, 1, 1, 1, 2, 2])
    cols = np.array([0, 1, 0, 1, 2, 1, 2])
    results = []
    monkeypatch.setenv("WAVEFORGE_CUDSS_MIN_SIZE", "0")
    for backend in ("scipy", "cudss"):
        monkeypatch.setenv("WAVEFORGE_SPARSE_BACKEND", backend)
        values = torch.tensor(
            [[4.0, -1.0, -2.0, 5.0, -1.0, -1.0, 3.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        rhs = torch.tensor(
            [[[1.0, 3.0], [2.0, 1.0], [4.0, 5.0]]],
            dtype=torch.float64,
            requires_grad=True,
        )
        records = []
        answer = sparse_solve(
            values,
            rhs,
            rows,
            cols,
            3,
            records,
            scaled_refinement=True,
            compensated_refinement=True,
        )
        gradients = torch.autograd.grad(answer.square().sum(), (values, rhs))
        assert all(row["accepted"] for row in records)
        results.append((answer.detach().numpy(), *(v.numpy() for v in gradients)))
    for expected, actual in zip(*results, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
