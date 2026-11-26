"""Reusable sparse-LU implicit autograd, including nonsymmetric transpose solves.

Factorization runs on CPU/SciPy. CUDA tensors are explicitly bridged; this is
NOT a native GPU Krylov backend. No dense production matrices are allocated.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd.function import once_differentiable

from .factorization import Factorization


class NumericalQualificationError(RuntimeError):
    """Strict tolerance rejection with full-precision, structured diagnostics."""

    def __init__(self, phase, residual, records):
        super().__init__(f"MCL {phase} solve unqualified: residual={residual:.17e}")
        self.relative_residual = residual
        self.records = [dict(record) for record in records]


def _residual(a, x, b):
    return float(
        np.max(
            np.linalg.norm(a @ x - b, axis=0)
            / np.maximum(np.linalg.norm(b, axis=0), 1e-30)
        )
    )


class _SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values,
        rhs,
        rows,
        cols,
        size,
        records,
        scaled_refinement,
        diagnostic_only,
        compensated_refinement,
    ):
        value_np = values.detach().cpu().numpy()
        rhs_np = rhs.detach().cpu().numpy()
        solutions, factors, matrices = [], [], []
        for val, b in zip(value_np, rhs_np, strict=True):
            a = sp.coo_matrix((val, (rows, cols)), shape=(size, size)).tocsc()
            a.eliminate_zeros()
            lu = Factorization(
                a,
                refine=scaled_refinement,
                ordering="MMD_AT_PLUS_A",
                compensated_refinement=compensated_refinement,
            )
            x = lu.solve(b)
            residual = _residual(a, x, b)
            compensated_residual = float(np.max(lu.relative_residual(x, b)))
            finite = bool(
                np.isfinite(x).all()
                and np.isfinite(residual)
                and np.isfinite(compensated_residual)
            )
            worst_residual = max(residual, compensated_residual)
            record = {
                "phase": "forward",
                "size": size,
                "rhs": b.shape[1],
                "relative_residual": residual,
                "backend": "scipy_lu",
                "refinement_steps": lu.refinement_steps,
                "compensated_relative_residual": compensated_residual,
                "residual_system": "original_unscaled",
                "accepted": finite and worst_residual < 1e-12,
                "diagnostic_only": diagnostic_only,
            }
            if records is not None:
                records.append(record)
            if not finite or (worst_residual >= 1e-12 and not diagnostic_only):
                raise NumericalQualificationError(
                    "forward",
                    worst_residual,
                    records if records is not None else [record],
                )
            solutions.append(x)
            factors.append(lu)
            matrices.append(a)
        answer = torch.as_tensor(
            np.stack(solutions), dtype=rhs.dtype, device=rhs.device
        )
        ctx.save_for_backward(answer)
        ctx.rows = torch.as_tensor(rows, dtype=torch.long, device=values.device)
        ctx.cols = torch.as_tensor(cols, dtype=torch.long, device=values.device)
        ctx.factors, ctx.matrices, ctx.records = factors, matrices, records
        ctx.scaled_refinement = scaled_refinement
        ctx.diagnostic_only = diagnostic_only
        return answer

    @staticmethod
    @once_differentiable
    def backward(ctx, incoming):
        (state,) = ctx.saved_tensors
        upstream = incoming.detach().cpu().numpy()
        adjoints = []
        for lu, a, b in zip(ctx.factors, ctx.matrices, upstream, strict=True):
            adjoint = lu.solve(b, transpose=True)
            residual = _residual(a.T, adjoint, b)
            compensated_residual = float(
                np.max(lu.relative_residual(adjoint, b, transpose=True))
            )
            finite = bool(
                np.isfinite(adjoint).all()
                and np.isfinite(residual)
                and np.isfinite(compensated_residual)
            )
            worst_residual = max(residual, compensated_residual)
            record = {
                "phase": "adjoint",
                "size": a.shape[0],
                "rhs": b.shape[1],
                "relative_residual": residual,
                "backend": "scipy_lu",
                "factorization_reused": True,
                "refinement_steps": lu.refinement_steps,
                "compensated_relative_residual": compensated_residual,
                "residual_system": "original_unscaled",
                "accepted": finite and worst_residual < 1e-12,
                "diagnostic_only": ctx.diagnostic_only,
            }
            if ctx.records is not None:
                ctx.records.append(record)
            if not finite or (worst_residual >= 1e-12 and not ctx.diagnostic_only):
                raise NumericalQualificationError(
                    "adjoint",
                    worst_residual,
                    ctx.records if ctx.records is not None else [record],
                )
            adjoints.append(adjoint)
        adjoint = torch.as_tensor(
            np.stack(adjoints), dtype=state.dtype, device=state.device
        )
        grad_values = -(adjoint[:, ctx.rows] * state[:, ctx.cols]).sum(-1)
        return grad_values, adjoint, None, None, None, None, None, None, None


def sparse_solve(
    values,
    rhs,
    rows,
    cols,
    size,
    records=None,
    *,
    scaled_refinement=False,
    diagnostic_only=False,
    compensated_refinement=False,
):
    """values[B,E], rhs[B,N,S]; duplicate COO entries have correct gradients."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    if diagnostic_only and records is None:
        raise ValueError("diagnostic continuation MUST retain acceptance records")
    if (
        values.dtype != torch.float64
        or rhs.dtype != torch.float64
        or values.device != rhs.device
    ):
        raise ValueError("MCL sparse solve requires colocated FP64 tensors")
    if (
        values.ndim != 2
        or rhs.ndim != 3
        or values.shape[0] < 1
        or rhs.shape[:2] != (values.shape[0], size)
        or rhs.shape[2] < 1
        or rows.ndim != 1
        or cols.shape != rows.shape
        or values.shape[1] != rows.size
        or rows.dtype.kind not in "iu"
        or cols.dtype.kind not in "iu"
        or np.any(rows < 0)
        or np.any(rows >= size)
        or np.any(cols < 0)
        or np.any(cols >= size)
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(rhs).all())
    ):
        raise ValueError("invalid sparse solve shapes, indices or nonfinite values")
    return _SparseSolve.apply(
        values,
        rhs,
        rows,
        cols,
        size,
        records,
        scaled_refinement,
        diagnostic_only,
        compensated_refinement,
    )
