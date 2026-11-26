"""Vectorized short-row CSR residual using FP64 error-free transformations.

Arithmetic keeps high/low words until final FP64 correction RHS. No longdouble,
no dense N-by-N matrix and no Python loop over cells. This is double-double-style
accumulation, NOT IEEE binary128. Intended for finite normal-range MCL operands.
"""

import os

import numpy as np


def two_sum(a, b):
    high = a + b
    virtual = high - a
    low = (a - (high - virtual)) + (b - virtual)
    return high, low


def _split(a):
    split = a * 134217729.0
    high = split - (split - a)
    return high, a - high


def two_prod(a, b):
    """Dekker product for the bounded, non-over/underflowing MCL arithmetic."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(abs(a) > 1e280)
        or np.any(abs(b) > 1e280)
    ):
        raise ValueError("compensated multiplication requires finite bounded operands")
    ah, al = _split(a)
    bh, bl = _split(b)
    product = a * b
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    return product, error


def _add(hi, lo, product, error):
    hi, carry = two_sum(hi, product)
    lo, carry2 = two_sum(lo, error)
    hi, carry = two_sum(hi, carry + lo)
    return two_sum(hi, carry + carry2)


class CompensatedCSR:
    """Cache a padded sparse-row layout, O(nnz + max_row_nnz*N) storage."""

    def __init__(self, matrix, *, backend=None):
        if backend is None:
            backend = os.environ.get("WAVEFORGE_RESIDUAL_BACKEND", "auto")
        if backend not in ("auto", "numpy", "numba"):
            raise ValueError("unknown compensated residual backend")
        self.compiled = None
        if backend != "numpy":
            try:
                from ._compiled_residual import real_residual

                self.compiled = real_residual
            except ImportError:
                if backend == "numba":
                    raise
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if not np.isfinite(matrix.data).all() or np.any(abs(matrix.data) > 1e280):
            raise ValueError("finite bounded sparse coefficients required")
        self.shape = matrix.shape
        counts = np.diff(matrix.indptr)
        width = int(counts.max(initial=0))
        # Thermal/pressure stencils are short; don't silently densify general matrices.
        if width > 64:
            raise ValueError("compensated CSR supports at most 64 entries per row")
        self.values = np.zeros((width, matrix.shape[0]), dtype=matrix.dtype)
        self.indices = np.zeros((width, matrix.shape[0]), dtype=np.intp)
        for slot in range(width):
            rows = np.flatnonzero(counts > slot)
            positions = matrix.indptr[rows] + slot
            self.values[slot, rows] = matrix.data[positions]
            self.indices[slot, rows] = matrix.indices[positions]
        self.real_parts = _split(self.values.real)
        self.imag_parts = (
            _split(self.values.imag) if np.iscomplexobj(self.values) else None
        )

    def _real_residual(self, x, b, *, imaginary_output=False):
        hi, lo = b.copy(), np.zeros_like(b)
        # Real(Ax)=Ar*xr-Ai*xi; Im(Ax)=Ar*xi+Ai*xr.
        terms = [
            (
                self.values.real,
                self.real_parts,
                x.imag if imaginary_output else x.real,
                -1,
            )
        ]
        if self.imag_parts is not None:
            terms.append(
                (
                    self.values.imag,
                    self.imag_parts,
                    x.real if imaginary_output else x.imag,
                    -1 if imaginary_output else 1,
                )
            )
        for values, parts, vector, sign in terms:
            vh, vl = _split(vector)
            for slot, indices in enumerate(self.indices):
                a = values[slot, :, None] * sign
                ah, al = parts[0][slot, :, None] * sign, parts[1][slot, :, None] * sign
                v, bh, bl = vector[indices], vh[indices], vl[indices]
                prod = a * v
                err = ((ah * bh - prod) + ah * bl + al * bh) + al * bl
                hi, lo = _add(hi, lo, prod, err)
        return hi + lo

    def residual(self, x, rhs):
        x, rhs = np.asarray(x), np.asarray(rhs)
        vector = x.ndim == 1
        if vector:
            x, rhs = x[:, None], rhs[:, None]
        if (
            x.ndim != 2
            or rhs.shape != (self.shape[0], x.shape[1])
            or x.shape[0] != self.shape[1]
            or not np.isfinite(x).all()
            or not np.isfinite(rhs).all()
            or np.any(abs(x) > 1e280)
        ):
            raise ValueError("finite compatible FP64 solution and RHS required")
        if (
            self.compiled is not None
            and self.values.dtype == np.float64
            and x.dtype == np.float64
            and rhs.dtype == np.float64
        ):
            result = self.compiled(self.values, self.indices, *self.real_parts, x, rhs)
            return result[:, 0] if vector else result
        real = self._real_residual(x, rhs.real)
        if np.iscomplexobj(x) or np.iscomplexobj(self.values) or np.iscomplexobj(rhs):
            real = real + 1j * self._real_residual(x, rhs.imag, imaginary_output=True)
        return real[:, 0] if vector else real
