"""LU with optional row scaling and one ordinary or two compensated corrections.

This changes neither equations nor acceptance thresholds. In particular, one
refinement is NOT assumed to pass an arbitrarily stringent residual test.
"""

import os
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .compensated import CompensatedCSR

_backend_override = ContextVar("waveforge_factorization_backend", default=None)


@contextmanager
def factorization_backend(name):
    """Choose an independent backend locally, without racing process environment."""
    if name not in ("scipy", "cudss"):
        raise ValueError("unknown sparse factorization backend")
    token = _backend_override.set(name)
    try:
        yield
    finally:
        _backend_override.reset(token)


class Factorization:
    def __init__(
        self, matrix, *, refine=False, ordering="COLAMD", compensated_refinement=False
    ):
        self.matrix = matrix
        self.refine = refine
        if compensated_refinement and not refine:
            raise ValueError("compensated refinement requires row scaling")
        self.refinement_steps = 2 if compensated_refinement else int(refine)
        self.compensated = CompensatedCSR(matrix) if compensated_refinement else None
        self.compensated_transpose = None
        diagonal = matrix.diagonal()
        if not np.isfinite(diagonal).all() or np.any(diagonal == 0):
            raise ValueError("finite nonzero matrix diagonal required")
        self.scale = 1 / diagonal if refine else np.ones_like(diagonal)
        scaled = (sp.diags(self.scale) @ matrix).tocsc() if refine else matrix
        backend = _backend_override.get() or os.environ.get(
            "WAVEFORGE_SPARSE_BACKEND", "scipy"
        )
        if backend not in ("scipy", "cudss"):
            raise ValueError("unknown sparse factorization backend")
        minimum_size = int(os.environ.get("WAVEFORGE_CUDSS_MIN_SIZE", "16384"))
        if (
            backend == "cudss"
            and matrix.dtype == np.float64
            and matrix.shape[0] >= minimum_size
        ):
            from .cudss_backend import CudssLU

            self.lu = CudssLU(scaled)
            self.backend = "nvidia_cudss"
        else:
            self.lu = spla.splu(scaled, permc_spec=ordering)
            self.backend = "scipy_lu"

    def _solve(self, rhs, transpose):
        scale = self.scale if rhs.ndim == 1 else self.scale[:, None]
        if transpose:
            # A_s = S A => A.T lambda=b => A_s.T y=b, lambda=S y.
            # Algebraic transpose (not conjugate) also supports complex-step.
            return scale * self.lu.solve(rhs, trans="T")
        return self.lu.solve(scale * rhs)

    def solve(self, rhs, *, transpose=False):
        result = self._solve(rhs, transpose)
        for _ in range(self.refinement_steps):
            result = result + self._solve(
                self.residual(result, rhs, transpose=transpose), transpose
            )
        return result

    def residual(self, x, rhs, *, transpose=False):
        if self.compensated is not None:
            if transpose and self.compensated_transpose is None:
                self.compensated_transpose = CompensatedCSR(self.matrix.T)
            operator = self.compensated_transpose if transpose else self.compensated
            return operator.residual(x, rhs)
        return rhs - (self.matrix.T if transpose else self.matrix) @ x

    def relative_residual(self, x, rhs, *, transpose=False):
        residual = self.residual(x, rhs, transpose=transpose)
        return np.linalg.norm(residual, axis=0) / np.maximum(
            np.linalg.norm(rhs, axis=0), 1e-30
        )
