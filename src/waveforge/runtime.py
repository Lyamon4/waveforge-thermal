"""Explicit, context-local choices of scientifically equivalent implementations."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeOptions:
    solver_backend: str = "legacy"
    vectorized_projection: bool = False
    reuse_reference_factorization: bool = False
    cg_tolerance: float | None = None
    accelerated_projection: bool = False

    def __post_init__(self) -> None:
        if self.solver_backend not in ("legacy", "cuda_persistent", "scipy_banded"):
            raise ValueError(f"unsupported solver backend: {self.solver_backend!r}")
        for name in (
            "vectorized_projection",
            "reuse_reference_factorization",
            "accelerated_projection",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if self.cg_tolerance is not None:
            if isinstance(self.cg_tolerance, bool) or not isinstance(
                self.cg_tolerance, (float, int)
            ):
                raise TypeError("cg_tolerance must be a real number or None")
            if (
                not math.isfinite(self.cg_tolerance)
                or not 0 < self.cg_tolerance <= 1e-6
            ):
                raise ValueError("cg_tolerance must be finite and in (0, 1e-6]")

    def to_dict(self) -> dict[str, str | bool | float]:
        values = {
            "solver_backend": self.solver_backend,
            "vectorized_projection": self.vectorized_projection,
            "reuse_reference_factorization": self.reuse_reference_factorization,
        }
        if self.cg_tolerance is not None:
            values["cg_tolerance"] = self.cg_tolerance
        if self.accelerated_projection:
            values["accelerated_projection"] = True
        return values


_DEFAULT_OPTIONS = RuntimeOptions()
_OPTIONS: ContextVar[RuntimeOptions] = ContextVar(
    "waveforge_runtime", default=_DEFAULT_OPTIONS
)


def runtime_options() -> RuntimeOptions:
    return _OPTIONS.get()


@contextmanager
def use_runtime(options: RuntimeOptions) -> Iterator[RuntimeOptions]:
    if not isinstance(options, RuntimeOptions):
        raise TypeError("options must be RuntimeOptions")
    token = _OPTIONS.set(options)
    try:
        yield options
    finally:
        _OPTIONS.reset(token)
