"""Locked paired-training components for the NCA-MT2B experiment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, TypeAlias

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.objectives import objective_components
from waveforge.ml.mt2b_conditioning import build_mt2b_conditioning
from waveforge.ml.mt2b_nca import MT2BNCA
from waveforge.ml.mt2b_tasks import balanced_task_batch
from waveforge.ml.multitask_protocol import MultitaskStage
from waveforge.ml.multitask_tasks import SourceLayoutTask
from waveforge.ml.multitask_training import MultitaskForward
from waveforge.ml.nca import project_nca_material
from waveforge.physics.fixed_operator import UniformPlateFactorization
from waveforge.physics.grid import Grid2D
from waveforge.reproducibility import set_deterministic_seed

MT2BVariant: TypeAlias = Literal["RAW", "PHYSICS"]
_GRID = Grid2D(nx=64, ny=64)


def initialize_mt2b_model(seed: int, device: torch.device) -> MT2BNCA:
    """Create either matched variant from the exact same seeded initialization."""
    set_deterministic_seed(seed)
    return MT2BNCA().to(device=device, dtype=torch.float32)


@lru_cache(maxsize=16)
def _locked_batch(seed: int, update: int) -> tuple[SourceLayoutTask, ...]:
    return balanced_task_batch(
        update,
        seed=seed,
        excluded_task_ids=frozenset(),
    )


def mt2b_task_provider(
    seed: int,
    update: int,
    microbatch_index: int,
) -> SourceLayoutTask:
    """Return one member of the prospectively balanced four-task update."""
    if microbatch_index not in range(4):
        raise ValueError("MT2B microbatch_index must lie in [0,4)")
    return _locked_batch(seed, update)[microbatch_index]


def _fixed_temperature_solver(
    factorization: UniformPlateFactorization,
    sources: NDArray[np.float64],
) -> NDArray[np.float64]:
    batch, scenarios, ny, nx = sources.shape
    result = factorization.solve_many(sources.reshape(batch * scenarios, ny, nx))
    if result.maximum_normalized_residual > 1.0e-10:
        raise FloatingPointError("fixed conditioning solve residual is invalid")
    return result.temperature.reshape(sources.shape)


def _model_device_and_dtype(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def build_mt2b_evaluator(variant: MT2BVariant):
    """Build the safe scenario-vectorized, task-sequential MT2B evaluator."""
    if variant not in {"RAW", "PHYSICS"}:
        raise ValueError(f"unknown MT2B variant {variant!r}")
    factorization = (
        UniformPlateFactorization(grid_size=64, conductivity=1.0)
        if variant == "PHYSICS"
        else None
    )

    def evaluate(
        model: nn.Module,
        sources: Tensor,
        stage: MultitaskStage,
        *,
        allow_cpu_unit_test: bool,
    ) -> MultitaskForward:
        device, dtype = _model_device_and_dtype(model)
        if dtype is not torch.float32:
            raise ValueError("MT2B model parameters must be float32")
        if device.type != "cuda" and not allow_cpu_unit_test:
            raise ValueError("MT2B training physics requires CUDA")
        if (
            sources.shape != (3, 64, 64)
            or sources.dtype is not torch.float64
            or sources.device != device
        ):
            raise ValueError(
                "sources must have shape [3,64,64], dtype float64, and model device"
            )
        if not torch.isfinite(sources).all():
            raise ValueError("sources must be finite")

        temperature_solver = None
        if factorization is not None:

            def solve_conditioning(array: NDArray[np.float64]) -> NDArray[np.float64]:
                return _fixed_temperature_solver(factorization, array)

            temperature_solver = solve_conditioning
        condition = build_mt2b_conditioning(
            sources.unsqueeze(0),
            variant=variant,
            temperature_solver=temperature_solver,
        )
        rollout = model.rollout(condition, steps=64)  # type: ignore[attr-defined]
        projected = project_nca_material(rollout.material_logit, beta=stage.beta)
        conductivity = 1.0 + 19.0 * projected.design.to(torch.float64).pow(3)
        trace = BatchedSolveTrace()
        temperatures = solve_steady_implicit_batched(
            conductivity.unsqueeze(0),
            sources.unsqueeze(0),
            _GRID,
            trace=trace,
        )[0]
        objective = objective_components(
            temperatures,
            projected.design,
            alpha=stage.alpha,
            tv_weight=stage.tv_weight,
            binarization_weight=stage.binary_weight,
        )
        return MultitaskForward(
            loss=objective.total,
            thermal_smooth=float(objective.thermal_smooth.detach().item()),
            exact_tmax=float(objective.exact_peak.detach().item()),
            continuous_material_fraction=float(projected.design.detach().mean().item()),
            projection_absolute_error=float(projected.projection.absolute_error),
            solve_trace=trace,  # type: ignore[arg-type]
        )

    return evaluate
