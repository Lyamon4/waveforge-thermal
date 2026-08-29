"""Differentiable thermal training path for the locked pure NCA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from waveforge.design.differentiable_solver import (
    SolveTrace,
    solve_steady_implicit,
)
from waveforge.design.objectives import ObjectiveComponents, objective_components
from waveforge.design.parameterization import ProjectionDiagnostics, binary_design
from waveforge.ml.nca import (
    NCARollout,
    PureNCA,
    build_static_condition,
    project_nca_material,
)
from waveforge.physics.grid import Grid2D


@dataclass(frozen=True)
class NCAForwardResult:
    """Full differentiable forward and separately retained diagnostics."""

    rollout: NCARollout
    continuous_design: Tensor
    binary_design: Tensor | None
    temperatures: Tensor
    objective: ObjectiveComponents
    projection: ProjectionDiagnostics
    solve_trace: SolveTrace


def _model_device_and_dtype(model: PureNCA) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def evaluate_nca(
    model: PureNCA,
    sources: Tensor,
    *,
    trace: SolveTrace | None = None,
    allow_cpu_unit_test: bool = False,
    include_binary_diagnostic: bool = False,
    snapshot_steps: tuple[int, ...] = (),
) -> NCAForwardResult:
    """Evaluate the fixed NCA, projection, steady physics and objective."""
    device, dtype = _model_device_and_dtype(model)
    if dtype is not torch.float32:
        raise ValueError("NCA model parameters must be float32")
    if device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("pure-NCA training physics requires CUDA")
    if sources.shape != (3, 64, 64) or sources.dtype is not torch.float64:
        raise ValueError("sources must have shape [3,64,64] and dtype float64")
    if sources.device != device:
        raise ValueError("sources and NCA model must share a device")
    if not torch.isfinite(sources).all():
        raise ValueError("sources must be finite")

    condition = build_static_condition(sources)
    rollout = model.rollout(condition, snapshot_steps=snapshot_steps)
    projected = project_nca_material(rollout.material_logit)
    conductivity = 1.0 + 19.0 * projected.design.to(torch.float64).pow(3)
    solve_trace = trace if trace is not None else SolveTrace()
    temperatures = solve_steady_implicit(
        conductivity,
        sources,
        Grid2D(nx=64, ny=64),
        trace=solve_trace,
    )
    objective = objective_components(
        temperatures,
        projected.design,
        alpha=500.0,
        tv_weight=1.0e-3,
        binarization_weight=0.02,
    )
    diagnostic_binary: Tensor | None = None
    if include_binary_diagnostic:
        with torch.no_grad():
            diagnostic_binary = binary_design(projected.design)
    return NCAForwardResult(
        rollout=rollout,
        continuous_design=projected.design,
        binary_design=diagnostic_binary,
        temperatures=temperatures,
        objective=objective,
        projection=projected.projection,
        solve_trace=solve_trace,
    )
