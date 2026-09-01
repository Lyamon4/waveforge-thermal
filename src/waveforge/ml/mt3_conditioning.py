"""Canonical feasible-state physics and sensitivity conditioning for MT3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from torch import Tensor

from waveforge.design.batched_differentiable_solver import (
    BatchedSolveTrace,
    solve_steady_implicit_batched,
)
from waveforge.design.objectives import objective_components
from waveforge.design.parameterization import filter_logits, project_volume
from waveforge.physics.grid import Grid2D

FIXED_TEMPERATURE_SCALE = 0.900613256638055
MT3ConditioningVariant: TypeAlias = Literal["FIELD_UNET", "SENS_UNET"]
_GRID = Grid2D(nx=64, ny=64)


@dataclass(frozen=True)
class MT3Probe:
    """Detached evidence from one canonical feasible-state physics probe."""

    design: Tensor
    temperatures: Tensor
    temperature_mean: Tensor
    temperature_max: Tensor
    benefit_raw: Tensor
    benefit_normalized: Tensor
    trace: BatchedSolveTrace


def _validate_sources(sources: Tensor, *, allow_cpu_unit_test: bool) -> None:
    if sources.ndim != 4 or tuple(sources.shape[1:]) != (3, 64, 64):
        raise ValueError("sources must have shape [batch,3,64,64]")
    if sources.dtype is not torch.float64:
        raise ValueError("sources must use float64 physics precision")
    if not torch.isfinite(sources).all():
        raise ValueError("sources must be finite")
    if sources.device.type != "cuda" and not allow_cpu_unit_test:
        raise ValueError("production MT3 probe requires CUDA")


def _project_initial_design(logits: Tensor) -> Tensor:
    designs: list[Tensor] = []
    for task_logits in logits:
        filtered = filter_logits(task_logits, sigma=1.0, radius=3, padding="reflect")
        design, diagnostics = project_volume(filtered, beta=8.0, target=0.25)
        if not diagnostics.converged or diagnostics.absolute_error > 1.0e-6:
            raise RuntimeError("canonical initial projection is invalid")
        designs.append(design)
    return torch.stack(designs)


def _normalize_benefit(benefit: Tensor) -> Tensor:
    scale = benefit.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
    return (benefit / scale).clamp(-8.0, 8.0)


def _validate_trace(trace: BatchedSolveTrace, *, batch_size: int) -> None:
    expected = 2 * batch_size * 3
    if len(trace.records) != expected:
        raise RuntimeError("canonical probe trace is incomplete")
    if any(not record.converged for record in trace.records):
        raise RuntimeError("canonical probe physics did not converge")
    if max(record.relative_residual for record in trace.records) > 1.0e-6:
        raise RuntimeError("canonical probe residual exceeds tolerance")


def evaluate_probe_objective(
    sources: Tensor,
    logits: Tensor,
    *,
    allow_cpu_unit_test: bool = False,
) -> Tensor:
    """Evaluate the locked thermal probe objective at supplied full-grid logits."""
    _validate_sources(sources, allow_cpu_unit_test=allow_cpu_unit_test)
    if (
        logits.shape != (sources.shape[0], 64, 64)
        or logits.dtype is not torch.float32
        or logits.device != sources.device
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("probe logits must be finite float32 [batch,64,64]")
    design = _project_initial_design(logits)
    conductivity = 1.0 + 19.0 * design.to(torch.float64).pow(3)
    trace = BatchedSolveTrace()
    temperatures = solve_steady_implicit_batched(
        conductivity,
        sources.detach(),
        _GRID,
        trace=trace,
    )
    if len(trace.records) != sources.shape[0] * 3:
        raise RuntimeError("probe objective forward trace is incomplete")
    if any(not record.converged for record in trace.records):
        raise RuntimeError("probe objective forward physics did not converge")
    objectives = [
        objective_components(
            temperatures[index],
            design[index],
            alpha=500.0,
            tv_weight=0.0,
            binarization_weight=0.0,
        ).thermal_smooth
        for index in range(sources.shape[0])
    ]
    return torch.stack(objectives).sum()


def compute_initial_probe(
    sources: Tensor,
    *,
    allow_cpu_unit_test: bool = False,
) -> MT3Probe:
    """Compute one feasible-state forward/adjoint probe for each task."""
    _validate_sources(sources, allow_cpu_unit_test=allow_cpu_unit_test)
    detached_sources = sources.detach()
    with torch.enable_grad():
        logits = torch.zeros(
            (sources.shape[0], 64, 64),
            dtype=torch.float32,
            device=sources.device,
            requires_grad=True,
        )
        design = _project_initial_design(logits)
        conductivity = 1.0 + 19.0 * design.to(torch.float64).pow(3)
        trace = BatchedSolveTrace()
        temperatures = solve_steady_implicit_batched(
            conductivity,
            detached_sources,
            _GRID,
            trace=trace,
        )
        thermal_objectives = [
            objective_components(
                temperatures[index],
                design[index],
                alpha=500.0,
                tv_weight=0.0,
                binarization_weight=0.0,
            ).thermal_smooth
            for index in range(sources.shape[0])
        ]
        total_thermal = torch.stack(thermal_objectives).sum()
        (gradient,) = torch.autograd.grad(total_thermal, logits)

    _validate_trace(trace, batch_size=sources.shape[0])
    benefit = -gradient.detach()
    detached_temperatures = temperatures.detach()
    return MT3Probe(
        design=design.detach(),
        temperatures=detached_temperatures,
        temperature_mean=detached_temperatures.mean(dim=1),
        temperature_max=detached_temperatures.max(dim=1).values,
        benefit_raw=benefit,
        benefit_normalized=_normalize_benefit(benefit),
        trace=trace,
    )


def build_mt3_conditioning(
    probe: MT3Probe,
    sources: Tensor,
    *,
    variant: MT3ConditioningVariant,
) -> Tensor:
    """Build matched five-channel FIELD or SENS U-Net conditioning."""
    _validate_sources(sources, allow_cpu_unit_test=True)
    batch = sources.shape[0]
    expected_shape = (batch, 64, 64)
    fields = (
        probe.design,
        probe.temperature_mean,
        probe.temperature_max,
        probe.benefit_normalized,
    )
    if any(tuple(field.shape) != expected_shape for field in fields):
        raise ValueError("probe tensors do not match the source batch")
    if variant not in {"FIELD_UNET", "SENS_UNET"}:
        raise ValueError(f"unknown MT3 conditioning variant {variant!r}")

    detached_sources = sources.detach()
    source_sum = detached_sources.sum(dim=1).to(torch.float32) / 25.0
    temperature_mean = (probe.temperature_mean / FIXED_TEMPERATURE_SCALE).to(
        torch.float32
    )
    temperature_max = (probe.temperature_max / FIXED_TEMPERATURE_SCALE).to(
        torch.float32
    )
    benefit = probe.benefit_normalized.to(torch.float32)
    if variant == "FIELD_UNET":
        benefit = torch.zeros_like(benefit)
    sink = torch.zeros_like(source_sum)
    sink[:, 0, :] = 1.0
    return torch.stack(
        (source_sum, temperature_mean, temperature_max, benefit, sink),
        dim=1,
    ).detach()
