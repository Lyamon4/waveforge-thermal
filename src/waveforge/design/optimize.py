"""Deterministic fail-closed Gate 2A design optimization loop."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from waveforge.design.constraints import binary_budget_satisfied, material_fraction
from waveforge.design.differentiable_solver import (
    SolveRecord,
    SolveTrace,
    solve_steady_implicit,
)
from waveforge.design.objectives import objective_components
from waveforge.design.parameterization import (
    VolumeProjectionError,
    binary_design,
    parameterize_design,
)
from waveforge.physics.cg import CGConfig, CGConvergenceError
from waveforge.physics.grid import Grid2D
from waveforge.verification.compare import Gate2Status

OptimizationMode = Literal["unit", "benchmark", "smoke", "production"]
ObjectiveScope = Literal["single_A", "robust"]


@dataclass(frozen=True)
class OptimizationConfig:
    """Immutable numerical optimization settings."""

    iterations: int = 600
    mode: OptimizationMode = "production"
    objective_scope: ObjectiveScope = "robust"
    learning_rate: float = 0.05
    gradient_clip_norm: float = 1.0
    checkpoint_interval: int = 50
    enforce_final_binary_budget: bool = True
    cg_config: CGConfig = field(default_factory=CGConfig)

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.iterations > 600:
            raise ValueError("optimization iterations must lie in [1,600]")
        if self.mode == "production" and self.iterations != 600:
            raise ValueError("production optimization requires exactly 600 iterations")
        if self.mode == "smoke" and self.iterations != 10:
            raise ValueError("smoke optimization requires exactly 10 iterations")
        if self.mode == "benchmark" and self.iterations != 1:
            raise ValueError("benchmark mode requires exactly one iteration")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("optimizer learning rate and clip norm must be positive")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint interval must be positive")
        if self.objective_scope not in ("single_A", "robust"):
            raise ValueError("objective scope must be single_A or robust")


@dataclass(frozen=True)
class IterationRecord:
    """All separately logged scientific and numerical terms for one step."""

    iteration: int
    beta: float
    alpha: float
    binarization_weight: float
    total_objective: float
    thermal_smooth: float
    exact_peak: float
    total_variation: float
    binarization_penalty: float
    continuous_material_fraction: float
    binary_material_fraction: float
    gradient_norm_before_clipping: float
    best_exact_peak: float
    maximum_cg_iterations: int
    maximum_explicit_relative_residual: float
    wall_seconds: float


@dataclass(frozen=True)
class OptimizationResult:
    """Numerical run outcome, including failed and scientifically negative runs."""

    status: Gate2Status
    reason_codes: tuple[str, ...]
    seed: int
    run_id: str
    completed_iterations: int
    records: tuple[IterationRecord, ...]
    solve_records: tuple[SolveRecord, ...]
    initial_logits: Tensor
    final_logits: Tensor
    initial_logits_hash: str
    final_logits_hash: str
    continuous_design: Tensor | None
    binary_design: Tensor | None
    continuous_material_fraction: float | None
    binary_material_fraction: float | None
    config_sha256: str


def beta_for_iteration(iteration: int) -> float:
    """Return the locked projection-sharpness schedule."""
    if not 0 <= iteration < 600:
        raise ValueError("iteration must lie in [0,599]")
    if iteration < 200:
        return 1.0
    if iteration < 350:
        return 2.0
    if iteration < 500:
        return 4.0
    return 8.0


def alpha_for_iteration(iteration: int) -> float:
    """Return the locked normalized-smooth-maximum schedule."""
    if not 0 <= iteration < 600:
        raise ValueError("iteration must lie in [0,599]")
    if iteration < 200:
        return 50.0
    if iteration < 350:
        return 200.0
    return 500.0


def binarization_weight_for_iteration(iteration: int) -> float:
    """Return the locked direct binarization-penalty schedule."""
    if not 0 <= iteration < 600:
        raise ValueError("iteration must lie in [0,599]")
    if iteration < 200:
        return 0.0
    if iteration < 350:
        return 0.005
    if iteration < 500:
        return 0.01
    return 0.02


def initialize_logits(seed: int, *, device: torch.device) -> Tensor:
    """Create isolated NumPy-PCG64 `N(0,0.1²)` CUDA float32 logits."""
    values = np.random.default_rng(seed).normal(
        loc=0.0,
        scale=0.1,
        size=(16, 16),
    )
    return torch.as_tensor(values, dtype=torch.float32, device=device).clone()


def array_sha256(tensor: Tensor) -> str:
    """Hash dtype, shape and row-major tensor bytes."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _config_hash() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return hashlib.sha256(
        (repository_root / "configs" / "inverse_design.yaml").read_bytes()
    ).hexdigest()


def _validate_sources(sources: Tensor) -> Grid2D:
    if sources.dtype is not torch.float64:
        raise ValueError("optimization sources must be float64")
    if sources.device.type != "cuda":
        raise ValueError("Gate 2A optimization requires CUDA")
    if sources.ndim != 3 or tuple(sources.shape[-2:]) != (64, 64):
        raise ValueError("optimization sources must have shape [scenario,64,64]")
    if not torch.isfinite(sources).all():
        raise ValueError("optimization sources must be finite")
    return Grid2D(nx=64, ny=64)


def _write_checkpoint(
    output_dir: Path,
    filename: str,
    *,
    seed: int,
    iteration: int,
    logits: Tensor,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 2,
        "seed": seed,
        "iteration": iteration,
        "logits": logits.detach().cpu(),
        "logits_sha256": array_sha256(logits),
        "config_sha256": _config_hash(),
        "protocol_tag": "v0.2.1-gate2a-mixed-precision-physics-locked",
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, output_dir / filename)


def _write_result_artifacts(result: OptimizationResult, output_dir: Path) -> None:
    pd.DataFrame([asdict(record) for record in result.records]).to_csv(
        output_dir / "optimization_metrics.csv",
        index=False,
    )
    pd.DataFrame([asdict(record) for record in result.solve_records]).to_csv(
        output_dir / "cg_records.csv",
        index=False,
    )
    payload = {
        "schema_version": 2,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "seed": result.seed,
        "run_id": result.run_id,
        "completed_iterations": result.completed_iterations,
        "initial_logits_sha256": result.initial_logits_hash,
        "final_logits_sha256": result.final_logits_hash,
        "continuous_material_fraction": result.continuous_material_fraction,
        "binary_material_fraction": result.binary_material_fraction,
        "config_sha256": result.config_sha256,
        "protocol_tag": "v0.2.1-gate2a-mixed-precision-physics-locked",
    }
    if result.continuous_design is not None and result.binary_design is not None:
        np.save(
            output_dir / "design_continuous_64.npy",
            result.continuous_design.numpy(),
            allow_pickle=False,
        )
        np.save(
            output_dir / "design_binary_64.npy",
            result.binary_design.numpy(),
            allow_pickle=False,
        )
        payload["continuous_design_sha256"] = array_sha256(result.continuous_design)
        payload["binary_design_sha256"] = array_sha256(result.binary_design)
    (output_dir / "optimization_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _failed_result(
    *,
    seed: int,
    mode: OptimizationMode,
    objective_scope: ObjectiveScope,
    reason_code: str,
    records: list[IterationRecord],
    solve_records: list[SolveRecord],
    initial_logits: Tensor,
    logits: Tensor,
) -> OptimizationResult:
    return OptimizationResult(
        status=Gate2Status.INVALID_RUN,
        reason_codes=(reason_code,),
        seed=seed,
        run_id=(f"gate2a_mixed_precision_v1_{mode}_{objective_scope}_seed_{seed}"),
        completed_iterations=len(records),
        records=tuple(records),
        solve_records=tuple(solve_records),
        initial_logits=initial_logits.detach().cpu(),
        final_logits=logits.detach().cpu(),
        initial_logits_hash=array_sha256(initial_logits),
        final_logits_hash=array_sha256(logits),
        continuous_design=None,
        binary_design=None,
        continuous_material_fraction=None,
        binary_material_fraction=None,
        config_sha256=_config_hash(),
    )


def optimize_design(
    sources: Tensor,
    *,
    seed: int,
    config: OptimizationConfig,
    output_dir: Path | None,
) -> OptimizationResult:
    """Optimize one registered seed and return an explicit machine status."""
    device = sources.device
    logits = initialize_logits(seed, device=device).requires_grad_(True)
    initial_logits = logits.detach().clone()
    optimizer = torch.optim.Adam([logits], lr=config.learning_rate)
    records: list[IterationRecord] = []
    solve_records: list[SolveRecord] = []
    best_exact_peak = float("inf")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_checkpoint(
            output_dir,
            "initial_logits.pt",
            seed=seed,
            iteration=-1,
            logits=initial_logits,
            optimizer=None,
        )

    try:
        grid = _validate_sources(sources)
        for iteration in range(config.iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            beta = beta_for_iteration(iteration)
            alpha = alpha_for_iteration(iteration)
            binarization_weight = binarization_weight_for_iteration(iteration)
            parameterized = parameterize_design(logits, beta=beta)
            design = parameterized.design
            conductivity = 1.0 + 19.0 * design.to(torch.float64) ** 3
            trace = SolveTrace()
            temperatures = solve_steady_implicit(
                conductivity,
                sources,
                grid,
                config=config.cg_config,
                trace=trace,
            )
            components = objective_components(
                temperatures,
                design,
                alpha=alpha,
                tv_weight=1.0e-3,
                binarization_weight=binarization_weight,
            )
            components.total.backward()
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise FloatingPointError("optimizer gradient is missing or non-finite")
            if logits.grad.dtype is not torch.float32:
                raise FloatingPointError("optimizer gradient must return to float32")
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    [logits],
                    max_norm=config.gradient_clip_norm,
                    error_if_nonfinite=True,
                ).item()
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            binary = binary_design(design)
            exact_peak = float(components.exact_peak.detach().item())
            best_exact_peak = min(best_exact_peak, exact_peak)
            iteration_solve_records = tuple(trace.records)
            solve_records.extend(iteration_solve_records)
            records.append(
                IterationRecord(
                    iteration=iteration,
                    beta=beta,
                    alpha=alpha,
                    binarization_weight=binarization_weight,
                    total_objective=float(components.total.detach().item()),
                    thermal_smooth=float(components.thermal_smooth.detach().item()),
                    exact_peak=exact_peak,
                    total_variation=float(components.total_variation.detach().item()),
                    binarization_penalty=float(
                        components.binarization_penalty.detach().item()
                    ),
                    continuous_material_fraction=material_fraction(design.detach()),
                    binary_material_fraction=material_fraction(binary.detach()),
                    gradient_norm_before_clipping=gradient_norm,
                    best_exact_peak=best_exact_peak,
                    maximum_cg_iterations=max(
                        record.iterations for record in iteration_solve_records
                    ),
                    maximum_explicit_relative_residual=max(
                        record.relative_residual for record in iteration_solve_records
                    ),
                    wall_seconds=elapsed,
                )
            )
            if (
                output_dir is not None
                and (iteration + 1) % config.checkpoint_interval == 0
            ):
                _write_checkpoint(
                    output_dir,
                    f"checkpoint_{iteration + 1:04d}.pt",
                    seed=seed,
                    iteration=iteration,
                    logits=logits,
                    optimizer=optimizer,
                )
    except CGConvergenceError:
        if "trace" in locals():
            solve_records.extend(trace.records)
        result = _failed_result(
            seed=seed,
            mode=config.mode,
            objective_scope=config.objective_scope,
            reason_code="CG_NONCONVERGENCE",
            records=records,
            solve_records=solve_records,
            initial_logits=initial_logits,
            logits=logits,
        )
        if output_dir is not None:
            _write_result_artifacts(result, output_dir)
        return result
    except (FloatingPointError, VolumeProjectionError, ValueError) as error:
        if "trace" in locals():
            solve_records.extend(trace.records)
        result = _failed_result(
            seed=seed,
            mode=config.mode,
            objective_scope=config.objective_scope,
            reason_code=f"NUMERICAL_FAILURE:{type(error).__name__}",
            records=records,
            solve_records=solve_records,
            initial_logits=initial_logits,
            logits=logits,
        )
        if output_dir is not None:
            _write_result_artifacts(result, output_dir)
        return result

    final_beta = beta_for_iteration(config.iterations - 1)
    final_parameterized = parameterize_design(logits.detach(), beta=final_beta)
    continuous = final_parameterized.design.detach()
    binary = binary_design(continuous)
    continuous_fraction = material_fraction(continuous)
    binary_fraction = material_fraction(binary)
    if config.enforce_final_binary_budget and not binary_budget_satisfied(binary):
        status = Gate2Status.NO_GO_EFFECT
        reason_codes = ("BINARY_BUDGET_FAILURE",)
    else:
        status = Gate2Status.PASS
        reason_codes = ()
    result = OptimizationResult(
        status=status,
        reason_codes=reason_codes,
        seed=seed,
        run_id=(
            "gate2a_mixed_precision_v1_"
            f"{config.mode}_{config.objective_scope}_seed_{seed}"
        ),
        completed_iterations=len(records),
        records=tuple(records),
        solve_records=tuple(solve_records),
        initial_logits=initial_logits.detach().cpu(),
        final_logits=logits.detach().cpu(),
        initial_logits_hash=array_sha256(initial_logits),
        final_logits_hash=array_sha256(logits),
        continuous_design=continuous.cpu(),
        binary_design=binary.cpu(),
        continuous_material_fraction=continuous_fraction,
        binary_material_fraction=binary_fraction,
        config_sha256=_config_hash(),
    )
    if output_dir is not None:
        _write_checkpoint(
            output_dir,
            "checkpoint_final.pt",
            seed=seed,
            iteration=config.iterations - 1,
            logits=logits,
            optimizer=optimizer,
        )
        _write_result_artifacts(result, output_dir)
    return result
