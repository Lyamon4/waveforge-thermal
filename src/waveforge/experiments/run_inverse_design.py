"""Gate 2A benchmark, smoke, and guarded production entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from waveforge.design.optimize import (
    OptimizationConfig,
    OptimizationResult,
    optimize_design,
)
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.grid import Grid2D
from waveforge.verification.compare import Gate2Status


def gate2_source_batch(*, device: torch.device) -> Tensor:
    """Create the three exact-area-overlap registered source scenarios."""
    grid = Grid2D(nx=64, ny=64)
    bounds = (
        (0.40, 0.60, 0.62, 0.82),
        (0.18, 0.38, 0.62, 0.82),
        (0.62, 0.82, 0.62, 0.82),
    )
    sources = np.stack(
        [area_overlap_rectangular_source(grid, item, 1.0) for item in bounds]
    )
    return torch.as_tensor(sources, dtype=torch.float64, device=device)


def benchmark_full_optimization_step(output_path: Path) -> dict[str, object]:
    """Measure one complete three-scenario forward/adjoint Adam step."""
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 2A locked environment requires CUDA")
    device = torch.device("cuda")
    sources = gate2_source_batch(device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result = optimize_design(
        sources,
        seed=20260828,
        config=OptimizationConfig(
            iterations=1,
            mode="benchmark",
            enforce_final_binary_budget=False,
        ),
        output_dir=None,
    )
    torch.cuda.synchronize(device)
    forward_records = [
        record for record in result.solve_records if record.role == "forward"
    ]
    adjoint_records = [
        record for record in result.solve_records if record.role == "adjoint"
    ]
    maximum_residual = max(
        (record.relative_residual for record in result.solve_records),
        default=float("inf"),
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "status": result.status.value,
        "reason_codes": list(result.reason_codes),
        "run_id": result.run_id,
        "config_sha256": result.config_sha256,
        "protocol_tag": "v0.2.1-gate2a-mixed-precision-physics-locked",
        "design_dtype": "float32",
        "physics_solve_dtype": "float64",
        "gradient_dtype": "float32",
        "forward_solve_count": len(forward_records),
        "adjoint_solve_count": len(adjoint_records),
        "maximum_cg_iterations": max(
            (record.iterations for record in result.solve_records),
            default=0,
        ),
        "maximum_explicit_relative_residual": maximum_residual,
        "step_wall_seconds": (
            result.records[0].wall_seconds if result.records else float("nan")
        ),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def run_smoke(output_dir: Path) -> OptimizationResult:
    """Run the registered ten-iteration numerical smoke optimization."""
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 2A locked environment requires CUDA")
    return optimize_design(
        gate2_source_batch(device=torch.device("cuda")),
        seed=20260828,
        config=OptimizationConfig(
            iterations=10,
            mode="smoke",
            enforce_final_binary_budget=False,
        ),
        output_dir=output_dir,
    )


def _load_pass_artifact(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required preflight artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != Gate2Status.PASS.value:
        raise RuntimeError(f"preflight artifact is not PASS: {path}")


def run_production(seed: int, output_dir: Path) -> OptimizationResult:
    """Run one 600-step seed only after all machine preflights are present."""
    if seed not in (20260828, 20260829, 20260830):
        raise ValueError("seed is not one of the three registered production seeds")
    repository_root = Path(__file__).resolve().parents[3]
    preflight = repository_root / "artifacts" / "gate2_design" / "preflight"
    for filename in (
        "mixed_precision_cg_stress.json",
        "gradient_validation_cpu.json",
        "gradient_validation_cuda.json",
        "full_iteration_benchmark.json",
    ):
        _load_pass_artifact(preflight / filename)
    return optimize_design(
        gate2_source_batch(device=torch.device("cuda")),
        seed=seed,
        config=OptimizationConfig(),
        output_dir=output_dir,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("benchmark", "smoke", "production"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    """Run one explicitly selected Gate 2A phase."""
    args = _parse_args()
    if args.mode == "benchmark":
        benchmark_full_optimization_step(args.output)
    elif args.mode == "smoke":
        result = run_smoke(args.output)
        if result.status is Gate2Status.INVALID_RUN:
            raise SystemExit(2)
    else:
        if args.seed is None:
            raise SystemExit("--seed is required for production mode")
        result = run_production(args.seed, args.output)
        if result.status is Gate2Status.INVALID_RUN:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
