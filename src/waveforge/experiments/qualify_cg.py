"""Pre-production CUDA float64 CG stress qualification."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from torch import Tensor

from waveforge.design.baselines import (
    dispersed_baseline,
    stable_top_k_mask,
    straight_path_baseline,
)
from waveforge.design.objectives import normalized_smooth_max
from waveforge.design.parameterization import filter_logits, parameterize_design
from waveforge.design.scenarios import area_overlap_rectangular_source
from waveforge.physics.cg import CGConfig, CGConvergenceError, CGResult, solve_cg
from waveforge.physics.grid import Grid2D
from waveforge.physics.torch_operator import (
    apply_steady_operator,
    operator_diagonal,
)


class QualificationStatus(StrEnum):
    """Numerical-only qualification outcome."""

    PASS = "PASS"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class QualificationFixture:
    """One immutable conductivity family member."""

    fixture_id: str
    conductivity: Tensor
    fixture_hash: str


@dataclass(frozen=True)
class QualificationRecord:
    """One explicit-residual-qualified forward or adjoint solve."""

    fixture_id: str
    fixture_hash: str
    role: Literal["forward", "adjoint"]
    scenario_index: int
    iterations: int
    explicit_relative_residual: float
    converged: bool
    reason: str
    wall_seconds: float
    dtype: str
    device: str


@dataclass(frozen=True)
class QualificationReport:
    """Machine-readable stress-suite result before optimization."""

    status: QualificationStatus
    fixture_ids: tuple[str, ...]
    records: tuple[QualificationRecord, ...]
    config_sha256: str
    reason_codes: tuple[str, ...] = ()


def _tensor_hash(tensor: Tensor) -> str:
    values = tensor.detach().cpu().numpy().astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _fixture(fixture_id: str, conductivity: Tensor) -> QualificationFixture:
    return QualificationFixture(
        fixture_id=fixture_id,
        conductivity=conductivity,
        fixture_hash=_tensor_hash(conductivity),
    )


def _conductivity_from_design(design: Tensor) -> Tensor:
    design_double = design.to(torch.float64)
    return 1.0 + 19.0 * design_double**3


def qualification_fixtures(
    grid: Grid2D,
    *,
    device: torch.device,
) -> tuple[QualificationFixture, ...]:
    """Create the literal mixed-precision stress registry in stable order."""
    if grid.shape != (64, 64):
        raise ValueError("CG qualification is locked to the 64x64 grid")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("mixed-precision qualification requires CUDA")

    fixtures: list[QualificationFixture] = []
    fixtures.append(
        _fixture(
            "uniform_k1",
            torch.ones(grid.shape, dtype=torch.float64, device=device),
        )
    )
    fixtures.append(
        _fixture(
            "uniform_k20",
            torch.full(grid.shape, 20.0, dtype=torch.float64, device=device),
        )
    )

    smooth_latent = torch.as_tensor(
        np.random.default_rng(9401).normal(size=(16, 16)),
        dtype=torch.float32,
        device=device,
    )
    smooth_upsampled = functional.interpolate(
        smooth_latent[None, None],
        size=grid.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    smooth_design = torch.sigmoid(
        filter_logits(
            smooth_upsampled,
            sigma=1.0,
            radius=3,
            padding="reflect",
        )
    )
    fixtures.append(
        _fixture(
            "smooth_random_seed_9401",
            _conductivity_from_design(smooth_design),
        )
    )

    high_contrast_values = np.random.default_rng(9402).normal(size=grid.shape)
    high_contrast_design = torch.as_tensor(
        stable_top_k_mask(high_contrast_values, count=1024),
        dtype=torch.float64,
        device=device,
    )
    fixtures.append(
        _fixture(
            "high_contrast_random_seed_9402",
            _conductivity_from_design(high_contrast_design),
        )
    )

    for fixture_id, design in (
        ("straight_path_binary", straight_path_baseline(grid).design),
        ("dispersed_binary", dispersed_baseline(grid).design),
    ):
        fixtures.append(
            _fixture(
                fixture_id,
                _conductivity_from_design(
                    torch.as_tensor(
                        np.array(design, copy=True),
                        dtype=torch.float64,
                        device=device,
                    )
                ),
            )
        )

    initial_logits = torch.as_tensor(
        np.random.default_rng(20260828).normal(
            loc=0.0,
            scale=0.1,
            size=(16, 16),
        ),
        dtype=torch.float32,
        device=device,
    )
    for beta in (1.0, 2.0, 4.0, 8.0):
        projected = parameterize_design(initial_logits, beta=beta).design
        fixtures.append(
            _fixture(
                f"projected_beta_{int(beta)}",
                _conductivity_from_design(projected),
            )
        )
    return tuple(fixtures)


def _sources(grid: Grid2D, device: torch.device) -> Tensor:
    bounds = (
        (0.40, 0.60, 0.62, 0.82),
        (0.18, 0.38, 0.62, 0.82),
        (0.62, 0.82, 0.62, 0.82),
    )
    values = np.stack(
        [area_overlap_rectangular_source(grid, item, 1.0) for item in bounds]
    )
    return torch.as_tensor(values, dtype=torch.float64, device=device)


def _config_hash() -> str:
    repository_root = Path(__file__).resolve().parents[3]
    return hashlib.sha256(
        (repository_root / "configs" / "inverse_design.yaml").read_bytes()
    ).hexdigest()


def _timed_solve(
    apply: Callable[[Tensor], Tensor],
    diagonal: Tensor,
    rhs: Tensor,
    config: CGConfig,
) -> tuple[CGResult, float]:
    torch.cuda.synchronize(rhs.device)
    started = time.perf_counter()
    result = solve_cg(apply, diagonal, rhs, config)
    torch.cuda.synchronize(rhs.device)
    return result, time.perf_counter() - started


def _record(
    fixture: QualificationFixture,
    role: Literal["forward", "adjoint"],
    scenario_index: int,
    result: CGResult,
    elapsed: float,
    apply: Callable[[Tensor], Tensor],
    rhs: Tensor,
) -> QualificationRecord:
    explicit_residual = float(
        torch.linalg.vector_norm(rhs - apply(result.solution))
        / max(float(torch.linalg.vector_norm(rhs).item()), 1.0e-12)
    )
    return QualificationRecord(
        fixture_id=fixture.fixture_id,
        fixture_hash=fixture.fixture_hash,
        role=role,
        scenario_index=scenario_index,
        iterations=result.diagnostics.iterations,
        explicit_relative_residual=explicit_residual,
        converged=result.diagnostics.converged and explicit_residual <= 1.0e-6,
        reason=result.diagnostics.reason,
        wall_seconds=elapsed,
        dtype=str(result.solution.dtype).removeprefix("torch."),
        device=str(result.solution.device),
    )


def _operator_for(conductivity: Tensor, grid: Grid2D) -> Callable[[Tensor], Tensor]:
    def apply(temperature: Tensor) -> Tensor:
        return apply_steady_operator(temperature, conductivity, grid)

    return apply


def qualify_mixed_precision_cg(
    *,
    device: torch.device,
) -> QualificationReport:
    """Stress all registered forward/adjoint solves and fail closed."""
    grid = Grid2D(nx=64, ny=64)
    fixtures = qualification_fixtures(grid, device=device)
    sources = _sources(grid, device)
    config = CGConfig()
    records: list[QualificationRecord] = []

    for fixture in fixtures:
        conductivity = fixture.conductivity
        diagonal = operator_diagonal(conductivity, grid)
        apply = _operator_for(conductivity, grid)

        temperatures: list[Tensor] = []
        try:
            for scenario_index, source in enumerate(sources):
                result, elapsed = _timed_solve(apply, diagonal, source, config)
                record = _record(
                    fixture,
                    "forward",
                    scenario_index,
                    result,
                    elapsed,
                    apply,
                    source,
                )
                records.append(record)
                if not record.converged:
                    return QualificationReport(
                        QualificationStatus.INVALID_RUN,
                        tuple(item.fixture_id for item in fixtures),
                        tuple(records),
                        _config_hash(),
                        ("EXPLICIT_RESIDUAL_FAILURE",),
                    )
                temperatures.append(result.solution)

            temperature_batch = torch.stack(temperatures).detach().requires_grad_(True)
            thermal_objective = normalized_smooth_max(temperature_batch, alpha=500.0)
            (adjoint_rhs_batch,) = torch.autograd.grad(
                thermal_objective,
                temperature_batch,
            )
            for scenario_index, adjoint_rhs in enumerate(adjoint_rhs_batch):
                result, elapsed = _timed_solve(apply, diagonal, adjoint_rhs, config)
                record = _record(
                    fixture,
                    "adjoint",
                    scenario_index,
                    result,
                    elapsed,
                    apply,
                    adjoint_rhs,
                )
                records.append(record)
                if not record.converged:
                    return QualificationReport(
                        QualificationStatus.INVALID_RUN,
                        tuple(item.fixture_id for item in fixtures),
                        tuple(records),
                        _config_hash(),
                        ("EXPLICIT_RESIDUAL_FAILURE",),
                    )
        except CGConvergenceError as error:
            records.append(
                QualificationRecord(
                    fixture_id=fixture.fixture_id,
                    fixture_hash=fixture.fixture_hash,
                    role="forward" if len(temperatures) < 3 else "adjoint",
                    scenario_index=len(temperatures) % 3,
                    iterations=error.diagnostics.iterations,
                    explicit_relative_residual=error.diagnostics.relative_residual,
                    converged=False,
                    reason=error.diagnostics.reason,
                    wall_seconds=float("nan"),
                    dtype="float64",
                    device=str(device),
                )
            )
            return QualificationReport(
                QualificationStatus.INVALID_RUN,
                tuple(item.fixture_id for item in fixtures),
                tuple(records),
                _config_hash(),
                ("CG_NONCONVERGENCE",),
            )

    return QualificationReport(
        QualificationStatus.PASS,
        tuple(item.fixture_id for item in fixtures),
        tuple(records),
        _config_hash(),
    )


def write_qualification_artifacts(
    report: QualificationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write schema-versioned CSV/JSON evidence without changing the result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mixed_precision_cg_stress.csv"
    json_path = output_dir / "mixed_precision_cg_stress.json"
    pd.DataFrame([asdict(record) for record in report.records]).to_csv(
        csv_path,
        index=False,
    )
    maximum_residual = max(
        (record.explicit_relative_residual for record in report.records),
        default=float("nan"),
    )
    payload = {
        "schema_version": 2,
        "status": report.status.value,
        "protocol_tag": "v0.2.1-gate2a-mixed-precision-physics-locked",
        "run_namespace": "gate2a_mixed_precision_v1",
        "config_sha256": report.config_sha256,
        "fixture_ids": list(report.fixture_ids),
        "record_count": len(report.records),
        "maximum_explicit_relative_residual": maximum_residual,
        "reason_codes": list(report.reason_codes),
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path
