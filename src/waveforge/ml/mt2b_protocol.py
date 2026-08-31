"""Immutable machine protocol for the prospective NCA-MT2B experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from waveforge.reproducibility import artifact_sha256

_LOCKED_SEMANTIC_SHA256 = (
    "33285b3f89da444f538dbdb0b07d1d530db9146ce30f3f88e9553876301bb0d4"
)


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class StrataProtocol(_Section):
    horizontal_span_threshold: Literal[0.46]
    vertical_span_threshold: Literal[0.21]


class TaskDistributionProtocol(_Section):
    batch_size: Literal[4]
    strata: StrataProtocol


class ArchitectureProtocol(_Section):
    condition_channels: Literal[4]
    parameter_count: Literal[12624]
    rollout_steps: Literal[64]


class ConditioningProtocol(_Section):
    fixed_temperature_scale: Literal[0.900613256638055]


class ValidationProtocol(_Section):
    candidate_binary_evaluator: Literal["independent_scipy_64"]
    reference_binary_evaluator: Literal["independent_scipy_64"]
    selected_design_secondary_evaluator: Literal["independent_scipy_256"]


class BootstrapProtocol(_Section):
    resamples: Literal[10000]
    seed: Literal[2026092203]
    percentile_bounds: tuple[Literal[0.025], Literal[0.975]]


class SplitAccessProtocol(_Section):
    validation: Literal["development_only"]
    test_id: Literal["sealed"]
    test_ood: Literal["sealed"]


class RuntimeProtocol(_Section):
    benchmark_maximum_a100_hours: Literal[0.5]
    paired_pilot_maximum_projected_a100_hours: Literal[10.0]
    long_training_authorized: Literal[False]


def _semantic_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class MT2BProtocol(BaseModel):
    """Validated view of the byte-independent, semantically locked YAML."""

    model_config = ConfigDict(frozen=True, extra="allow")

    schema_version: Literal[1]
    scope: Literal["nca_mt2b_physics_informed_multitask_ablation"]
    task_distribution: TaskDistributionProtocol
    architecture: ArchitectureProtocol
    conditioning: ConditioningProtocol
    validation: ValidationProtocol
    bootstrap: BootstrapProtocol
    runtime: RuntimeProtocol
    split_access: SplitAccessProtocol

    @model_validator(mode="before")
    @classmethod
    def reject_any_protocol_change(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("MT2B protocol root must be a mapping")
        if _semantic_sha256(value) != _LOCKED_SEMANTIC_SHA256:
            raise ValueError("MT2B protocol differs from the prospective lock")
        return value


@dataclass(frozen=True)
class MT2BStage:
    projection_beta: float
    smooth_max_alpha: float
    binarization_weight: float
    tv_weight: float
    learning_rate: float


def training_settings_at(update: int) -> MT2BStage:
    """Return the locked continuation stage for one zero-based update."""
    if 0 <= update < 400:
        return MT2BStage(2.0, 100.0, 0.0, 0.001, 0.001)
    if 400 <= update < 800:
        return MT2BStage(4.0, 250.0, 0.01, 0.001, 0.0003)
    if 800 <= update < 2000:
        return MT2BStage(8.0, 500.0, 0.02, 0.001, 0.0001)
    raise ValueError(f"update {update} is outside MT2B training range [0, 2000)")


def load_mt2b_protocol(path: Path) -> MT2BProtocol:
    """Load and validate the exact prospective MT2B YAML."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return MT2BProtocol.model_validate(payload)


def protocol_bundle_hash(config_path: Path, spec_path: Path) -> str:
    """Hash the ordered canonical-LF component hashes of the protocol bundle."""
    payload = f"{artifact_sha256(config_path)}\n{artifact_sha256(spec_path)}\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()
