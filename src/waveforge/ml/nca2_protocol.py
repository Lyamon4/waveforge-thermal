"""Immutable machine-readable protocol for prospective NCA-2 stabilization."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from waveforge.ml.nca_protocol import (
    ArchitectureProtocol,
    ConditioningProtocol,
    HashProtocol,
    OptimizerProtocol,
    ParameterizationProtocol,
    PhysicsProtocol,
    PrecisionProtocol,
)


class _LockedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EarlyObjectiveStage(_LockedModel):
    start_inclusive: Literal[0]
    stop_exclusive: Literal[250]
    projection_beta: Literal[2.0]
    smooth_max_alpha: Literal[100.0]
    binarization_weight: Literal[0.0]


class MiddleObjectiveStage(_LockedModel):
    start_inclusive: Literal[250]
    stop_exclusive: Literal[500]
    projection_beta: Literal[4.0]
    smooth_max_alpha: Literal[250.0]
    binarization_weight: Literal[0.01]


class FinalObjectiveStage(_LockedModel):
    start_inclusive: Literal[500]
    stop_exclusive: Literal[1500]
    projection_beta: Literal[8.0]
    smooth_max_alpha: Literal[500.0]
    binarization_weight: Literal[0.02]


class ContinuationProtocol(_LockedModel):
    stages: tuple[EarlyObjectiveStage, MiddleObjectiveStage, FinalObjectiveStage]
    tv_weight: Literal[0.001]
    material_penalty: Literal[0.0]
    target_material_fraction: Literal[0.25]


class ProtocolAStage(_LockedModel):
    start_inclusive: Literal[0]
    stop_exclusive: Literal[1500]
    learning_rate: Literal[0.001]


class ProtocolAEvolution(_LockedModel):
    protocol_id: Literal["A"]
    stages: tuple[ProtocolAStage]


class ProtocolBEarlyStage(_LockedModel):
    start_inclusive: Literal[0]
    stop_exclusive: Literal[250]
    learning_rate: Literal[0.001]


class ProtocolBMiddleStage(_LockedModel):
    start_inclusive: Literal[250]
    stop_exclusive: Literal[500]
    learning_rate: Literal[0.0003]


class ProtocolBFinalStage(_LockedModel):
    start_inclusive: Literal[500]
    stop_exclusive: Literal[1500]
    learning_rate: Literal[0.0001]


class ProtocolBEvolution(_LockedModel):
    protocol_id: Literal["B"]
    stages: tuple[ProtocolBEarlyStage, ProtocolBMiddleStage, ProtocolBFinalStage]


class LearningRateProtocols(_LockedModel):
    protocol_a: ProtocolAEvolution
    protocol_b: ProtocolBEvolution


class DevelopmentProtocol(_LockedModel):
    seeds: tuple[Literal[20260901], Literal[20260902], Literal[20260903]]
    iterations: Literal[700]
    checkpoint_interval: Literal[50]
    evaluation_checkpoints: tuple[
        Literal[500], Literal[550], Literal[600], Literal[650], Literal[700]
    ]
    stable_seeds_required: Literal[2]
    binary_budget_minimum: Literal[0.24]
    binary_budget_maximum: Literal[0.26]
    maximum_late_degradation: Literal[0.05]
    late_degradation_absolute_tolerance: Literal[0.001]
    tmax_relative_tolerance: Literal[0.001]
    practical_tie_protocol: Literal["B"]
    connectivity_has_selection_authority: Literal[False]


class ProductionProtocol(_LockedModel):
    seeds: tuple[Literal[20260911], Literal[20260912], Literal[20260913]]
    iterations: Literal[1500]
    diagnostic_interval: Literal[10]
    checkpoint_interval: Literal[50]
    batch_size: Literal[1]
    early_stopping: Literal[False]
    final_iteration_index: Literal[1499]


class VerificationProtocol(_LockedModel):
    primary_resolution: Literal[256]
    secondary_resolution: Literal[128]
    tree_peak: Literal[0.1650978093408512]
    primary_improvement_fraction: Literal[0.02]
    pass_peak: Literal[0.1617958531540342]
    noncollapse_peak: Literal[0.1683997655276682]
    binary_budget_minimum: Literal[0.24]
    binary_budget_maximum: Literal[0.26]
    passing_seeds_required: Literal[2]
    connectivity_has_primary_authority: Literal[False]
    previous_waveforge: dict[Literal[20260828, 20260829, 20260830], float]


class RuntimeProtocol(_LockedModel):
    warmup_steps: Literal[3]
    measured_steps: Literal[10]
    qualification_updates: Literal[4200]
    production_updates: Literal[4500]
    total_updates: Literal[8700]
    maximum_projected_gpu_hours: Literal[6.6]


class NCA2Protocol(_LockedModel):
    schema_version: Literal[1]
    scope: Literal["pure_nca_fixed_abc_stabilization"]
    architecture: ArchitectureProtocol
    conditioning: ConditioningProtocol
    precision: PrecisionProtocol
    physics: PhysicsProtocol
    parameterization: ParameterizationProtocol
    objective: ContinuationProtocol
    optimizer: OptimizerProtocol
    learning_rate_protocols: LearningRateProtocols
    development: DevelopmentProtocol
    production: ProductionProtocol
    verification: VerificationProtocol
    runtime: RuntimeProtocol
    hashing: HashProtocol
    statuses: tuple[
        Literal["NCA2_STABILITY_GO"],
        Literal["NCA2_NO_GO_EFFECT"],
        Literal["NCA2_INVALID_RUN"],
        Literal["NCA2_QUALIFICATION_FAIL"],
        Literal["NCA2_RUNTIME_REVIEW_REQUIRED"],
    ]


def load_nca2_protocol(path: Path) -> NCA2Protocol:
    """Load and validate the exact prospective NCA-2 protocol."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("NCA-2 protocol root must be a mapping")
    return NCA2Protocol.model_validate(payload)
