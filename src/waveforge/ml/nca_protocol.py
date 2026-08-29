"""Immutable machine-readable protocol for the pure-NCA feasibility spike."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _LockedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ArchitectureProtocol(_LockedModel):
    grid: tuple[Literal[64], Literal[64]]
    mutable_channels: Literal[16]
    material_channel: Literal[0]
    hidden_channels: Literal[15]
    condition_channels: Literal[2]
    perception_width: Literal[64]
    rollout_steps: Literal[64]
    update_scale: Literal[0.1]
    padding_mode: Literal["reflect"]
    activation: Literal["SiLU"]
    parameter_count: Literal[11472]
    initial_state: Literal["zeros"]
    final_layer_initialization: Literal["zeros"]


class ConditioningProtocol(_LockedModel):
    source_aggregation: Literal["sum"]
    fixed_source_scale: Literal[25.0]
    sink_mask: Literal["bottom_cell_row"]
    persistent_every_step: Literal[True]
    clamp_overlaps: Literal[False]


class PrecisionProtocol(_LockedModel):
    neural_dtype: Literal["float32"]
    projection_dtype: Literal["float32"]
    physics_dtype: Literal["float64"]
    device: Literal["cuda"]


class PhysicsProtocol(_LockedModel):
    scenarios: tuple[Literal["A"], Literal["B"], Literal["C"]]
    k_low: Literal[1.0]
    k_high: Literal[20.0]
    interpolation_power: Literal[3.0]
    harmonic_epsilon: Literal[1.0e-12]
    cg_relative_residual_tolerance: Literal[1.0e-6]
    cg_maximum_iterations: Literal[2000]


class ParameterizationProtocol(_LockedModel):
    gaussian_sigma: Literal[1.0]
    gaussian_radius: Literal[3]
    gaussian_padding: Literal["reflect"]
    gaussian_normalization: Literal["unit_sum"]
    projection_bracket: tuple[Literal[-40.0], Literal[40.0]]
    projection_maximum_iterations: Literal[80]
    projection_mean_tolerance: Literal[1.0e-6]
    binary_threshold: Literal[0.5]


class ObjectiveProtocol(_LockedModel):
    projection_beta: Literal[8.0]
    smooth_max_alpha: Literal[500.0]
    tv_weight: Literal[0.001]
    binarization_weight: Literal[0.02]
    material_penalty: Literal[0.0]
    target_material_fraction: Literal[0.25]


class OptimizerProtocol(_LockedModel):
    name: Literal["Adam"]
    betas: tuple[Literal[0.9], Literal[0.999]]
    eps: Literal[1.0e-8]
    weight_decay: Literal[0.0]
    gradient_clip_norm: Literal[1.0]


class QualificationProtocol(_LockedModel):
    seed: Literal[20260831]
    candidate_learning_rates: tuple[
        Literal[0.0003],
        Literal[0.001],
        Literal[0.003],
    ]
    iterations: Literal[200]
    early_window: tuple[Literal[20], Literal[40]]
    late_window: tuple[Literal[180], Literal[200]]
    primary_score_tolerance: Literal[1.0e-4]
    late_loss_relative_tolerance: Literal[1.0e-4]
    volume_mean_tolerance: Literal[1.0e-6]
    initial_gradient_threshold: Literal[1.0e-12]
    maximum_absolute_delta: Literal[0.100001]
    maximum_absolute_state: Literal[6.4001]
    minimum_objective_learning_fraction: Literal[0.01]
    minimum_late_material_std: Literal[0.001]


class ProductionProtocol(_LockedModel):
    seeds: tuple[Literal[20260901], Literal[20260902], Literal[20260903]]
    iterations: Literal[2000]
    diagnostic_interval: Literal[10]
    checkpoint_interval: Literal[100]
    batch_size: Literal[1]
    early_stopping: Literal[False]
    final_iteration_index: Literal[1999]


class VerificationProtocol(_LockedModel):
    primary_resolution: Literal[256]
    secondary_resolution: Literal[128]
    peak_threshold_256: Literal[0.1721575074379424]
    binary_budget_minimum: Literal[0.24]
    binary_budget_maximum: Literal[0.26]
    passing_seeds_required: Literal[2]
    waveforge_reference_seed: Literal[20260828]
    waveforge_reference_peak: Literal[0.156506824943584]
    tree_reference_peak: Literal[0.1650978093408512]
    straight_path_reference_peak: Literal[0.3169417981503212]


class HashProtocol(_LockedModel):
    mode: Literal["canonical_lf_text_raw_binary"]
    text_extensions: tuple[
        Literal[".md"],
        Literal[".json"],
        Literal[".csv"],
        Literal[".yaml"],
        Literal[".yml"],
    ]


class NCAProtocol(_LockedModel):
    schema_version: Literal[1]
    scope: Literal["pure_nca_fixed_abc_feasibility"]
    architecture: ArchitectureProtocol
    conditioning: ConditioningProtocol
    precision: PrecisionProtocol
    physics: PhysicsProtocol
    parameterization: ParameterizationProtocol
    objective: ObjectiveProtocol
    optimizer: OptimizerProtocol
    qualification: QualificationProtocol
    production: ProductionProtocol
    verification: VerificationProtocol
    hashing: HashProtocol
    statuses: tuple[
        Literal["NCA_FEASIBILITY_GO"],
        Literal["NCA_NO_GO_EFFECT"],
        Literal["NCA_SPIKE_INVALID_PRODUCTION_RUN"],
        Literal["NCA_SPIKE_INVALID_REPRODUCIBILITY"],
        Literal["NCA_SPIKE_INVALID_TRAINING_PATHOLOGY"],
        Literal["NCA_QUALIFICATION_NO_ELIGIBLE_LR"],
    ]


def load_nca_protocol(path: Path) -> NCAProtocol:
    """Load and validate the exact preregistered pure-NCA protocol."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("pure-NCA protocol root must be a mapping")
    return NCAProtocol.model_validate(payload)
