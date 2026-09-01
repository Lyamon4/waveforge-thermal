"""Validated machine protocol for the MT3 learned warm-start experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class ArchitectureProtocol(_Section):
    condition_channels: Literal[5]
    candidate_count: Literal[4]
    parameter_count: Literal[2918724]


class RefinementProtocol(_Section):
    candidate_scores: Literal[4]
    selected_candidates: Literal[1]
    primary_steps: Literal[25]
    secondary_steps: Literal[50]
    learning_rate: Literal[0.01]


class TrainingProtocol(_Section):
    model_seed: Literal[2026092311]
    task_stream_seed: Literal[2026092312]
    batch_size: Literal[4]
    updates: Literal[4000]
    checkpoint_interval: Literal[500]


class RuntimeProtocol(_Section):
    current_credit_usd: Literal[2.0]
    benchmark_cost_maximum_usd: Literal[0.20]
    campaign_cost_maximum_usd: Literal[1.70]
    campaign_hours_maximum: Literal[2.5]
    safety_buffer_usd: Literal[0.10]
    long_training_authorized: Literal[False]


class SplitAccessProtocol(_Section):
    validation: Literal["development_only"]
    test_id: Literal["sealed"]
    test_ood: Literal["sealed"]


class MT3Protocol(BaseModel):
    """Strict view of result-sensitive MT3 protocol values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1]
    scope: Literal["mt3_sensitivity_conditioned_learned_warmstart"]
    status: Literal["design_approved_implementation_only"]
    task_distribution: _Section
    architecture: ArchitectureProtocol
    conditioning: _Section
    candidate_training: _Section
    refinement: RefinementProtocol
    training: TrainingProtocol
    qualification: _Section
    validation: _Section
    runtime: RuntimeProtocol
    split_access: SplitAccessProtocol


@dataclass(frozen=True)
class MT3Stage:
    projection_beta: float
    smooth_max_alpha: float
    binarization_weight: float
    tv_weight: float
    learning_rate_multiplier: float


def training_settings_at(update: int) -> MT3Stage:
    """Return the locked zero-based MT3 continuation stage."""
    if 0 <= update < 800:
        return MT3Stage(2.0, 100.0, 0.0, 0.001, 1.0)
    if 800 <= update < 1600:
        return MT3Stage(4.0, 250.0, 0.01, 0.001, 0.3)
    if 1600 <= update < 4000:
        return MT3Stage(8.0, 500.0, 0.02, 0.001, 0.1)
    raise ValueError(f"update {update} is outside MT3 training range [0, 4000)")


def load_mt3_protocol(path: Path) -> MT3Protocol:
    """Load and validate the approved MT3 YAML."""
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return MT3Protocol.model_validate(payload)


def assert_paid_runtime_authorized(
    projected_hours: float,
    hourly_usd: float,
    *,
    credit_usd: float = 2.0,
) -> None:
    """Fail closed when projected A100 work exceeds the current credit guard."""
    values = (projected_hours, hourly_usd, credit_usd)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(
            "runtime, hourly price, and credit must be finite and positive"
        )
    projected_cost = projected_hours * hourly_usd
    cost_limit = min(1.70, credit_usd - 0.10)
    if projected_hours > 2.5 or projected_cost > cost_limit:
        raise RuntimeError("paid runtime is not authorized by the current credit guard")
