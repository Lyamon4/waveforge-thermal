"""Machine-readable Gate 2A verdict contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Gate2Status(StrEnum):
    """Mutually exclusive scientific and numerical campaign outcomes."""

    PASS = "PASS"
    NO_GO_EFFECT = "NO_GO_EFFECT"
    INVALID_RUN = "INVALID_RUN"


@dataclass(frozen=True)
class SeedVerdict:
    """Verdict for one registered production seed."""

    status: Gate2Status
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""


@dataclass(frozen=True)
class CampaignVerdict:
    """Aggregate verdict with numerical-invalidity precedence."""

    status: Gate2Status
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""


def classify_campaign(
    *,
    valid: bool,
    passing_seed_count: int,
    required: int,
    metrics: dict[str, Any] | None = None,
    config_hash: str = "",
) -> CampaignVerdict:
    """Classify the campaign without confusing invalidity with no effect."""
    if passing_seed_count < 0 or required < 1:
        raise ValueError("seed counts must be non-negative and required positive")
    shared = {"passing_seed_count": passing_seed_count, "required": required}
    shared.update(metrics or {})
    if not valid:
        return CampaignVerdict(
            status=Gate2Status.INVALID_RUN,
            reason_codes=("NUMERICAL_OR_INTEGRITY_FAILURE",),
            metrics=shared,
            config_hash=config_hash,
        )
    if passing_seed_count < required:
        return CampaignVerdict(
            status=Gate2Status.NO_GO_EFFECT,
            reason_codes=("INSUFFICIENT_PASSING_SEEDS",),
            metrics=shared,
            config_hash=config_hash,
        )
    return CampaignVerdict(
        status=Gate2Status.PASS,
        metrics=shared,
        config_hash=config_hash,
    )
