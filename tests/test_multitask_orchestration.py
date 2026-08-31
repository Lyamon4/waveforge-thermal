"""Tests for paid-A100 campaign gates and CLI orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.experiments.run_multitask_nca import (
    BenchmarkCandidate,
    MultitaskGateError,
    PilotStatus,
    build_parser,
    calculate_runtime_gate,
    classify_pilot,
    select_microbatch,
    validate_production_gate,
)


def test_runtime_gate_requires_at_least_5000_updates_per_seed() -> None:
    verdict = calculate_runtime_gate(
        seconds_per_update=2.0,
        remaining_hours=6.0,
    )
    assert verdict.production_authorized is False
    assert verdict.updates_per_seed == 3600


def test_runtime_gate_caps_each_seed_at_15000_updates() -> None:
    verdict = calculate_runtime_gate(
        seconds_per_update=0.1,
        remaining_hours=6.0,
    )
    assert verdict.production_authorized is True
    assert verdict.updates_per_seed == 15_000


def test_microbatch_selection_uses_throughput_and_two_percent_tie_break() -> None:
    selected = select_microbatch(
        [
            BenchmarkCandidate(1, 1.0, 1.1, 1.00, True, 100),
            BenchmarkCandidate(2, 1.9, 2.0, 1.04, True, 200),
            BenchmarkCandidate(4, 3.8, 4.0, 1.052, True, 400),
        ]
    )
    assert selected.microbatch_size == 2


def test_invalid_benchmark_candidate_cannot_be_selected() -> None:
    selected = select_microbatch(
        [
            BenchmarkCandidate(1, 1.0, 1.1, 1.0, True, 100),
            BenchmarkCandidate(4, 1.0, 1.1, 4.0, False, 400),
        ]
    )
    assert selected.microbatch_size == 1


def test_pilot_gate_distinguishes_go_conditional_kill_and_invalid() -> None:
    common = {
        "numerically_valid": True,
        "projection_valid": True,
        "binary_budget_valid": True,
        "validation_improved": True,
        "matched_condition_wins": 24,
        "source_independent": False,
    }
    assert classify_pilot(median_gradient_gap=0.14, **common) is PilotStatus.PILOT_GO
    assert (
        classify_pilot(median_gradient_gap=0.18, **common)
        is PilotStatus.PILOT_CONDITIONAL
    )
    assert classify_pilot(median_gradient_gap=0.21, **common) is PilotStatus.PILOT_KILL
    assert (
        classify_pilot(
            median_gradient_gap=0.1, **(common | {"numerically_valid": False})
        )
        is PilotStatus.INVALID_RUN
    )


def test_production_requires_benchmark_and_pilot_go(tmp_path: Path) -> None:
    with pytest.raises(MultitaskGateError, match="benchmark"):
        validate_production_gate(tmp_path)

    (tmp_path / "benchmark_verdict.json").write_text(
        json.dumps({"production_runtime_authorized": True}),
        encoding="utf-8",
    )
    with pytest.raises(MultitaskGateError, match="pilot"):
        validate_production_gate(tmp_path)

    (tmp_path / "pilot_verdict.json").write_text(
        json.dumps({"status": "PILOT_CONDITIONAL"}),
        encoding="utf-8",
    )
    with pytest.raises(MultitaskGateError, match="PILOT_GO"):
        validate_production_gate(tmp_path)

    (tmp_path / "pilot_verdict.json").write_text(
        json.dumps({"status": "PILOT_GO"}),
        encoding="utf-8",
    )
    validate_production_gate(tmp_path)


@pytest.mark.parametrize(
    "phase",
    ["preflight", "benchmark", "pilot", "production", "test", "hashes"],
)
def test_parser_accepts_every_locked_phase(phase: str, tmp_path: Path) -> None:
    parser = build_parser()
    parsed = parser.parse_args(["--phase", phase, "--output", str(tmp_path)])
    assert parsed.phase == phase
