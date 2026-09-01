from __future__ import annotations

import json
from pathlib import Path

import pytest

from waveforge.experiments.run_mt3_training import (
    MT3ExecutionError,
    MT3QualificationEvaluation,
    build_variant_identity,
    qualification_specs,
    run_qualification_campaign,
    validate_variant_identity,
)


def test_mt3_identity_keeps_test_splits_sealed() -> None:
    identity = build_variant_identity(
        "FIELD_UNET",
        source_sha="a" * 40,
        selected_learning_rate=1.0e-4,
    )

    assert identity["variant"] == "FIELD_UNET"
    assert identity["model_seed"] == 2026092311
    assert identity["task_stream_seed"] == 2026092312
    assert identity["updates"] == 4000
    assert identity["batch_size"] == 4
    assert identity["test_id_accessed"] is False
    assert identity["test_ood_accessed"] is False


def test_mt3_identity_rejects_cross_variant_resume(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(
        json.dumps(
            build_variant_identity(
                "FIELD_UNET",
                source_sha="b" * 40,
                selected_learning_rate=3.0e-4,
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(MT3ExecutionError, match="identity"):
        validate_variant_identity(
            path,
            expected=build_variant_identity(
                "SENS_UNET",
                source_sha="b" * 40,
                selected_learning_rate=3.0e-4,
            ),
        )


def test_mt3_qualification_specs_lock_two_rates_and_two_seeds() -> None:
    specs = qualification_specs()

    assert [
        (row.learning_rate, row.model_seed, row.task_stream_seed) for row in specs
    ] == [
        (1.0e-4, 2026092303, 2026092305),
        (1.0e-4, 2026092304, 2026092306),
        (3.0e-4, 2026092303, 2026092305),
        (3.0e-4, 2026092304, 2026092306),
    ]


def test_mt3_qualification_campaign_writes_sealed_machine_verdict(
    tmp_path: Path,
) -> None:
    trained: list[tuple[float, int, int]] = []

    def train(spec, directory: Path) -> Path:
        trained.append((spec.learning_rate, spec.model_seed, spec.task_stream_seed))
        checkpoint = directory / "checkpoint_000500.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"registered qualification checkpoint")
        return checkpoint

    def evaluate(checkpoint: Path, spec) -> MT3QualificationEvaluation:
        assert checkpoint.name == "checkpoint_000500.pt"
        if spec.learning_rate == 1.0e-4:
            return MT3QualificationEvaluation(True, 0.06, 0.12)
        return MT3QualificationEvaluation(True, 0.04, 0.10)

    verdict = run_qualification_campaign(
        output_root=tmp_path,
        trainer=train,
        evaluator=evaluate,
        source_sha="c" * 40,
        protocol_bundle_sha="d" * 64,
    )

    assert trained == [
        (1.0e-4, 2026092303, 2026092305),
        (1.0e-4, 2026092304, 2026092306),
        (3.0e-4, 2026092303, 2026092305),
        (3.0e-4, 2026092304, 2026092306),
    ]
    assert verdict.production_authorized is True
    assert verdict.selected_learning_rate == 3.0e-4
    payload = json.loads((tmp_path / "qualification_verdict.json").read_text())
    assert payload["selected_learning_rate"] == 3.0e-4
    assert payload["validation_accessed"] is True
    assert payload["test_id_accessed"] is False
    assert payload["test_ood_accessed"] is False
    assert len(payload["rows"]) == 4
