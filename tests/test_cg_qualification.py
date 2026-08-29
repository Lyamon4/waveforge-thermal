"""Pre-production mixed-precision CG stress qualification."""

import json
from pathlib import Path

import numpy as np
import torch

from waveforge.experiments.qualify_cg import (
    QualificationStatus,
    qualification_fixtures,
    qualify_mixed_precision_cg,
    write_qualification_artifacts,
)
from waveforge.physics.grid import Grid2D

EXPECTED_FIXTURE_IDS = (
    "uniform_k1",
    "uniform_k20",
    "smooth_random_seed_9401",
    "high_contrast_random_seed_9402",
    "straight_path_binary",
    "dispersed_binary",
    "projected_beta_1",
    "projected_beta_2",
    "projected_beta_4",
    "projected_beta_8",
)


def test_qualification_registry_covers_every_locked_conductivity_family() -> None:
    """Omitting a contrast or beta stage must fail before production."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"
    fixtures = qualification_fixtures(
        Grid2D(nx=64, ny=64),
        device=torch.device("cuda"),
    )

    assert tuple(fixture.fixture_id for fixture in fixtures) == EXPECTED_FIXTURE_IDS
    assert len({fixture.fixture_hash for fixture in fixtures}) == len(fixtures)
    for fixture in fixtures:
        assert fixture.conductivity.dtype is torch.float64
        assert fixture.conductivity.device.type == "cuda"
        assert torch.isfinite(fixture.conductivity).all()
        assert torch.all(fixture.conductivity >= 1.0)
        assert torch.all(fixture.conductivity <= 20.0)


def test_mixed_precision_cg_stress_suite_passes_all_forward_and_adjoint_solves(
    tmp_path: Path,
) -> None:
    """A residual, iteration-limit, dtype, or artifact failure must invalidate."""
    assert torch.cuda.is_available(), "Gate 2A locked environment requires CUDA"

    report = qualify_mixed_precision_cg(device=torch.device("cuda"))

    assert report.status is QualificationStatus.PASS
    assert tuple(report.fixture_ids) == EXPECTED_FIXTURE_IDS
    assert len(report.records) == 60
    assert sum(record.role == "forward" for record in report.records) == 30
    assert sum(record.role == "adjoint" for record in report.records) == 30
    assert all(record.dtype == "float64" for record in report.records)
    assert all(record.converged for record in report.records)
    assert max(record.explicit_relative_residual for record in report.records) <= 1e-6
    assert max(record.iterations for record in report.records) <= 2000
    assert np.isfinite([record.wall_seconds for record in report.records]).all()

    output_dir = tmp_path / "qualification"
    csv_path, json_path = write_qualification_artifacts(report, output_dir)
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["status"] == "PASS"
    assert payload["config_sha256"] == report.config_sha256
    assert payload["record_count"] == 60
