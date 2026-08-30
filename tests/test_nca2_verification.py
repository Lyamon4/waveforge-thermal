"""Independent thermal verdict contracts for stabilized NCA-2."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from waveforge.experiments.run_nca2_stabilization import run_verification_phase
from waveforge.reproducibility import content_hash
from waveforge.verification.nca2_verification import (
    NCA2SeedVerdict,
    classify_nca2_campaign,
)


def _seed(
    seed: int,
    peak: float,
    *,
    fraction: float = 0.25,
    valid: bool = True,
) -> NCA2SeedVerdict:
    return NCA2SeedVerdict.classify(
        seed=seed,
        peak_256=peak,
        binary_fraction=fraction,
        numerically_valid=valid,
    )


@pytest.mark.parametrize("fraction", [0.24, 0.26])
def test_disconnected_thermally_strong_seed_passes_inclusive_primary_gate(
    fraction: float,
) -> None:
    verdict = _seed(20260911, 0.1617958531540342, fraction=fraction)

    assert verdict.primary_pass is True
    assert verdict.noncollapse_pass is True
    assert verdict.budget_pass is True


def test_seed_boundaries_fail_immediately_above_locked_values() -> None:
    effect = _seed(20260911, 0.1617958531540342 + 1.0e-15)
    noncollapse = _seed(20260911, 0.1683997655276682 + 1.0e-15)
    budget = _seed(20260911, 0.16, fraction=0.2600001)

    assert not effect.primary_pass
    assert effect.noncollapse_pass
    assert not noncollapse.noncollapse_pass
    assert not budget.budget_pass


def test_campaign_verdict_has_invalidity_and_noncollapse_precedence() -> None:
    passing = _seed(20260911, 0.16)
    passing_two = _seed(20260912, 0.161)
    boundary = _seed(20260913, 0.1683997655276682)

    go = classify_nca2_campaign((passing, passing_two, boundary))
    assert go.status == "NCA2_STABILITY_GO"
    assert go.passing_seed_count == 2

    collapsed = classify_nca2_campaign((passing, passing_two, _seed(20260913, 0.1685)))
    assert collapsed.status == "NCA2_NO_GO_EFFECT"
    assert "CATASTROPHIC_COLLAPSE" in collapsed.reason_codes

    invalid = classify_nca2_campaign(
        (passing, passing_two, _seed(20260913, 0.16, valid=False))
    )
    assert invalid.status == "NCA2_INVALID_RUN"


def test_campaign_reports_all_seed_statistics_not_only_best() -> None:
    campaign = classify_nca2_campaign(
        (
            _seed(20260911, 0.15),
            _seed(20260912, 0.16),
            _seed(20260913, 0.165),
        )
    )

    assert campaign.mean_peak_256 == pytest.approx(0.15833333333333333)
    assert campaign.median_peak_256 == pytest.approx(0.16)
    assert campaign.minimum_peak_256 == pytest.approx(0.15)
    assert campaign.maximum_peak_256 == pytest.approx(0.165)


def test_connectivity_cannot_change_primary_campaign_verdict() -> None:
    verdicts = (
        _seed(20260911, 0.15),
        _seed(20260912, 0.16),
        _seed(20260913, 0.165),
    )

    assert classify_nca2_campaign(verdicts).status == "NCA2_STABILITY_GO"


def test_verification_phase_reports_all_three_seeds_and_comparators(tmp_path) -> None:
    design = np.zeros((64, 64), dtype=np.float64)
    design[:, :16] = 1.0
    for seed in (20260911, 20260912, 20260913):
        run_dir = tmp_path / f"production_seed_{seed}"
        run_dir.mkdir()
        np.save(run_dir / "design_continuous_64.npy", design, allow_pickle=False)
        np.save(run_dir / "design_binary_64.npy", design, allow_pickle=False)
        (run_dir / "production_manifest.json").write_text(
            json.dumps(
                {
                    "status": "VALID_PRODUCTION_RUN",
                    "seed": seed,
                    "continuous_design_sha256": content_hash(design),
                    "binary_design_sha256": content_hash(design),
                }
            ),
            encoding="utf-8",
        )

    peaks = {20260911: 0.15, 20260912: 0.16, 20260913: 0.165}

    def fake_verifier(**kwargs):
        seed = kwargs["seed"]
        peak = peaks[seed]
        records = tuple(
            SimpleNamespace(scenario_id=scenario, peak_temperature=peak - index * 0.01)
            for index, scenario in enumerate(("A", "B", "C"))
        )
        grid_128 = SimpleNamespace(
            worst_peak=peak + 0.001,
            average_peak=peak - 0.009,
            protected_zone_peak=peak - 0.02,
            material_fraction=0.25,
            total_wall_seconds=0.1,
            scenario_records=records,
        )
        grid_256 = SimpleNamespace(
            worst_peak=peak,
            average_peak=peak - 0.01,
            protected_zone_peak=peak - 0.02,
            material_fraction=0.25,
            total_wall_seconds=0.2,
            scenario_records=records,
        )
        connectivity = SimpleNamespace(
            conductive_cell_count=1024,
            component_count=2,
            sink_connected_cell_count=1000,
            sink_connected_fraction=1000 / 1024,
            source_intersection_cell_counts={"A": 10, "B": 10, "C": 10},
            sink_component_source_intersections={"A": True, "B": True, "C": True},
        )
        return SimpleNamespace(
            seed=seed,
            verification_128=grid_128,
            verification_256=grid_256,
            relative_128_to_256_change=0.001 / peak,
            connectivity=connectivity,
            engineering_connectivity_pass=True,
            verdict=_seed(seed, peak),
            previous_waveforge_relative_differences={
                20260828: 0.01,
                20260829: 0.02,
                20260830: 0.03,
            },
        )

    payload = run_verification_phase(
        tmp_path,
        verifier=fake_verifier,
        gate_validator=lambda _: None,
    )

    assert payload["campaign"]["status"] == "NCA2_STABILITY_GO"
    metrics = pd.read_csv(tmp_path / "verified_256_metrics.csv")
    comparisons = pd.read_csv(tmp_path / "comparator_metrics.csv")
    assert metrics["seed"].tolist() == [20260911, 20260912, 20260913]
    assert len(comparisons) == 12
    assert set(comparisons["comparator_id"]) == {
        "parametric_branching_tree",
        "waveforge_20260828",
        "waveforge_20260829",
        "waveforge_20260830",
    }
