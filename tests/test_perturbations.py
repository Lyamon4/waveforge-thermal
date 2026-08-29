"""Tests for the pre-registered Gate 2A robustness protocol."""

import numpy as np
import pytest

from waveforge.design.baselines import dispersed_baseline, straight_path_baseline
from waveforge.physics.grid import Grid2D
from waveforge.verification.compare import Gate2Status
from waveforge.verification.perturbations import (
    apply_morphology,
    classify_seed_robustness,
    evaluate_perturbation_case,
    four_neighbor_component_count,
    morphology_diagnostics,
    perturbed_source_batch,
    registered_primary_cases,
)


def test_primary_registry_contains_exactly_28_non_morphological_cases() -> None:
    """Adding, dropping, or mixing morphology into the denominator must fail."""
    cases = registered_primary_cases()

    assert len(cases) == 28
    assert len({case.case_id for case in cases}) == 28
    assert sum(case.kind == "source_shift" for case in cases) == 24
    assert sum(case.kind == "intensity_scale" for case in cases) == 2
    assert sum(case.kind == "conductivity_scale" for case in cases) == 2
    assert not any(
        "erosion" in case.case_id or "dilation" in case.case_id for case in cases
    )


def test_source_shift_moves_only_registered_scenario_and_preserves_power() -> None:
    """Moving all sources or changing integrated power must fail."""
    grid = Grid2D(nx=256, ny=256)
    cases = {case.case_id: case for case in registered_primary_cases()}
    nominal_sources, nominal_k_high = perturbed_source_batch(grid, case=None)

    shifted_sources, shifted_k_high = perturbed_source_batch(
        grid,
        case=cases["shift_B_2_right"],
    )

    assert nominal_k_high == shifted_k_high == 20.0
    np.testing.assert_array_equal(shifted_sources[0], nominal_sources[0])
    np.testing.assert_array_equal(shifted_sources[2], nominal_sources[2])
    assert not np.array_equal(shifted_sources[1], nominal_sources[1])
    integrated = shifted_sources.sum(axis=(1, 2)) * grid.dx * grid.dy
    np.testing.assert_allclose(integrated, np.ones(3), rtol=0.0, atol=1e-14)


def test_morphology_uses_three_by_three_structure_without_budget_repair() -> None:
    """A quantile repair or four-neighbor morphology must fail."""
    design = np.zeros((7, 7), dtype=np.float64)
    design[2:5, 2:5] = 1.0

    eroded = apply_morphology(design, operation="erosion")
    dilated = apply_morphology(design, operation="dilation")

    assert eroded.sum() == 1.0
    assert dilated.sum() == 25.0
    assert eroded.mean() != pytest.approx(design.mean())
    assert dilated.mean() != pytest.approx(design.mean())
    assert four_neighbor_component_count(design) == 1


def test_four_neighbor_components_do_not_connect_diagonal_pixels() -> None:
    """Eight-neighbor connectivity must fail the registered complexity metric."""
    design = np.zeros((3, 3), dtype=np.float64)
    design[0, 0] = 1.0
    design[1, 1] = 1.0

    assert four_neighbor_component_count(design) == 2


def test_robustness_verdict_requires_23_of_28_and_invalidity_precedes_effect() -> None:
    """A 22-case pass or numerical failure masked as NO-GO must fail."""
    passing_23 = [0.02] * 23 + [0.019] * 5
    passing_22 = [0.02] * 22 + [0.019] * 6

    assert classify_seed_robustness(passing_23, valid=True).status is Gate2Status.PASS
    assert (
        classify_seed_robustness(passing_22, valid=True).status
        is Gate2Status.NO_GO_EFFECT
    )
    assert (
        classify_seed_robustness(passing_23, valid=False).status
        is Gate2Status.INVALID_RUN
    )


def test_perturbation_evaluation_and_morphology_report_independent_256_metrics() -> (
    None
):
    """Low-fidelity peaks or repaired morphology budgets must fail."""
    grid = Grid2D(nx=64, ny=64)
    cases = {case.case_id: case for case in registered_primary_cases()}
    dispersed = dispersed_baseline(grid).design

    evaluation = evaluate_perturbation_case(
        "dispersed",
        dispersed,
        cases["intensity_plus_5pct"],
    )
    morphology = morphology_diagnostics(
        "straight",
        straight_path_baseline(grid).design,
    )

    assert evaluation.grid_shape == (256, 256)
    assert evaluation.case_id == "intensity_plus_5pct"
    assert len(evaluation.scenario_peaks) == 3
    assert np.isfinite(evaluation.worst_peak)
    assert evaluation.material_fraction == pytest.approx(0.25)
    assert tuple(record.operation for record in morphology) == (
        "unperturbed",
        "erosion",
        "dilation",
    )
    assert morphology[1].material_fraction < morphology[0].material_fraction
    assert morphology[2].material_fraction > morphology[0].material_fraction
    assert all(np.isfinite(record.worst_peak_256) for record in morphology)
