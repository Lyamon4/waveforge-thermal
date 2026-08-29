"""Tests for Gate 2A verdict precedence and plotting purity."""

from pathlib import Path

import numpy as np

from waveforge.reporting.figures import (
    save_design_animation,
    save_metric_curves,
    save_multi_field_figure,
)
from waveforge.reporting.summary import final_gate2a_verdict
from waveforge.verification.compare import Gate2Status, SeedVerdict


def _seed(status: Gate2Status) -> SeedVerdict:
    return SeedVerdict(status=status)


def test_final_verdict_requires_two_nominal_and_robustness_passes() -> None:
    """A seed only counts when both locked scientific criteria pass."""
    nominal = {
        1: _seed(Gate2Status.PASS),
        2: _seed(Gate2Status.PASS),
        3: _seed(Gate2Status.PASS),
    }
    one_robust = {
        1: _seed(Gate2Status.PASS),
        2: _seed(Gate2Status.NO_GO_EFFECT),
        3: _seed(Gate2Status.NO_GO_EFFECT),
    }
    two_robust = {
        1: _seed(Gate2Status.PASS),
        2: _seed(Gate2Status.PASS),
        3: _seed(Gate2Status.NO_GO_EFFECT),
    }

    assert (
        final_gate2a_verdict(nominal, one_robust, mandatory_valid=True).status
        is Gate2Status.NO_GO_EFFECT
    )
    assert (
        final_gate2a_verdict(nominal, two_robust, mandatory_valid=True).status
        is Gate2Status.PASS
    )


def test_final_verdict_invalidity_precedes_good_effect() -> None:
    """A numerical, integrity, or mandatory-artifact failure is INVALID_RUN."""
    passing = {seed: _seed(Gate2Status.PASS) for seed in (1, 2, 3)}
    invalid = dict(passing)
    invalid[3] = _seed(Gate2Status.INVALID_RUN)

    assert (
        final_gate2a_verdict(passing, passing, mandatory_valid=False).status
        is Gate2Status.INVALID_RUN
    )
    assert (
        final_gate2a_verdict(invalid, passing, mandatory_valid=True).status
        is Gate2Status.INVALID_RUN
    )


def test_gate2_plotting_does_not_modify_scientific_inputs(tmp_path: Path) -> None:
    """Plotting or GIF generation must be scientifically side-effect free."""
    fields = np.arange(3 * 8 * 8, dtype=np.float64).reshape(3, 8, 8)
    trajectory = np.linspace(0.0, 1.0, 5 * 8 * 8).reshape(5, 8, 8)
    iterations = np.arange(5, dtype=np.float64)
    curves = np.stack((iterations, iterations**2))
    fields_before = fields.copy()
    trajectory_before = trajectory.copy()
    curves_before = curves.copy()

    save_multi_field_figure(
        fields,
        ("A", "B", "C"),
        tmp_path / "fields.png",
        title="Designs",
        cmap="viridis",
        colorbar_label="D",
    )
    save_metric_curves(
        iterations,
        curves,
        ("seed 1", "seed 2"),
        tmp_path / "curves.png",
        title="Objective",
        ylabel="J",
    )
    save_design_animation(
        trajectory,
        np.arange(5),
        tmp_path / "design.gif",
    )

    np.testing.assert_array_equal(fields, fields_before)
    np.testing.assert_array_equal(trajectory, trajectory_before)
    np.testing.assert_array_equal(curves, curves_before)
    assert (tmp_path / "fields.png").stat().st_size > 0
    assert (tmp_path / "curves.png").stat().st_size > 0
    assert (tmp_path / "design.gif").stat().st_size > 0
