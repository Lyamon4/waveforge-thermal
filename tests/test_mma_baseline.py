from __future__ import annotations

import numpy as np
import pytest

from waveforge.design.mma_baseline import (
    mma_objective_callback,
    optimize_mma,
    qualify_mma_backend,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def test_mma_callback_matches_directional_finite_difference() -> None:
    task = sample_primary_task(2026092317, 0)
    logits = np.random.default_rng(2026092318).normal(0.0, 0.05, size=256)
    direction = np.random.default_rng(2026092319).normal(size=256)
    direction /= np.linalg.norm(direction)

    value, gradient = mma_objective_callback(
        logits,
        task,
        beta=8.0,
        alpha=500.0,
        binarization_weight=0.02,
        allow_cpu_unit_test=True,
    )
    # Float32 design variables need a macroscopic directional step so the
    # finite-difference subtraction is not dominated by rounding noise.
    epsilon = 5.0e-2
    plus, _ = mma_objective_callback(
        logits + epsilon * direction,
        task,
        beta=8.0,
        alpha=500.0,
        binarization_weight=0.02,
        allow_cpu_unit_test=True,
    )
    minus, _ = mma_objective_callback(
        logits - epsilon * direction,
        task,
        beta=8.0,
        alpha=500.0,
        binarization_weight=0.02,
        allow_cpu_unit_test=True,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)

    assert np.isfinite(value)
    assert gradient.shape == (256,)
    assert np.isfinite(gradient).all()
    assert float(np.dot(gradient, direction)) == pytest.approx(
        finite_difference,
        rel=5.0e-3,
        abs=1.0e-5,
    )


def test_mma_backend_is_exact_registered_nlopt_version_or_unavailable() -> None:
    qualification = qualify_mma_backend()

    if qualification.available:
        assert qualification.backend == "nlopt"
        assert qualification.algorithm == "LD_MMA"
        assert qualification.version.startswith("2.10.")
    else:
        assert qualification.reason == "MMA_BACKEND_UNAVAILABLE"


@pytest.mark.parametrize("evaluations", [0, 24, 26, 599, 601])
def test_mma_rejects_unregistered_evaluation_budgets(evaluations: int) -> None:
    task = sample_primary_task(2026092317, 1)
    with pytest.raises(ValueError, match="25/50/100/200/600"):
        optimize_mma(task, evaluations=evaluations, seed=2026092320)


def test_mma_trajectory_rejects_unregistered_or_out_of_range_snapshots() -> None:
    task = sample_primary_task(2026092317, 2)

    with pytest.raises(ValueError, match="snapshot"):
        optimize_mma(
            task,
            evaluations=600,
            seed=2026092321,
            snapshot_evaluations=(25, 75, 600),
        )
    with pytest.raises(ValueError, match="snapshot"):
        optimize_mma(
            task,
            evaluations=200,
            seed=2026092321,
            snapshot_evaluations=(25, 600),
        )
