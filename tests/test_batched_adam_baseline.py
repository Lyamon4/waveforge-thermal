from __future__ import annotations

import numpy as np
import pytest
import torch

from waveforge.design.batched_adam_baseline import (
    optimize_adam_batched,
    taskwise_clip_gradients_,
)
from waveforge.ml.multitask_tasks import sample_primary_task


def test_taskwise_clipping_does_not_couple_independent_layouts() -> None:
    gradients = torch.tensor(
        [
            [[3.0, 4.0], [0.0, 0.0]],
            [[0.3, 0.4], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )

    norms = taskwise_clip_gradients_(gradients, max_norm=1.0)

    assert norms.tolist() == pytest.approx([5.0, 0.5])
    assert torch.linalg.vector_norm(gradients[0]).item() == pytest.approx(1.0)
    assert torch.linalg.vector_norm(gradients[1]).item() == pytest.approx(0.5)


def test_batched_adam_matches_two_independent_runs_on_real_cpu_physics() -> None:
    tasks = (
        sample_primary_task(2026092601, 0),
        sample_primary_task(2026092601, 1),
    )
    seeds = (2026092602, 2026092603)

    together = optimize_adam_batched(
        tasks,
        seeds=seeds,
        total_updates=2,
        snapshot_updates=(1, 2),
        device=torch.device("cpu"),
        allow_cpu_unit_test=True,
    )
    separate = tuple(
        optimize_adam_batched(
            (task,),
            seeds=(seed,),
            total_updates=2,
            snapshot_updates=(1, 2),
            device=torch.device("cpu"),
            allow_cpu_unit_test=True,
        ).tasks[0]
        for task, seed in zip(tasks, seeds, strict=True)
    )

    assert together.completed_updates == 2
    assert together.snapshot_updates == (1, 2)
    for batched, single in zip(together.tasks, separate, strict=True):
        np.testing.assert_allclose(
            batched.final_logits,
            single.final_logits,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        for update in (1, 2):
            np.testing.assert_array_equal(
                batched.snapshots[update].binary_design,
                single.snapshots[update].binary_design,
            )
            assert batched.snapshots[update].binary_cell_count == 1024
            assert batched.snapshots[update].binary_material_fraction == 0.25


def test_production_batched_adam_rejects_unregistered_budget() -> None:
    task = sample_primary_task(2026092604, 0)

    with pytest.raises(ValueError, match="600 updates"):
        optimize_adam_batched(
            (task,),
            seeds=(2026092605,),
            total_updates=599,
            snapshot_updates=(25, 50, 100, 200, 599),
            device=torch.device("cpu"),
            allow_cpu_unit_test=False,
        )
