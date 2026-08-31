from __future__ import annotations

import torch

from waveforge.ml.mt2b_tasks import balanced_task_batch, classify_geometry
from waveforge.ml.mt2b_training import (
    build_mt2b_evaluator,
    initialize_mt2b_model,
    mt2b_task_provider,
)
from waveforge.ml.multitask_protocol import settings_at
from waveforge.ml.nca_training import model_state_sha256


def test_mt2b_task_provider_uses_exact_locked_balanced_batch() -> None:
    expected = balanced_task_batch(
        17,
        seed=2026092201,
        excluded_task_ids=frozenset(),
    )
    actual = tuple(mt2b_task_provider(2026092201, 17, index) for index in range(4))

    assert [task.task_id for task in actual] == [task.task_id for task in expected]
    assert [classify_geometry(task.centers) for task in actual] == [
        "compact",
        "wide_horizontal",
        "vertically_spread",
        "mixed",
    ]


def test_mt2b_model_initialization_is_identical_for_both_variants() -> None:
    first = initialize_mt2b_model(2026092202, torch.device("cpu"))
    second = initialize_mt2b_model(2026092202, torch.device("cpu"))

    assert model_state_sha256(first) == model_state_sha256(second)
    assert sum(parameter.numel() for parameter in first.parameters()) == 12624


def test_raw_mt2b_evaluator_runs_safe_scenario_batched_backward_on_cpu() -> None:
    model = initialize_mt2b_model(2026092202, torch.device("cpu"))
    task = mt2b_task_provider(2026092201, 0, 0)
    sources = torch.as_tensor(task.sources, dtype=torch.float64)
    evaluator = build_mt2b_evaluator("RAW")

    forward = evaluator(
        model,
        sources,
        settings_at(0, 2000),
        allow_cpu_unit_test=True,
    )
    forward.loss.backward()

    assert torch.isfinite(forward.loss)
    assert forward.loss.requires_grad
    assert forward.solve_trace is not None
    assert [record.role for record in forward.solve_trace.records] == [
        "forward",
        "forward",
        "forward",
        "adjoint",
        "adjoint",
        "adjoint",
    ]
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_physics_mt2b_evaluator_uses_same_objective_contract_on_cpu() -> None:
    model = initialize_mt2b_model(2026092202, torch.device("cpu"))
    task = mt2b_task_provider(2026092201, 0, 0)
    sources = torch.as_tensor(task.sources, dtype=torch.float64)
    evaluator = build_mt2b_evaluator("PHYSICS")

    forward = evaluator(
        model,
        sources,
        settings_at(0, 2000),
        allow_cpu_unit_test=True,
    )

    assert torch.isfinite(forward.loss)
    assert forward.continuous_material_fraction == 0.25
    assert forward.projection_absolute_error <= 1.0e-6
