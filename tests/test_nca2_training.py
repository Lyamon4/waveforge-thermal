from __future__ import annotations

from pathlib import Path

import pytest
import torch

from waveforge.experiments.run_inverse_design import gate2_source_batch
from waveforge.ml.nca import project_nca_material
from waveforge.ml.nca2_schedule import ObjectiveSettings
from waveforge.ml.nca2_training import ScheduledNCAController
from waveforge.ml.nca_training import evaluate_nca, initialize_nca, run_nca_training


def test_projection_beta_changes_relaxed_design_but_preserves_volume() -> None:
    logits = torch.linspace(-1.0, 1.0, 4096, dtype=torch.float32).reshape(1, 1, 64, 64)

    soft = project_nca_material(logits, beta=2.0).design
    sharp = project_nca_material(logits, beta=8.0).design

    assert soft.mean().item() == pytest.approx(0.25, abs=1.0e-6)
    assert sharp.mean().item() == pytest.approx(0.25, abs=1.0e-6)
    assert not torch.equal(soft, sharp)


def test_old_evaluate_nca_defaults_are_unchanged() -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))
    model = initialize_nca(20260830, torch.device("cpu"))

    result = evaluate_nca(model, sources, allow_cpu_unit_test=True)

    assert result.objective.total.item() == pytest.approx(0.7078165659453859)
    assert result.continuous_design.mean().item() == pytest.approx(0.25, abs=1.0e-6)


def test_scheduled_controller_applies_objective_and_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}
    controller = ScheduledNCAController("B")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    sentinel = object()

    def fake_evaluate(*args, **kwargs):
        captured.update(
            {
                "projection_beta": kwargs["projection_beta"],
                "smooth_max_alpha": kwargs["smooth_max_alpha"],
                "tv_weight": kwargs["tv_weight"],
                "binarization_weight": kwargs["binarization_weight"],
            }
        )
        return sentinel

    monkeypatch.setattr("waveforge.ml.nca2_training.evaluate_nca", fake_evaluate)
    controller.configure(500, optimizer)

    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert controller.settings == ObjectiveSettings(8.0, 500.0, 0.001, 0.02)
    assert controller.evaluate(object(), object()) is sentinel
    assert captured == {
        "projection_beta": 8.0,
        "smooth_max_alpha": 500.0,
        "tv_weight": 0.001,
        "binarization_weight": 0.02,
    }


def test_iteration_configurator_runs_before_each_forward_and_checkpoint(
    tmp_path: Path,
) -> None:
    sources = gate2_source_batch(device=torch.device("cpu"))
    observed: list[tuple[int, float]] = []
    current_iteration = -1
    current_lr = 0.0

    def configure(iteration: int, optimizer: torch.optim.Optimizer) -> None:
        nonlocal current_iteration, current_lr
        current_iteration = iteration
        current_lr = (1.0e-3, 3.0e-4)[iteration]
        optimizer.param_groups[0]["lr"] = current_lr

    def observe_then_evaluate(*args, **kwargs):
        observed.append((current_iteration, current_lr))
        return evaluate_nca(*args, **kwargs)

    result = run_nca_training(
        sources,
        seed=7,
        learning_rate=1.0e-3,
        iterations=2,
        mode="unit",
        output_dir=tmp_path,
        evaluator=observe_then_evaluate,
        allow_cpu_unit_test=True,
        checkpoint_interval=1,
        iteration_configurator=configure,
    )

    assert result.completed_iterations == 2
    assert observed == [(0, 1.0e-3), (1, 3.0e-4), (1, 3.0e-4)]
    checkpoint = torch.load(
        tmp_path / "checkpoint_000002.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["initial_learning_rate"] == 1.0e-3
    assert checkpoint["learning_rate"] == 3.0e-4
