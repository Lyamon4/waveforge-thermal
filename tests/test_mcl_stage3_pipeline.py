import importlib.util

import numpy as np
import pytest
import torch

from waveforge.lab.epyc_tasks import sample_epyc_task
from waveforge.lab.mcl.contracts import problem_from_task, stage12_spec


def test_training_module_exists():
    assert importlib.util.find_spec("waveforge.lab.mcl.stage3_training") is not None


def test_seeded_stream_excludes_protected_geometry():
    from waveforge.lab.mcl.stage3_training import TrainingStream

    seed = 37
    protected = sample_epyc_task(np.random.default_rng(seed))
    first = TrainingStream(seed, (protected,))
    second = TrainingStream(seed, (protected,))
    tasks = [first.next() for _ in range(5)]
    assert first.rejected >= 1
    assert len({t.fingerprint() for t in tasks}) == 5
    assert [t.fingerprint() for t in tasks] == [
        second.next().fingerprint() for _ in tasks
    ]
    assert not any(first.protected.contains(t) for t in tasks)


def test_conditioning_order_units_and_finite_sensitivities():
    from waveforge.lab.mcl.stage3_training import condition_worker

    task = sample_epyc_task(np.random.default_rng(121))
    problem = problem_from_task(task, stage12_spec())
    channels, report = condition_worker(task)
    assert channels.shape == (13, 64, 64)
    np.testing.assert_allclose(channels[:3], problem.power_w, rtol=1e-6)
    np.testing.assert_array_equal(channels[9], problem.chip_mask)
    assert channels[10].sum() == 32
    assert np.all(channels[10][0, 24:40] == 1)
    np.testing.assert_allclose(channels[11, :, 0], np.arange(64) / 63, atol=1e-7)
    np.testing.assert_allclose(channels[12, 0, :], np.arange(64) / 63, atol=1e-7)
    assert np.isfinite(channels).all()
    assert np.max(np.abs(channels[6:9])) <= 1
    assert np.linalg.norm(channels[6:9]) > 0
    assert report["max_linear_residual"] < 1e-12


def test_worker_physics_derivative_matches_finite_difference():
    from waveforge.lab.mcl.stage3_training import physics_worker

    task = sample_epyc_task(np.random.default_rng(122))
    rng = np.random.default_rng(123)
    # Avoid the TV penalty's nonsmooth equal-neighbour points in symmetric anchors.
    density = rng.uniform(0.1, 0.4, (1, 64, 64))
    losses, grads, report = physics_worker(task, density, 0)
    direction = rng.normal(size=density.shape)
    epsilon = 1e-5
    plus = physics_worker(task, density + epsilon * direction, 0)[0]
    minus = physics_worker(task, density - epsilon * direction, 0)[0]
    np.testing.assert_allclose(
        (plus - minus) / (2 * epsilon),
        (grads * direction).sum((1, 2)),
        rtol=2e-4,
        atol=1e-7,
    )
    assert losses.shape == (1,)
    assert report["max_linear_residual"] < 1e-12


def test_gradient_bridge_softmin_keeps_all_heads_differentiable():
    from waveforge.lab.mcl.stage3_training import bridged_losses

    density = torch.full((2, 4, 64, 64), 0.25, dtype=torch.float64, requires_grad=True)
    losses = np.tile(np.array([1.0, 1.1, 1.2, 1.3]), (2, 1))
    gradients = np.ones(density.shape)
    linked = bridged_losses(density, losses, gradients)
    torch.testing.assert_close(linked, torch.tensor(losses))
    (
        -0.05 * torch.logsumexp(-linked / 0.05, 1) + 0.1 * linked.mean(1)
    ).mean().backward()
    assert (density.grad > 0).all()
    assert density.grad[0, 0].mean() > density.grad[0, 3].mean()


def test_selection_rejects_invalid_even_if_colder_and_breaks_ties():
    from waveforge.lab.mcl.stage3_evaluation import select_valid

    rows = [
        dict(accepted=False, peak_rise_k=1, step=0, head=0),
        dict(accepted=True, peak_rise_k=10, step=5, head=1),
        dict(accepted=True, peak_rise_k=10, step=0, head=2),
    ]
    assert select_valid(rows) is rows[2]
    assert select_valid(rows[:1]) is None


def test_velocity_export_preserves_flux_units_and_refuses_overwrite(tmp_path):
    from types import SimpleNamespace

    from waveforge.lab.mcl.stage3_evaluation import save_fields

    spec = stage12_spec()
    result = SimpleNamespace(
        temperature=np.zeros((3, 4, 64, 64)),
        pressure=np.zeros((64, 64)),
        qx=np.ones((63, 64)) * spec.dy * spec.dz[3],
        qy=np.ones((64, 63)) * 2 * spec.dx * spec.dz[3],
        qin=np.zeros(16),
        qout=np.zeros(16),
    )
    path = tmp_path / "fields.npz"
    save_fields(path, np.zeros(4116), np.zeros((64, 64)), result, spec)
    with np.load(path) as data:
        assert data["velocity_x_m_s"].shape == (63, 64)
        np.testing.assert_allclose(data["velocity_x_m_s"], 1)
        np.testing.assert_allclose(data["velocity_y_m_s"], 2)
        np.testing.assert_array_equal(data["qx"], result.qx)
        assert str(data["temperature_units"]) == "K above inlet"
    with pytest.raises(FileExistsError):
        save_fields(path, np.zeros(4116), np.zeros((64, 64)), result, spec)


def test_exclusions_read_only_sealed_geometry_not_closed_workloads(tmp_path):
    import json

    from waveforge.lab.contracts import content_hash
    from waveforge.lab.mcl.stage3_training import protected_geometries

    task = sample_epyc_task(np.random.default_rng(124))
    geometry = {
        "schema": "epyc-v2-geometry-index-v1",
        "retrospective_v1": [],
        "v2": {"test_id": [task.canonical()["chiplets"]], "test_ood": []},
    }
    (tmp_path / "geometry_exclusions.json").write_text(json.dumps(geometry))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "counts": {"test_id": 1, "test_ood": 0},
                "geometry_sha256": content_hash(geometry),
            }
        )
    )
    # Neither closed workload file exists: this must still work.
    protected, report = protected_geometries(tmp_path)
    assert protected[0].geometry_fingerprint() == task.geometry_fingerprint()
    assert report["closed_workloads_read"] is False
    assert report["protected_count"] == 1
    geometry["v2"]["test_id"] = []
    (tmp_path / "geometry_exclusions.json").write_text(json.dumps(geometry))
    with pytest.raises(ValueError, match="seal"):
        protected_geometries(tmp_path)


def test_no_admissible_r0_never_launches_refinement(monkeypatch, tmp_path):
    import json

    from waveforge.lab.mcl import stage3_evaluation as evaluation

    # Isolate orchestration from costly physical solves using explicit rejected reports.
    def rejected(problem, parameters, output, step, *, head=None):
        return dict(accepted=False, peak_rise_k=1.0, step=step, head=head)

    def only_control(task, initial, output, *, steps, checkpoints, method):
        assert method == "Vascular-Adam600-Halo"
        return dict(method=method, selected=None)

    monkeypatch.setattr(evaluation, "record_candidate", rejected)
    monkeypatch.setattr(evaluation, "optimize_parameters", only_control)
    task = sample_epyc_task(np.random.default_rng(125))
    folder = tmp_path / "evaluation"
    reports = evaluation.evaluate_task(task, np.zeros((4, 4116)), folder)
    assert reports[0]["selected"] is None
    assert len(reports[0]["checkpoints"]) == 4
    assert reports[1]["steps"] == 0 and reports[1]["selected"] is None
    saved = json.loads((folder / "Vascular-UNet-R25/result.json").read_text())
    assert saved["reason"] == "No admissible R0 head; no fallback or topology repair"
    assert not (folder / "Vascular-UNet-R25/selected.npz").exists()
