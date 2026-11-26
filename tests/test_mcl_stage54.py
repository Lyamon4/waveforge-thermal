"""Locked channel crossings, real reflection symmetry and joint-mode admission."""

import numpy as np
import pytest
import torch
from scipy.ndimage import label


def task(seed=54):
    from waveforge.lab.epyc_tasks import sample_epyc_task

    return sample_epyc_task(np.random.default_rng(seed))


def test_locked_centres_come_from_mask_without_task_labels():
    from waveforge.lab.mcl.locked_channels import LockedGeometry

    t = task()
    mask = np.zeros((64, 64), bool)
    for c in t.chiplets:
        mask[c.x : c.x + c.width, c.y : c.y + c.height] = True
    g = LockedGeometry.from_mask(mask)
    expected = sorted(
        (c.x + (c.width - 1) / 2, c.y + (c.height - 1) / 2)
        for c in t.chiplets
        if c.width == 8
    )
    assert sorted(map(tuple, np.concatenate((g.bottom, g.top)))) == expected


def test_locked_binary_routes_cross_every_assigned_ccd_even_at_extreme_residuals():
    from waveforge.lab.mcl.locked_channels import LockedDecoder, LockedGeometry

    for seed in (54, 55, 56):
        t = task(seed)
        g = LockedGeometry.from_task(t)
        decoder = LockedDecoder(g)
        mask = np.zeros((64, 64), bool)
        for c in t.chiplets:
            mask[c.x : c.x + c.width, c.y : c.y + c.height] = True
        for h in range(4):
            p = torch.tensor(decoder.initial(h))
            p[:20] = torch.tensor(
                np.random.default_rng(seed + h).choice([-100.0, 100.0], 20)
            )
            p[25:] = 100
            xy = decoder.coordinates(p[None], [h])
            d = decoder.binary(p[None], [h])[0].numpy()
            assert d.sum() == 1024 and label(d)[1] == 1
            core = decoder.core(xy)[0].numpy()
            assert core.sum() <= 832
            rows = decoder.row_kinds[h]
            for b, kind in enumerate(rows):
                if kind == "io":
                    continue
                for k in range(1, 5):
                    x = int(xy[0, b, k, 0])
                    y = int(torch.round(xy[0, b, k, 1]))
                    assert mask[x, y] and core[x, y] and d[x, y]
            if h > 0:
                assert (xy[0, 2, 1:-1, 1] < 24).all()


def test_canonical_wrapper_is_exact_on_true_mirrors_not_on_load_swapped_pairs():
    from waveforge.lab.mcl.adaptive_manifold import contrast_pair, reflected_task
    from waveforge.lab.mcl.locked_channels import (
        LockedDecoder,
        LockedGeometry,
        canonical_frame,
    )

    for seed in (54, 55, 56, 57):
        t = task(seed)
        a = canonical_frame(t)
        b = canonical_frame(reflected_task(t))
        assert a.task.fingerprint() == b.task.fingerprint()
        decoder = LockedDecoder(LockedGeometry.from_task(a.task))
        d = decoder.binary(torch.tensor(decoder.initial(1)[None]), [1])[0].numpy()
        assert np.array_equal(a.to_original(d), np.flip(b.to_original(d), axis=-1))
        lower, upper = contrast_pair(t, np.random.default_rng(seed))
        assert upper.fingerprint() != reflected_task(lower).fingerprint()


def test_locked_continuous_projection_and_fixed_core_gradient():
    from waveforge.lab.mcl.locked_channels import LockedDecoder, LockedGeometry

    decoder = LockedDecoder(LockedGeometry.from_task(task()))
    p = torch.tensor(decoder.initial(0)[None], requires_grad=True)
    d = decoder.density(p, [0], beta=4, hard=False)
    assert d.sum().item() == pytest.approx(1024, abs=1e-8)
    weights = torch.tensor(np.random.default_rng(54).normal(size=(1, 64, 64)))
    f = (d * weights).sum()
    gradient = torch.autograd.grad(f, p)[0]
    v = torch.zeros_like(p)
    v[0, 20] = 1
    eps = 1e-6
    fd = (
        (decoder.density(p.detach() + eps * v, [0], beta=4, hard=False) * weights).sum()
        - (
            decoder.density(p.detach() - eps * v, [0], beta=4, hard=False) * weights
        ).sum()
    ) / (2 * eps)
    assert gradient[0, 20].item() == pytest.approx(fd.item(), rel=1e-6, abs=1e-7)


def row(re=1000, si=70, water=40):
    return dict(
        accepted=True,
        silicon_peak_c=si,
        peak_rise_k=si - 25,
        diagnostics=dict(
            max_local_reynolds_estimate=re,
            peak_open_water_temperature_c=water,
            pressure_residual=1e-14,
            compensated_pressure_residual=1e-14,
            thermal_residual=[1e-14] * 3,
            compensated_thermal_residual=[1e-14] * 3,
            mass_relative_error=1e-13,
            energy_relative_error=[1e-13] * 3,
        ),
    )


def test_joint_gate_rejects_either_regime_and_does_not_weaken_old_numeric_limits():
    from waveforge.lab.mcl.joint_design import joint_admission

    modes = {str(g): {m: row() for m in ("nominal", "power")} for g in (64, 128)}
    assert joint_admission(modes)["admissible"]
    modes["128"]["power"] = row(re=1200)
    assert not joint_admission(modes)["admissible"]
    modes["128"]["power"] = row(si=95.00001)
    assert not joint_admission(modes)["admissible"]
    modes["128"]["power"] = row(si=95, water=59.999)
    assert joint_admission(modes)["admissible"]
    modes["128"]["power"]["diagnostics"]["thermal_residual"][0] = 1e-12
    assert not joint_admission(modes)["admissible"]


def test_joint_flow_probe_and_penalty_have_correct_fixed_pump_gradient():
    from waveforge.lab.hydraulic_probe import (
        HydraulicProbe,
        refined_spec,
        replicated_design,
    )
    from waveforge.lab.mcl.joint_design import joint_flow_penalty, joint_reynolds
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    s = refined_spec(stage5_spec(nx=8, ny=8))
    d = torch.full((1, 8, 8), 0.999, dtype=torch.float64, requires_grad=True)
    probe = HydraulicProbe(s)
    flow = probe(replicated_design(d))
    re = joint_reynolds(flow, s)
    assert torch.equal(re["joint"], torch.maximum(re["nominal"], re["power"]))
    f = joint_flow_penalty(flow, s)
    grad = torch.autograd.grad(f, d)[0]
    v = torch.tensor(np.random.default_rng(54).normal(size=d.shape))
    v /= v.norm()
    eps = 1e-6

    def loss(x):
        return joint_flow_penalty(probe(replicated_design(x)), s)

    fd = (loss(d.detach() + eps * v) - loss(d.detach() - eps * v)) / (2 * eps)
    assert (grad * v).sum().item() == pytest.approx(fd.item(), rel=1e-6, abs=1e-6)


def test_locked_model_starts_from_all_four_shared_classical_seeds():
    from waveforge.lab.mcl.joint_design import LockedUNet, head_loss
    from waveforge.lab.mcl.locked_channels import LockedDecoder, LockedGeometry

    model = LockedUNet().eval()
    with torch.no_grad():
        p = model(torch.ones(1, 13, 64, 64))[0].double()
    decoder = LockedDecoder(LockedGeometry.from_task(task()))
    for h in range(4):
        torch.testing.assert_close(
            p[h], torch.tensor(decoder.initial(h)), rtol=0, atol=1e-6
        )
    values = torch.tensor([[0.1, 0.3, 0.5, 0.7]], requires_grad=True)
    gradient = torch.autograd.grad(head_loss(values), values)[0]
    assert gradient.min() >= 0.125


def test_true_y_mirrors_have_the_same_independent_temperature():
    from waveforge.experiments.audit_mcl_stage5_admission import record
    from waveforge.lab.mcl.adaptive_manifold import reflected_task
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.locked_channels import (
        LockedDecoder,
        LockedGeometry,
        canonical_frame,
    )
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    for seed in (54, 55, 56, 57):
        original = task(seed)
        mirror = reflected_task(original)
        frame = canonical_frame(original)
        decoder = LockedDecoder(LockedGeometry.from_task(frame.task))
        d = decoder.binary(torch.tensor(decoder.initial(1)[None]), [1])[0].numpy()
        a = frame.to_original(d)
        b = canonical_frame(mirror).to_original(d)
        assert np.array_equal(a, b[:, ::-1])
        r1, _ = record(a, problem_from_task(original, stage5_spec()))
        r2, _ = record(b, problem_from_task(mirror, stage5_spec()))
        assert abs(r1["silicon_peak_c"] - r2["silicon_peak_c"]) < 1e-9


def test_full_size_joint_objective_is_finite_and_keeps_strict_solver_tolerance():
    from waveforge.lab.mcl.locked_channels import (
        LockedDecoder,
        LockedGeometry,
        canonical_frame,
    )
    from waveforge.lab.mcl.locked_training import physics_worker

    t = canonical_frame(task()).task
    decoder = LockedDecoder(LockedGeometry.from_task(t))
    d = decoder.density(torch.tensor(decoder.initial(0)[None]), [0], beta=6, hard=True)
    value, gradient, residual = physics_worker(t, d.numpy(), 2)
    assert np.isfinite(value).all() and np.isfinite(gradient).all()
    assert residual < 1e-12 and np.linalg.norm(gradient) > 0


def test_reused_design_gradient_preserves_the_full_parameter_vjp():
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.design_gradient_cache import DesignGradientCache
    from waveforge.lab.mcl.joint_design import JointObjective
    from waveforge.lab.mcl.locked_channels import LockedDecoder, LockedGeometry
    from waveforge.lab.mcl.stage5_physics import Stage5Solver, stage5_spec

    t = task()
    decoder = LockedDecoder(LockedGeometry.from_task(t))
    objective = JointObjective(Stage5Solver([problem_from_task(t, stage5_spec())]))
    memo = DesignGradientCache(objective)
    initial = decoder.initial(1)
    initial[:25] += np.random.default_rng(541).normal(0, 0.1, 25)
    maps = []
    for delta in (0.0, 1e-8):
        p = torch.tensor(initial.copy(), requires_grad=True)
        with torch.no_grad():
            p[20] += delta
        d = decoder.density(p[None], [1], beta=6, hard=True)[0]
        maps.append(d.detach().numpy())
        full, _ = objective(d, 2)
        expected = torch.autograd.grad(full, p)[0]
        q = p.detach().clone().requires_grad_(True)
        used = decoder.density(q[None], [1], beta=6, hard=True)[0]
        value, g, _ = memo.evaluate(used, 2, cache=True)
        actual = torch.autograd.grad(used, q, g)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        assert value == pytest.approx(full.item(), abs=1e-13)
    assert np.array_equal(*maps)
    assert memo.hits == 1 and memo.misses == 1
