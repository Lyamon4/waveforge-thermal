"""Learned replacement of refinement must keep head identity and split hygiene."""

import numpy as np
import torch


def row(peak, *, valid=True):
    return dict(valid=valid, violation=0.0 if valid else 1.0, peak_c=float(peak))


def test_training_targets_keep_head_identity_and_reject_invalid_improvements():
    from waveforge.lab.mcl.amortized import select_head_targets

    initial = np.zeros((4, 4125))
    initial[:, 20:25] = 1
    baseline = [row(70 + h) for h in range(4)]
    wrong = initial[0].copy()
    wrong[0] = 0.4
    better = initial[2].copy()
    better[0] = 0.5
    trajectories = [
        dict(head=0, parameters=wrong, raw=row(72)),
        dict(head=1, parameters=wrong, raw=row(50, valid=False)),
        dict(head=2, parameters=better, raw=row(68)),
    ]
    targets, weights = select_head_targets(initial, baseline, trajectories)
    np.testing.assert_array_equal(targets[[0, 1, 3]], initial[[0, 1, 3]])
    np.testing.assert_array_equal(targets[2], better)
    assert weights[2] > weights[[0, 1, 3]].max()
    np.testing.assert_array_equal(initial[:, 0], 0)


def test_distillation_loss_is_head_aligned_finite_at_physical_bounds():
    from waveforge.lab.mcl.amortized import distillation_loss
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder

    raw = torch.zeros((2, 4, 4125), dtype=torch.float32, requires_grad=True)
    targets = RibbedDecoder.to_physical(raw.detach().flatten(0, 1)).reshape_as(raw)
    weights = torch.ones((2, 4))
    assert distillation_loss(raw, targets, weights).item() == 0
    targets[:, 1, :20] = 5.5
    targets[:, 1, 20:25] = 2.2
    targets[:, 1, 25:29] = -2
    targets[:, 1, 29:] = 1
    loss = distillation_loss(raw, targets, weights)
    gradient = torch.autograd.grad(loss, raw)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient[:, 1].abs().sum() > 0
    assert gradient[:, [0, 2, 3]].abs().max() == 0


def test_fresh_validation_geometry_and_its_mirror_never_enter_training():
    from waveforge.lab.epyc_tasks import GeometryIndex, sample_epyc_task
    from waveforge.lab.mcl.adaptive_manifold import reflected_task
    from waveforge.lab.mcl.amortized import fresh_task_groups

    protected = tuple(sample_epyc_task(np.random.default_rng(s)) for s in (21, 22))
    first = fresh_task_groups(2026103001, protected, train_count=8, validation_count=4)
    again = fresh_task_groups(2026103001, protected, train_count=8, validation_count=4)
    for split in ("train", "validation"):
        assert [t.fingerprint() for t in first[split]] == [
            t.fingerprint() for t in again[split]
        ]
    forbidden = (*protected, *first["validation"])
    index = GeometryIndex(forbidden + tuple(reflected_task(t) for t in forbidden))
    assert all(not index.contains(t) for t in first["train"])
    assert len({t.fingerprint() for t in first["train"]}) == 8


def test_validation_gap_uses_overheat_not_celsius_temperature():
    from waveforge.lab.mcl.amortized_training import validation_summary

    actual = [
        dict(selected=row(65), fallback_used=False),
        dict(selected=row(70), fallback_used=True),
    ]
    teachers = [row(65), row(75)]
    original = [row(70), row(80)]
    summary = validation_summary(actual, teachers, original)
    np.testing.assert_allclose(summary["gaps_to_teacher_percent"], [0, -10])
    assert summary["median_gap_to_teacher_percent"] == -5
    assert summary["fallback_count"] == 1
    assert summary["median_gain_vs_original_r0_percent"] > 11


def test_residual_policy_starts_at_source_and_never_updates_source_weights():
    from waveforge.lab.mcl.amortized import distillation_loss
    from waveforge.lab.mcl.amortized_policy import ResidualRibbedUNet
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder

    torch.set_num_threads(1)
    torch.manual_seed(54)
    model = ResidualRibbedUNet()
    inputs = torch.randn(1, 13, 64, 64)
    model.train().requires_grad_(True)
    actual = model(inputs)
    with torch.no_grad():
        expected = RibbedDecoder.to_physical(
            model.source(inputs).flatten(0, 1).double()
        )
    torch.testing.assert_close(actual, expected.reshape_as(actual), rtol=0, atol=0)
    target = actual.detach().clone()
    target[:, :, :20] += 0.1
    saved = {k: v.clone() for k, v in model.source.state_dict().items()}
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )
    loss = distillation_loss(
        actual, target, torch.ones(1, 4), prediction_space="physical"
    )
    loss.backward()
    assert all(p.grad is None for p in model.source.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.delta_geometry.parameters()
    )
    optimizer.step()
    for key, value in model.source.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)


def test_polish_labels_use_the_existing_joint_physics_objective():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.amortized_polish import physics_labels
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.joint_design import JointObjective
    from waveforge.lab.mcl.stage5_physics import Stage5Solver, stage5_spec

    torch.set_num_threads(1)
    task = sample_epyc_task(np.random.default_rng(91))
    d = uniform_parallel().astype(float)
    values, gradients, residual = physics_labels(task, d[None])
    x = torch.tensor(d, requires_grad=True)
    objective = JointObjective(Stage5Solver([problem_from_task(task, stage5_spec())]))
    expected, _ = objective(x, 2)
    derivative = torch.autograd.grad(expected, x)[0]
    np.testing.assert_allclose(values, [expected.item()], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradients[0], derivative.numpy(), rtol=1e-12, atol=1e-12)
    assert residual < 1e-12


def test_topology_proxy_keeps_budget_and_teaches_material_boundaries():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.amortized import topology_distillation_loss, topology_proxy
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    torch.set_num_threads(1)
    geometry = RibbedGeometry.from_task(sample_epyc_task(np.random.default_rng(34)))
    decoder = RibbedDecoder(geometry)
    raw = torch.tensor(
        np.stack([decoder.initial(h) for h in range(4)])[None], requires_grad=True
    )
    target_raw = raw.detach().clone()
    target_raw[:, :, 20:25] += 0.3
    target_q = RibbedDecoder.to_physical(target_raw.flatten(0, 1)).reshape_as(raw)
    target_binary = decoder.binary(target_raw.flatten(0, 1), range(4)).reshape(
        1, 4, 64, 64
    )
    proxy = topology_proxy(raw, [geometry])
    torch.testing.assert_close(
        proxy.sum((-2, -1)),
        torch.full((1, 4), 1024.0, dtype=torch.float64),
        rtol=0,
        atol=1e-8,
    )
    loss = topology_distillation_loss(
        raw, target_q, torch.ones(1, 4), [geometry], target_binary
    )
    gradient = torch.autograd.grad(loss, raw)[0]
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient[..., 20:25].abs().sum() > 0


def test_frozen_portfolio_retains_source_and_correct_decoder_head_ids(tmp_path):
    from waveforge.lab.mcl.amortized_policy import CandidatePortfolio, load_policy
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder

    torch.set_num_threads(1)
    model = CandidatePortfolio((0, 1, 4, 5)).eval()
    with torch.no_grad():
        model.student.geometry.bias[:20] = 0.1
    x = torch.randn(1, 13, 64, 64)
    with torch.inference_mode():
        a = RibbedDecoder.to_physical(model.source(x).flatten(0, 1).double()).reshape(
            1, 4, 4125
        )
        b = RibbedDecoder.to_physical(
            model.student(x).flatten(0, 1).double()
        ).reshape_as(a)
        expected = torch.cat((a[:, :2], b[:, :2]), 1)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert model.proposal_heads == (0, 1, 0, 1)
    checkpoint = tmp_path / "portfolio.pt"
    torch.save(
        dict(
            model=model.state_dict(),
            architecture="CandidatePortfolio",
            proposal_indices=[0, 1, 4, 5],
        ),
        checkpoint,
    )
    restored = load_policy(checkpoint)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    assert not any(p.requires_grad for p in restored.parameters())


def test_eight_reference_starts_are_fixed_and_preserve_the_original_four():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry
    from waveforge.lab.mcl.ribbed_evaluation import analytic_reference_starts

    torch.set_num_threads(1)
    four, eight = analytic_reference_starts(4), analytic_reference_starts(8)
    assert len(four) == 4 and len(eight) == 8
    for (expected, h), (actual, other) in zip(four, eight[:4], strict=True):
        assert h == other
        np.testing.assert_array_equal(actual, expected)
    geometry = RibbedGeometry.from_task(sample_epyc_task(np.random.default_rng(331)))
    decoder = RibbedDecoder(geometry)
    masks = [
        decoder.binary(torch.tensor(q[None]), [h], physical=True)[0].numpy()
        for q, h in eight
    ]
    assert len({m.tobytes() for m in masks}) >= 6
