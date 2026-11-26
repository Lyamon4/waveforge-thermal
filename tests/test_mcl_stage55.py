"""Separated channels, physical-coordinate refinement and exact shared budgets."""

import numpy as np
import pytest
import torch
from scipy.ndimage import label


def example(seed=55):
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.locked_channels import canonical_frame

    return canonical_frame(sample_epyc_task(np.random.default_rng(seed))).task


def test_five_channels_stay_separate_with_two_cell_copper_ribs_at_extreme_controls():
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    for seed in (55, 56):
        decoder = RibbedDecoder(RibbedGeometry.from_task(example(seed)))
        for h in range(4):
            p = decoder.initial(h)
            p[:20] = np.random.default_rng(seed + h).choice([-6.0, 6.0], 20)
            p[20:25] = 6
            p[25:29] = np.random.default_rng(seed).choice([-6.0, 6.0], 4)
            p[29:] = 6
            x = torch.tensor(p[None])
            d = decoder.binary(x, [h])[0].numpy()
            core, wall, rows = decoder.masks(decoder.coordinates(x, [h]))
            assert d.sum() == 1024 and label(d)[1] == 1
            assert label(d[4:60, 7:57])[1] == 5
            assert not np.any(d & wall[0].numpy())
            assert core[0].sum() <= 955
            assert np.all(rows[0, :, 1:].min(1) >= 8)
            for column in range(56):
                intervals = []
                for b in range(5):
                    previous = rows[0, b, max(column - 1, 0)]
                    current = rows[0, b, column]
                    intervals.append((min(previous, current), max(previous, current)))
                assert all(intervals[b + 1][0] - intervals[b][1] >= 3 for b in range(4))


def test_new_heads_have_distinct_routes_and_movable_io_x_nodes():
    from waveforge.lab.mcl.ribbed_channels import (
        RIB_ROWS,
        RibbedDecoder,
        RibbedGeometry,
    )

    g = RibbedGeometry.from_task(example())
    decoder = RibbedDecoder(g)
    assert RIB_ROWS[1] == ("bottom", "bottom", "bottom", "io", "top")
    assert RIB_ROWS[2] == ("bottom", "bottom", "io", "top", "top")
    a = decoder.initial(2)
    b = a.copy()
    b[25:29] = 1.0
    ax = decoder.knots(torch.tensor(a[None]), [2])
    bx = decoder.knots(torch.tensor(b[None]), [2])
    assert not torch.equal(ax[0, 2, :, 0], bx[0, 2, :, 0])
    assert torch.equal(ax[0, 0, :, 0], bx[0, 0, :, 0])
    maps = decoder.binary(
        torch.tensor(np.stack([decoder.initial(h) for h in range(4)])), range(4)
    ).numpy()
    assert len({d.tobytes() for d in maps}) >= 3


def test_masked_density_preserves_budget_ribs_and_fixed_mask_gradient():
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    decoder = RibbedDecoder(RibbedGeometry.from_task(example()))
    p = torch.tensor(decoder.initial(0)[None], requires_grad=True)
    d = decoder.density(p, [0], beta=4, hard=False)
    core, wall, _ = decoder.masks(decoder.coordinates(p, [0]))
    assert d.sum().item() == pytest.approx(1024, abs=1e-8)
    assert not d[wall].any() and torch.equal(d[core], torch.ones_like(d[core]))
    w = torch.tensor(np.random.default_rng(55).normal(size=d.shape))
    gradient = torch.autograd.grad((d * w).sum(), p)[0]
    v = torch.zeros_like(p)
    v[0, 20] = 1
    eps = 1e-6

    def f(x):
        return (decoder.density(x, [0], beta=4, hard=False) * w).sum()

    fd = (f(p.detach() + eps * v) - f(p.detach() - eps * v)) / (2 * eps)
    assert gradient[0, 20].item() == pytest.approx(fd.item(), rel=1e-6, abs=1e-6)


def test_physical_coordinates_do_not_inherit_saturated_tanh_steps():
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder

    p = torch.tensor(RibbedDecoder.initial(1))
    p[:20] = 5
    q = RibbedDecoder.to_physical(p[None])[0].detach().requires_grad_(True)
    before = q.detach().clone()
    opt = torch.optim.Adam([q], lr=0.12)
    q[:20].sum().backward()
    opt.step()
    assert (before[:20] - q[:20]).mean().item() == pytest.approx(0.12, rel=1e-5)
    assert q.shape == (4125,)


def test_coincident_io_endpoint_is_a_constant_segment_not_nan():
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    bottom = np.array([[9.5, 13.0], [22.5, 13.0], [35.5, 13.0], [57.5, 13.0]])
    top = bottom.copy()
    top[:, 1] = 50.0
    decoder = RibbedDecoder(RibbedGeometry(bottom, top))
    q = (
        RibbedDecoder.to_physical(torch.tensor(decoder.initial(0)[None]))
        .detach()
        .requires_grad_(True)
    )
    with torch.no_grad():
        q[:, 25:29] = 2.0
    xy = decoder.coordinates(q, [0], physical=True)
    assert torch.isfinite(xy).all()
    d = decoder.density(q, [0], beta=4, hard=False, physical=True)
    gradient = torch.autograd.grad(d.square().sum(), q)[0]
    assert torch.isfinite(d).all() and torch.isfinite(gradient).all()
    assert d.sum().item() == pytest.approx(1024, abs=1e-8)


def test_true_mirrors_restore_the_same_final_binary_channels():
    from waveforge.lab.mcl.adaptive_manifold import reflected_task
    from waveforge.lab.mcl.locked_channels import canonical_frame
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    task = example()
    a = canonical_frame(task)
    b = canonical_frame(reflected_task(task))
    assert a.task.fingerprint() == b.task.fingerprint()
    decoder = RibbedDecoder(RibbedGeometry.from_task(a.task))
    d = decoder.binary(torch.tensor(decoder.initial(1)[None]), [1])[0].numpy()
    assert np.array_equal(a.to_original(d), b.to_original(d)[:, ::-1])


def test_standard_reference_starts_from_exact_uniform_without_task_information():
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.ribbed_evaluation import fit_uniform_start
    from waveforge.lab.mcl.stage5_decoder import binary

    p = fit_uniform_start()
    assert np.array_equal(binary(torch.tensor(p[None]))[0].numpy(), uniform_parallel())


def test_refinement_keeps_moving_when_every_binary_proposal_is_worse(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    import waveforge.lab.mcl.ribbed_evaluation as module
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    class FakeDecoder:
        def density(self, q, *args, **kwargs):
            return (10 + q[:, 0, None, None]).expand(-1, 64, 64)

    class Evaluator:
        problem = problem_from_task(example(), stage5_spec())
        decoder = FakeDecoder()
        output = tmp_path

        def __init__(self):
            self.seen = []

        def parameters(self, q, head):
            self.seen.append(q.copy())
            return dict(
                valid=True,
                violation=0.0,
                peak_c=50 + len(self.seen),
                binary_sha256=str(len(self.seen)),
            )

    def objective(d, level):
        return d.square().mean(), SimpleNamespace(scenario_peaks=torch.tensor([10.0]))

    monkeypatch.setattr(module, "JointObjective", lambda solver: objective)
    q = RibbedDecoder.to_physical(torch.tensor(RibbedDecoder.initial(1)[None]))[
        0
    ].numpy()
    evaluator = Evaluator()
    result = module.optimize_physical(evaluator, q, 1, kind="Adam", steps=25)
    assert result["objective_evaluations"] == 25 and len(evaluator.seen) == 26
    assert evaluator.seen[-1][0] < evaluator.seen[0][0] - 1
    assert (
        result["raw"]["peak_c"] == 51
    )  # original incumbent kept, trajectory not reset
    assert min(v["lr_branch"] for v in result["history"]) == pytest.approx(0.03)
    assert max(v["beta"] for v in result["history"]) == 6
