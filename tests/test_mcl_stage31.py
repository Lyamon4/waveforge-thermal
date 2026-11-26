import importlib
import importlib.util

import pytest
import torch

from waveforge.lab.mcl.stage3 import anchor_parameters, binary, continuous


def module():
    name = "waveforge.lab.mcl.stage31"
    assert importlib.util.find_spec(name) is not None, "Stage31 not implemented"
    return importlib.import_module(name)


@pytest.mark.parametrize("hard", [False, True])
def test_ste_forward_exact_and_backward_only_auxiliary(hard):
    m = module()
    torch.set_num_threads(1)
    params = anchor_parameters().requires_grad_(True)
    used, auxiliary = m.density(params, beta=4.0, hard=hard)
    expected = binary(params).double() if hard else continuous(params, 4.0)
    assert torch.equal(used, expected)
    assert torch.max(torch.abs(used.sum((1, 2)) - 1024)) < 1e-8
    weight = torch.randn(used.shape, generator=torch.Generator().manual_seed(31))
    actual = torch.autograd.grad((used * weight).sum(), params, retain_graph=True)[0]
    wanted = torch.autograd.grad((auxiliary * weight).sum(), params)[0]
    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert actual[:, :16].norm() > 0
    assert actual[:, 16:20].norm() > 0
    assert actual[:, 20:].norm() > 0
    assert torch.isfinite(actual).all()


def test_auxiliary_projection_has_no_core_priority_and_correct_derivative():
    m = module()
    torch.set_num_threads(1)
    p = anchor_parameters()[:1]
    score, core, auxiliary_score = m.score_fields(p)
    assert torch.all(score[core] > auxiliary_score[core] + 4.9)
    auxiliary_score = auxiliary_score.detach().requires_grad_(True)
    projected = m.project_all(auxiliary_score, 2.0)
    assert ((2 * projected * (1 - projected))[core] > 1e-8).all()
    rng = torch.Generator().manual_seed(32)
    direction = torch.randn(p.shape[0], 64, 64, generator=rng, dtype=torch.float64)
    weight = torch.randn(direction.shape, generator=rng, dtype=torch.float64)
    grad = torch.autograd.grad((projected * weight).sum(), auxiliary_score)[0]
    epsilon = 1e-5
    difference = (
        (
            m.project_all(auxiliary_score + epsilon * direction, 2.0)
            - m.project_all(auxiliary_score - epsilon * direction, 2.0)
        )
        * weight
    ).sum() / (2 * epsilon)
    torch.testing.assert_close(
        (grad * direction).sum(), difference, rtol=1e-6, atol=1e-6
    )
    assert abs(grad.sum()) < 1e-10


def test_desaturation_bounds_branch_pixels_not_widths_and_leaves_source():
    m = module()
    p = torch.linspace(-6, 6, 4116, dtype=torch.float64)
    before = p.clone()
    result = m.desaturate(p)
    assert torch.equal(p, before)
    assert result[:16].tanh().abs().max() <= 0.85 + 1e-15
    assert result[20:].tanh().abs().max() <= 0.85 + 1e-15
    assert torch.equal(result[16:20], p[16:20])


@pytest.mark.parametrize(
    "index,level", [(0, 0), (8, 0), (9, 1), (16, 1), (17, 2), (24, 2)]
)
def test_refinement_phase_boundaries(index, level):
    phase = module().refinement_phase(index, 25)
    assert (phase.beta, phase.alpha, phase.branch_lr, phase.pixel_lr, phase.hard) == (
        (2.0, 150.0, 0.08, 0.04, False),
        (4.0, 300.0, 0.05, 0.03, False),
        (8.0, 500.0, 0.02, 0.015, True),
    )[level]


def test_control_and_training_phase_boundaries_and_cosine_endpoints():
    m = module()
    for index, hard in [
        (0, False),
        (199, False),
        (200, False),
        (399, False),
        (400, True),
        (599, True),
    ]:
        assert m.refinement_phase(index, 600).hard is hard
    assert [m.training_level(n) for n in (0, 266, 267, 532, 533, 799)] == [
        0,
        0,
        1,
        1,
        2,
        2,
    ]
    assert m.learning_rate(0) == pytest.approx(0.0005)
    assert m.learning_rate(799) == pytest.approx(0.0001)
    with pytest.raises(ValueError):
        m.refinement_phase(25, 25)


def test_logit_penalty_pushes_large_branch_and_pixel_values_toward_zero():
    p = torch.ones(2, 4, 4116, requires_grad=True)
    penalty = module().logit_penalty(p)
    assert penalty.item() == pytest.approx(2.0)
    penalty.backward()
    assert (p.grad[..., :16] > 0).all()
    assert (p.grad[..., 20:] > 0).all()
    assert torch.count_nonzero(p.grad[..., 16:20]) == 0
