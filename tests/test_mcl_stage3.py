import numpy as np
import pytest
import torch

from waveforge.lab.mcl.stage3 import (
    VascularUNet,
    _CoreProjection,
    anchor_parameters,
    binary,
    continuous,
    fields,
    halo_residual,
)
from waveforge.lab.mcl.vascular import connectivity
from waveforge.lab.mcl.vascular_stage12 import limited_waypoints, smooth_scores


def test_anchors_and_extreme_outputs_respect_final_segment():
    raw = anchor_parameters()[:, :16].reshape(4, 4, 4)
    for values in (raw, raw + 100, raw - 100):
        nodes = limited_waypoints(values)
        ends = torch.full_like(nodes[..., :1], 31.5)
        complete = torch.cat((ends, nodes, ends), -1)
        assert (complete.diff(dim=-1).abs() <= 16 + 1e-12).all()


def test_bounded_halo_cannot_raise_far_cells():
    distance = torch.tensor([0.0, 0.75, 1.45, 1.6, 3.0], dtype=torch.float64)
    for u in (-1e6, 0, 1e6):
        result = halo_residual(torch.full_like(distance, u), distance)
        assert (result[3:] < 0).all()


def test_wrapper_is_exactly_compatible_with_frozen_decoder():
    params = anchor_parameters()
    score, core, waypoints, residual = fields(params)
    expected, dist = smooth_scores(
        waypoints, 1 + 3 * params[:, 16:20].sigmoid(), residual
    )
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    assert torch.equal(core, dist.amin(1) <= 0.75)
    maps = binary(params).numpy()
    for item in maps:
        assert item.sum() == 1024
        assert connectivity(item)["all_cells_port_connected"]


@pytest.mark.parametrize("beta", [2.0, 4.0, 8.0])
def test_core_projection_budget_and_fixed_core_finite_difference(beta):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(11)
    score = torch.randn(
        1, 64, 64, generator=generator, dtype=torch.float64, requires_grad=True
    )
    core = torch.zeros_like(score, dtype=torch.bool)
    core[:, :8] = True
    density = _CoreProjection.apply(score, core, beta)
    assert abs(density.sum().item() - 1024) < 1e-8
    assert (density[core] == 1).all()
    weight = torch.randn(score.shape, generator=generator, dtype=score.dtype)
    direction = torch.randn(score.shape, generator=generator, dtype=score.dtype)
    analytic = (
        torch.autograd.grad((density * weight).sum(), score)[0] * direction
    ).sum()
    eps = 1e-5
    plus = _CoreProjection.apply(score + eps * direction, core, beta)
    minus = _CoreProjection.apply(score - eps * direction, core, beta)
    finite = ((plus - minus) * weight).sum() / (2 * eps)
    torch.testing.assert_close(analytic, finite, atol=2e-6, rtol=2e-6)


def test_impossible_core_rejected_and_exact_core_supported():
    score = torch.zeros(1, 64, 64, dtype=torch.float64, requires_grad=True)
    core = torch.zeros_like(score, dtype=torch.bool)
    core[:, :16] = True
    result = _CoreProjection.apply(score, core, 4.0)
    assert torch.equal(result, core.double())
    core[:, 16, 0] = True
    with pytest.raises(ValueError, match="budget"):
        _CoreProjection.apply(score, core, 4.0)


def test_unet_four_heads_have_finite_end_to_end_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = VascularUNet()
    outputs = model(torch.randn(2, 13, 64, 64))
    assert outputs.shape == (2, 4, 4116)
    density = continuous(outputs.flatten(0, 1).double())
    weight = torch.linspace(0, 1, 4096).reshape(64, 64)
    (density * weight).mean().backward()
    assert np.isfinite(density.detach().numpy()).all()
    assert model.geometry.weight.grad.norm() > 0
    assert model.pixels.weight.grad.norm() > 0
