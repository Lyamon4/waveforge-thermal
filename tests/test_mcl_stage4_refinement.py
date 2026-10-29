import numpy as np
import pytest
import torch


def test_refinement_preparation_only_a_changes_saturated_start():
    from waveforge.lab.mcl.stage4_refinement import prepare

    p = torch.ones(4116, dtype=torch.float64) * 5
    original = p.clone()
    assert prepare(p, "A")[0] < 2
    assert torch.equal(prepare(p, "B"), p)
    assert torch.equal(prepare(p, "C"), p)
    assert torch.equal(original, p)


def test_refinement_schedules_and_smooth_c():
    from waveforge.lab.mcl.stage4_refinement import schedule

    assert schedule("A", 0).beta == 2
    assert schedule("B", 24).hard
    assert schedule("C", 0).branch_lr == 0.015
    assert schedule("C", 24).branch_lr == pytest.approx(0.003)
    assert schedule("C", 7).beta == 8 and schedule("C", 7).alpha == 500
    assert not schedule("C", 24).hard


def test_counter_distinguishes_multirhs_and_adjoint_calls():
    from waveforge.lab.mcl.stage4_refinement import count_solves

    rows = [
        dict(phase="forward", rhs=1),
        dict(phase="forward", rhs=3),
        dict(phase="adjoint", rhs=3),
        dict(phase="adjoint", rhs=1),
    ]
    assert count_solves(rows) == dict(
        forward_calls=2, adjoint_calls=2, forward_rhs=4, adjoint_rhs=4
    )


def test_nonlinear_nonconvergence_is_not_silently_accepted():
    from waveforge.lab.mcl.contracts import MCLProblem, stage12_spec, straight_channel
    from waveforge.lab.mcl.stage4_nonlinear import solve_nonlinear

    s = stage12_spec(nx=8, ny=8)
    p = MCLProblem(s, np.ones((8, 8), bool), np.ones((1, 8, 8)) * 10 / 64)
    result = solve_nonlinear(straight_channel(s), p, h_air=10, max_iterations=1)
    assert not result["converged"]
