"""Acceleration must preserve arithmetic, admission and selected designs."""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.mark.parametrize("columns", [1, 3])
def test_compiled_compensation_is_bit_identical_to_numpy(columns):
    pytest.importorskip("numba")
    from waveforge.lab.mcl.compensated import CompensatedCSR

    rng = np.random.default_rng(20261030)
    a = rng.normal(size=(19, 23)) * 10.0 ** rng.integers(-8, 9, size=(19, 23))
    a[rng.random(a.shape) < 0.75] = 0
    x = rng.normal(size=(23, columns))
    b = rng.normal(size=(19, columns))
    matrix = sp.csr_matrix(a)
    expected = CompensatedCSR(matrix, backend="numpy").residual(x, b)
    actual = CompensatedCSR(matrix, backend="numba").residual(x, b)
    np.testing.assert_array_equal(actual, expected)


def test_compiled_path_keeps_cancelled_unit_and_complex_step():
    pytest.importorskip("numba")
    from waveforge.lab.mcl.compensated import CompensatedCSR

    a = sp.csr_matrix([[1e16, 1.0, -1e16]])
    x = np.ones(3)
    b = np.zeros(1)
    np.testing.assert_array_equal(
        CompensatedCSR(a, backend="numba").residual(x, b), [-1.0]
    )
    xc = x * (1 + 1e-20j)
    expected = CompensatedCSR(a, backend="numpy").residual(xc, b.astype(complex))
    actual = CompensatedCSR(a, backend="auto").residual(xc, b.astype(complex))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("backend", ["unknown", ""])
def test_compensation_backend_rejects_unknown_name(backend):
    from waveforge.lab.mcl.compensated import CompensatedCSR

    with pytest.raises(ValueError, match="backend"):
        CompensatedCSR(sp.eye(3, format="csr"), backend=backend)


@pytest.mark.parametrize("factor", [1, 2])
def test_dual_regime_forward_matches_independent_solves(factor):
    import json
    from pathlib import Path

    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.dual_profile import solve_dual_profile
    from waveforge.lab.mcl.high_fidelity_eval import refine_problem
    from waveforge.lab.mcl.stage3_training import task_from_dict
    from waveforge.lab.mcl.stage5_decoder import binary, initial_parameters
    from waveforge.lab.mcl.stage5_evaluation import PUMP_W
    from waveforge.lab.mcl.stage5_physics import stage5_spec
    from waveforge.lab.mcl.stage5_preflight import solve_profile

    task = task_from_dict(
        json.loads(
            Path("results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )["tasks"][0]
    )
    problem = problem_from_task(task, stage5_spec())
    import torch

    d = binary(torch.tensor(initial_parameters()[None]))[0].numpy()
    problem, d = refine_problem(problem, d, factor)
    nominal = solve_profile(d, problem, k_tim=40.0)
    dp = np.sqrt(PUMP_W * problem.spec.pressure_drop_pa / (nominal.flow_ml_s * 1e-6))
    powered = solve_profile(d, problem, k_tim=40.0, pressure_drop_pa=dp)
    results = solve_dual_profile(d, problem, pump_power_w=PUMP_W)
    for actual, expected in zip(results, (nominal, powered), strict=True):
        np.testing.assert_allclose(
            actual.temperature, expected.temperature, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(
            actual.pressure, expected.pressure, rtol=1e-12, atol=1e-8
        )
        assert abs(actual.flow_ml_s - expected.flow_ml_s) < 1e-9
        for key in (
            "pressure_residual",
            "compensated_pressure_residual",
            "thermal_residual",
            "compensated_thermal_residual",
        ):
            assert np.max(actual.diagnostics[key]) < 1e-12
        assert actual.diagnostics["mass_relative_error"] < 1e-10
        assert max(actual.diagnostics["energy_relative_error"]) < 1e-10


def test_dual_regime_factorizes_pressure_once(monkeypatch):
    import json
    from pathlib import Path

    from waveforge.lab.mcl import high_fidelity_eval
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.dual_profile import solve_dual_profile
    from waveforge.lab.mcl.stage3_training import task_from_dict
    from waveforge.lab.mcl.stage5_evaluation import PUMP_W
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    task = task_from_dict(
        json.loads(
            Path("results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )["tasks"][0]
    )
    problem = problem_from_task(task, stage5_spec())
    original = high_fidelity_eval._factor
    pressure_calls = []

    def count(matrix):
        if matrix.shape[0] == 4096:
            pressure_calls.append(1)
        return original(matrix)

    monkeypatch.setattr(high_fidelity_eval, "_factor", count)
    solve_dual_profile(uniform_parallel(), problem, pump_power_w=PUMP_W)
    assert len(pressure_calls) == 1


def test_pressure_scaling_keeps_independent_fallback_on_failed_residual(monkeypatch):
    import json
    from pathlib import Path

    from waveforge.lab.mcl import dual_profile
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.stage3_training import task_from_dict
    from waveforge.lab.mcl.stage5_evaluation import PUMP_W
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    task = task_from_dict(
        json.loads(
            Path("results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )["tasks"][0]
    )
    problem = problem_from_task(task, stage5_spec())
    original_flow = dual_profile.solve_flow
    original_profile = dual_profile.solve_profile
    calls = []

    def unqualified_scaled_flow(*args, **kwargs):
        flow, matrix, rhs, factor = original_flow(*args, **kwargs)
        factor.relative_residual = lambda *a, **kw: 2e-12
        return flow, matrix, rhs, factor

    def independent(*args, **kwargs):
        result = original_profile(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(dual_profile, "solve_flow", unqualified_scaled_flow)
    monkeypatch.setattr(dual_profile, "solve_profile", independent)
    _, powered = dual_profile.solve_dual_profile(
        uniform_parallel(), problem, pump_power_w=PUMP_W
    )
    assert len(calls) == 1 and powered is calls[0]
    assert powered.diagnostics["compensated_pressure_residual"] < 1e-12


def test_fast_candidate_reuses_flow_and_matches_all_admission_values(tmp_path):
    import json
    from pathlib import Path

    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.ribbed_evaluation import RibbedEvaluator
    from waveforge.lab.mcl.stage3_training import task_from_dict

    task = task_from_dict(
        json.loads(
            Path("results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )["tasks"][0]
    )
    original = RibbedEvaluator(
        task, tmp_path / "reference", forward_backend="reference"
    )
    fast = RibbedEvaluator(task, tmp_path / "fast", forward_backend="fast")
    a = original.evaluate(uniform_parallel())

    def repeated_probe(_):
        raise AssertionError("an already solved pressure field was recomputed")

    fast.probe = repeated_probe
    b = fast.evaluate(uniform_parallel())
    assert b["valid"] == a["valid"] and b["gate"] == a["gate"]
    assert abs(a["search_objective"] - b["search_objective"]) < 1e-9
    for grid in ("64", "128"):
        for mode in ("nominal", "power"):
            x, y = a["modes"][grid][mode], b["modes"][grid][mode]
            assert abs(x["silicon_peak_c"] - y["silicon_peak_c"]) < 1e-9
            assert (
                abs(
                    x["diagnostics"]["max_local_reynolds_estimate"]
                    - y["diagnostics"]["max_local_reynolds_estimate"]
                )
                < 1e-8
            )


def test_threaded_independent_trajectories_preserve_parameters_and_designs(tmp_path):
    import json
    from pathlib import Path

    import torch

    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder
    from waveforge.lab.mcl.ribbed_evaluation import (
        RibbedEvaluator,
        refine_independent_starts,
    )
    from waveforge.lab.mcl.stage3_training import task_from_dict, worker_init

    worker_init()
    task = task_from_dict(
        json.loads(
            Path("results/stage55_20261029/inputs/evaluation_tasks.json").read_text()
        )["tasks"][0]
    )
    starts = [
        dict(
            parameters=RibbedDecoder.to_physical(
                torch.tensor(RibbedDecoder.initial(h)[None])
            )[0].numpy(),
            head=h,
        )
        for h in range(2)
    ]
    evaluators = [RibbedEvaluator(task, tmp_path / str(n)) for n in (1, 2)]
    results = []
    for workers, evaluator in zip((1, 2), evaluators, strict=True):
        evaluator.evaluate(uniform_parallel())
        results.append(
            refine_independent_starts(evaluator, starts, steps=2, workers=workers)
        )
    assert evaluators[0].cache.keys() == evaluators[1].cache.keys()
    for expected, actual in zip(*results, strict=True):
        np.testing.assert_array_equal(actual["parameters"], expected["parameters"])
        assert actual["raw"] == expected["raw"]
        assert actual["objective_evaluations"] == expected["objective_evaluations"] == 2
