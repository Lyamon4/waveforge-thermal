"""The online route must preserve geometry without invoking physical solvers."""

import numpy as np
import pytest
import torch


def test_compiled_channel_fill_is_bit_exact_and_keeps_ribs():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.fast_channels import compiled_fill
    from waveforge.lab.mcl.ribbed_channels import (
        RibbedDecoder,
        RibbedGeometry,
        separated_fill,
    )

    torch.set_num_threads(1)
    rng = np.random.default_rng(2026103011)
    for seed in (34, 91):
        decoder = RibbedDecoder(
            RibbedGeometry.from_task(sample_epyc_task(np.random.default_rng(seed)))
        )
        q = np.stack([decoder.initial(h) for h in range(4)])
        q += rng.normal(0, 0.2, q.shape)
        score, core, wall = decoder.fields(torch.tensor(q), range(4))
        for s, c, w in zip(score.numpy(), core.numpy(), wall.numpy(), strict=True):
            expected = separated_fill(s, c, w)
            actual = compiled_fill(s, c, w)
            np.testing.assert_array_equal(actual, expected)
            assert actual.sum() == 1024 and not (actual & w).any()


def test_compiled_skeleton_keeps_every_reserved_cell_and_rib():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.fast_channels import FastRibbedDecoder
    from waveforge.lab.mcl.ribbed_channels import RibbedDecoder, RibbedGeometry

    rng = np.random.default_rng(83)
    for seed in range(4):
        decoder = RibbedDecoder(
            RibbedGeometry.from_task(sample_epyc_task(np.random.default_rng(seed)))
        )
        raw = rng.normal(0, 1.5, (8, 4125))
        coordinates = decoder.coordinates(torch.tensor(raw), list(range(4)) * 2)
        expected = decoder.masks(coordinates)
        actual = FastRibbedDecoder.masks(coordinates)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_raw_input_contains_only_geometry_power_and_coordinates():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.solver_free import raw_features

    task = sample_epyc_task(np.random.default_rng(22))
    x = raw_features(task)
    assert x.shape == (7, 64, 64) and x.dtype == np.float32
    np.testing.assert_allclose(x[:3].sum((1, 2)), 360, rtol=1e-6)
    assert np.all(x[:3, x[3] == 0] == 0)
    assert x[4].sum() == 32
    np.testing.assert_array_equal(x[5, :, 0], np.arange(64, dtype=np.float32) / 63)


def test_independent_solver_backend_override_is_local_and_restored(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import scipy.sparse as sp

    from waveforge.lab.mcl.factorization import Factorization, factorization_backend

    monkeypatch.setenv("WAVEFORGE_SPARSE_BACKEND", "invalid")

    def independent():
        with factorization_backend("scipy"):
            factor = Factorization(sp.eye(3, format="csc"))
            np.testing.assert_array_equal(factor.solve(np.ones(3)), np.ones(3))
            assert factor.backend == "scipy_lu"

    with ThreadPoolExecutor(2) as pool:
        for future in (pool.submit(independent), pool.submit(independent)):
            future.result()
    with pytest.raises(ValueError, match="unknown sparse"):
        Factorization(sp.eye(3, format="csc"))


def test_selector_learns_good_ties_and_rejects_invalid_candidates():
    from waveforge.lab.mcl.solver_free import ranking_loss

    temperature = torch.tensor([[65.0, 65.0, 67.0, 60.0]])
    valid = torch.tensor([[True, True, True, False]])
    good = torch.tensor([[3.0, 3.0, -2.0, -3.0]], requires_grad=True)
    bad = torch.tensor([[-3.0, -3.0, 2.0, 4.0]], requires_grad=True)
    assert ranking_loss(good, temperature, valid) < ranking_loss(
        bad, temperature, valid
    )
    gradient = torch.autograd.grad(ranking_loss(bad, temperature, valid), bad)[0]
    assert torch.isfinite(gradient).all() and gradient[0, 3] > 0


def test_online_request_decodes_one_candidate_and_never_solves_physics(monkeypatch):
    import waveforge.lab.mcl.solver_free as online
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.amortized_policy import CandidatePortfolio
    from waveforge.lab.mcl.solver_free import RawPortfolioPolicy, SolverFreeSession
    from waveforge.lab.mcl.stage5_physics import Stage5Solver

    def forbidden(*args, **kwargs):
        raise AssertionError("a physical solve entered online inference")

    monkeypatch.setattr(Stage5Solver, "__call__", forbidden)
    model = RawPortfolioPolicy(CandidatePortfolio(tuple(range(8)))).eval()
    model.selector = torch.nn.Linear(128 + 29 + 32 + 9, 6)
    with torch.no_grad():
        model.selector.weight.zero_()
        model.selector.bias.zero_()
        model.selector.weight[0, -9] = 1
    session = SolverFreeSession(model, device="cpu")
    actual = online.decode_one
    calls = []

    def count(*args):
        calls.append(args[2])
        return actual(*args)

    monkeypatch.setattr(online, "decode_one", count)
    result = session.infer(sample_epyc_task(np.random.default_rng(43)))
    assert result["binary"].sum() == 1024 and len(calls) == 1
    assert result["online_physical_solves"] == 0
    assert result["physics_checked"] is False
    assert result["proposal_count"] == 8
    assert all(not p.requires_grad for p in model.teacher.parameters())


def test_raw_bridge_keeps_original_proposal_weights_frozen():
    from waveforge.lab.mcl.amortized_policy import CandidatePortfolio
    from waveforge.lab.mcl.solver_free import RawPortfolioPolicy

    torch.set_num_threads(1)
    model = RawPortfolioPolicy(CandidatePortfolio(tuple(range(8))))
    model.train().requires_grad_(True)
    before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    raw = torch.randn(1, 7, 64, 64)
    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=1e-4)
    prediction = model(raw)
    loss = prediction["parameters"].square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.adapter.parameters())
    assert all(p.grad is None for p in model.teacher.parameters())
    optimizer.step()
    for key, value in model.teacher.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_quality_report_counts_invalid_outputs_and_uses_temperature_rise():
    from waveforge.lab.mcl.solver_free_training import quality_summary

    summary = quality_summary(
        np.array([65.0, 69.0]), np.array([True, False]), np.array([65.0, 65.0])
    )
    assert summary["invalid_count"] == 1
    assert summary["median_gap_percent"] == 5
    assert not summary["near_quality_target_passed"]


def test_compiled_tree_prediction_preserves_values_and_missing_branches():
    sklearn = pytest.importorskip("sklearn.ensemble")
    from waveforge.lab.mcl.fast_trees import pack_predictor, predict_packed

    rng = np.random.default_rng(19)
    x = rng.normal(size=(128, 7))
    x[::7, 2] = np.nan
    y = np.nan_to_num(x[:, 2]) + x[:, 0] ** 2
    estimator = sklearn.HistGradientBoostingRegressor(
        max_iter=20, max_leaf_nodes=7, min_samples_leaf=5
    ).fit(x, y)
    boundary = estimator._predictors[0][0].nodes["num_threshold"][0]
    feature = estimator._predictors[0][0].nodes["feature_idx"][0]
    x[0, feature] = boundary
    np.testing.assert_array_equal(
        predict_packed(pack_predictor(estimator), x), estimator.predict(x)
    )


def test_raster_scorer_is_candidate_order_equivariant():
    from waveforge.lab.mcl.solver_free_raster import RasterCritic

    torch.set_num_threads(1)
    model = RasterCritic().eval()
    raw = torch.rand(2, 7, 64, 64)
    designs = torch.rand(2, 9, 64, 64) > 0.7
    order = torch.tensor([8, 3, 2, 6, 5, 4, 7, 1, 0])
    with torch.inference_mode():
        first = model(raw, designs)
        second = model(raw, designs[:, order])
    torch.testing.assert_close(
        second["logits"], first["logits"][:, order], rtol=1e-5, atol=1e-5
    )


def test_offline_linear_responses_reproduce_full_physics_without_new_equations():
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.contracts import problem_from_task
    from waveforge.lab.mcl.dual_profile import solve_dual_profile
    from waveforge.lab.mcl.solver_free_augmentation import response_labels
    from waveforge.lab.mcl.stage5_evaluation import PUMP_W
    from waveforge.lab.mcl.stage5_physics import stage5_spec

    task = sample_epyc_task(np.random.default_rng(18))
    d = uniform_parallel()
    metrics, valid = response_labels(
        task, d, np.asarray(task.powers_w)[None], grids=(64,)
    )
    nom, pow = solve_dual_profile(
        d, problem_from_task(task, stage5_spec()), pump_power_w=PUMP_W
    )
    np.testing.assert_allclose(
        metrics[0, :2],
        [nom.scenario_peaks.max() / 50, pow.scenario_peaks.max() / 50],
        atol=1e-10,
        rtol=0,
    )
    assert valid[0]


def test_hydraulic_prediction_does_not_depend_on_workload():
    from waveforge.lab.mcl.solver_free_raster import RasterCritic

    model = RasterCritic(hydraulic_branch=True).eval()
    d = torch.rand(1, 9, 64, 64) > 0.7
    first = torch.rand(1, 7, 64, 64)
    second = first.clone()
    second[:, :3] *= 2
    with torch.inference_mode():
        a = model(first, d)["metrics"][:, :, 2]
        b = model(second, d)["metrics"][:, :, 2]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_predicted_limits_are_not_traded_for_a_lower_temperature():
    from waveforge.lab.mcl.solver_free_raster import choose_predictions

    metrics = torch.tensor(
        [[[0.7, 0.7, 1.01, 0.8], [0.8, 0.8, 0.8, 0.8]] + [[1.0, 1.0, 0.8, 0.8]] * 7]
    )
    assert choose_predictions(dict(metrics=metrics), re_margin=0).item() == 1


def test_flow_geometry_features_detect_a_neck_at_the_same_channel_budget():
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.fast_channels import flow_geometry_features

    original = uniform_parallel()
    neck = original.copy()
    neck[20, 8] = False
    neck[21, 9] = True
    assert neck.sum() == original.sum()
    a = flow_geometry_features(original[None])
    b = flow_geometry_features(neck[None])
    assert a.shape == (1, 89) and np.isfinite(a).all()
    assert b[0, 0] < a[0, 0]


@pytest.mark.parametrize(
    "version,size", [("binary-geometry-519-v1", 519), ("binary-branch-690-v2", 690)]
)
def test_structured_selector_features_are_y_invariant_and_candidate_equivariant(
    version, size
):
    from waveforge.lab.epyc_tasks import sample_epyc_task
    from waveforge.lab.mcl.adaptive_manifold import reflected_task
    from waveforge.lab.mcl.baselines import uniform_parallel
    from waveforge.lab.mcl.solver_free_tabular import candidate_features

    task = sample_epyc_task(np.random.default_rng(29))
    d = np.stack((uniform_parallel(), uniform_parallel()[:, ::-1]))
    a, flow = candidate_features(task, d, version=version)
    b, other = candidate_features(reflected_task(task), d[:, :, ::-1], version=version)
    assert a.shape == (2, size) and np.isfinite(a).all()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(flow, other)
    reverse, _ = candidate_features(task, d[::-1], version=version)
    np.testing.assert_array_equal(reverse, a[::-1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_captured_generator_updates_inputs_and_keeps_proposals():
    from waveforge.lab.mcl.amortized_policy import CandidatePortfolio
    from waveforge.lab.mcl.solver_free import RawPortfolioPolicy
    from waveforge.lab.mcl.solver_free_graph import GraphedGenerator
    from waveforge.lab.mcl.solver_free_tabular import GeneratorOnlyPolicy

    torch.manual_seed(53)
    model = (
        GeneratorOnlyPolicy(RawPortfolioPolicy(CandidatePortfolio(tuple(range(8)))))
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    with torch.no_grad():
        for network in (model.teacher.source, model.teacher.student):
            network.geometry.weight.normal_(std=0.001)
            network.pixels.weight.normal_(std=0.001)
    captured = GraphedGenerator(model)
    for value in (0.1, 0.4):
        raw = np.full((7, 64, 64), value, dtype=np.float32)
        with torch.inference_mode():
            expected = (
                model(torch.from_numpy(raw[None]).cuda())["parameters"][0].cpu().numpy()
            )
        actual = captured.generate(raw)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
