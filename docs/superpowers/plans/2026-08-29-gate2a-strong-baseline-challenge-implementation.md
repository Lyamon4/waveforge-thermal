# Gate 2A Strong Branching Baseline Challenge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exhaustively evaluate the locked 41,055-member parametric branching-tree family and compare its independently verified winner with all three frozen WaveForge Gate 2A seeds.

**Architecture:** Keep deterministic geometry in `design/branching_baseline.py`, physics-evaluation and verdict rules in `verification/challenge.py`, and long-running search/artifact orchestration in `experiments/run_branching_challenge.py`. Every candidate is constructed without WaveForge pixels; one SciPy factorization is reused across the three source RHS at each fidelity, then the selected winner alone enters perturbation and morphology comparison.

**Tech Stack:** Python 3.11, NumPy/SciPy `float64`, pandas, matplotlib, tqdm, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-08-29-gate2a-strong-baseline-challenge-design.md`

## Global Constraints

- This is a prospective post-result challenge; never modify `artifacts/gate2_design/`.
- Candidate axes are exactly 17 × 21 × 23 × 5 = 41,055 combinations.
- Every design is strict binary with exactly 1,024 selected `64×64` cells.
- Score and ties follow the locked normalized-distance and lower-row-major rules.
- Frozen masks transfer by exact factor-2/factor-4 replication; no rerasterization or repair.
- Source maps are independently exact-area-overlap rasterized at every resolution.
- Primary physics is SciPy finite-volume `float64`; residual must be `≤1e-10`.
- Do not use WaveForge design pixel values for construction, pruning, bounds or tuning.
- Do not install `neuraloperator`; do not start Gate 2B, ML training or UI work.
- Any invalid registry, solve, artifact or metric produces `INVALID_RUN`, never `CHALLENGE_FAIL`.

---

### Task 1: Deterministic branching-tree geometry

**Files:**
- Create: `src/waveforge/design/branching_baseline.py`
- Create: `tests/test_branching_baseline.py`

**Interfaces:**
- Produces: `BranchingTreeParameters`, `candidate_axes()`, `iter_candidate_parameters()`, `segment_distance()`, `branching_score()`, `build_branching_tree()`.
- `build_branching_tree(parameters, grid=Grid2D(64,64)) -> BaselineDesign` returns a frozen strict-binary exact-budget map with stable identity.

- [ ] **Step 1: Write failing registry tests**

```python
def test_candidate_registry_has_exact_locked_cartesian_product() -> None:
    axes = candidate_axes()
    assert tuple(map(len, axes)) == (17, 21, 23, 5)
    parameters = tuple(iter_candidate_parameters())
    assert len(parameters) == 41055
    assert len({item.candidate_id for item in parameters}) == 41055
    assert parameters[0].as_tuple() == (0.30, 0.25, 0.10, 0.75)
    assert parameters[-1].as_tuple() == (0.70, 0.75, 0.65, 2.0)
```

- [ ] **Step 2: Run RED check**

Run: `pytest tests/test_branching_baseline.py -q`

Expected: collection failure because `waveforge.design.branching_baseline` does not exist.

- [ ] **Step 3: Implement parameter registry only**

Use integer-index formulas (`start + index * 0.025`) and immutable dataclasses. Format `candidate_id` from fixed three/two-decimal fields so lexical tie-breaking is stable.

- [ ] **Step 4: Run registry test GREEN**

Run: `pytest tests/test_branching_baseline.py -q`

Expected: registry test passes.

- [ ] **Step 5: Add failing geometry tests**

```python
def test_segment_distance_uses_clamped_finite_segment_projection() -> None:
    points = np.array([[0.5, 0.5], [2.0, 0.0], [-1.0, 0.0]])
    actual = segment_distance(points, (0.0, 0.0), (1.0, 0.0))
    np.testing.assert_allclose(actual, [0.5, 1.0, 1.0])


def test_tree_mask_is_strict_binary_exact_budget_and_repeatable() -> None:
    parameters = BranchingTreeParameters(0.5, 0.5, 0.3, 1.5)
    first = build_branching_tree(parameters)
    second = build_branching_tree(parameters)
    assert first.name == parameters.candidate_id
    assert np.array_equal(first.design, second.design)
    assert int(first.design.sum()) == 1024
    assert set(np.unique(first.design)) == {0.0, 1.0}
```

Add a tie fixture with constant score and assert selected flat indices are `0..1023`. Add a ratio fixture asserting that increasing `trunk_to_branch_width_ratio` cannot worsen normalized trunk score at any cell.

- [ ] **Step 6: Run geometry tests RED**

Run: `pytest tests/test_branching_baseline.py -q`

Expected: missing geometry functions fail.

- [ ] **Step 7: Implement minimal geometry**

Vectorize clamped segment distance over `[...,2]` points. Compute branch minimum, normalized trunk distance and `score=-minimum`. Reuse `stable_top_k_mask(values, count=1024)` for the locked lower-row-major tie rule. Include exact parameters and formula version in `parameter_hash`.

- [ ] **Step 8: Run geometry tests GREEN and lint**

Run: `pytest tests/test_branching_baseline.py -q`

Run: `ruff check src/waveforge/design/branching_baseline.py tests/test_branching_baseline.py`

- [ ] **Step 9: Commit**

```powershell
git add src/waveforge/design/branching_baseline.py tests/test_branching_baseline.py
git commit -m "feat: add locked parametric branching baseline"
```

### Task 2: Reusable-factorization SciPy challenge evaluator

**Files:**
- Create: `src/waveforge/verification/challenge.py`
- Create: `tests/test_branching_challenge.py`

**Interfaces:**
- Consumes: frozen `64×64` binary maps and Gate 1 `assemble_steady_system`, `factorize_system`, `solve_factorized`.
- Produces: `ChallengeEvaluation`, `evaluate_frozen_binary_design()`, `ChallengeStatus`, `ChallengeSeedComparison`, `classify_challenge()`.

- [ ] **Step 1: Write failing evaluator agreement test**

```python
@pytest.mark.parametrize("resolution", [64, 128])
def test_reusable_factorization_evaluator_matches_public_verifier(
    resolution: int,
) -> None:
    design = build_branching_tree(BranchingTreeParameters(0.5, 0.5, 0.3, 1.0)).design
    actual = evaluate_frozen_binary_design("tree", design, resolution=resolution)
    fidelity = "low_64" if resolution == 64 else "reference_128"
    expected = verify_candidate("tree", design, fidelity=fidelity)
    np.testing.assert_allclose(
        actual.scenario_peaks,
        [r.peak_temperature for r in expected.scenario_records],
        rtol=0.0,
        atol=1e-12,
    )
    assert actual.worst_peak == pytest.approx(expected.worst_peak, abs=1e-12)
    assert actual.maximum_residual <= 1e-10
```

Also monkeypatch `factorize_system` with a counting wrapper and assert exactly one factorization for three scenarios.

- [ ] **Step 2: Run evaluator tests RED**

Run: `pytest tests/test_branching_challenge.py -q`

Expected: missing challenge evaluator fails.

- [ ] **Step 3: Implement evaluator**

Transfer with nested `np.repeat`, build three independently rasterized source maps, assemble first system, factorize once, and solve three RHS by `dataclasses.replace(system, source_rhs=..., rhs=source_rhs+dirichlet_rhs)`. Validate strict binary input, shape, exact fraction `0.25`, finite positive peaks and residuals.

- [ ] **Step 4: Run evaluator tests GREEN**

Run: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 5: Write failing verdict-table tests**

Cover:

```python
assert (
    classify_challenge(two_strong_one_comparable).status
    is ChallengeStatus.STRONG_CHALLENGE_PASS
)
assert (
    classify_challenge(three_small_positive).status
    is ChallengeStatus.CHALLENGE_COMPARABLE
)
assert classify_challenge(two_negative).status is ChallengeStatus.CHALLENGE_FAIL
assert classify_challenge(nonfinite).status is ChallengeStatus.INVALID_RUN
```

Each strong seed must satisfy both nominal `>=0.05` and robustness `>=23/28`.

- [ ] **Step 6: Run verdict tests RED, implement, then GREEN**

Run RED: `pytest tests/test_branching_challenge.py -q`

Implement literal precedence `INVALID_RUN` → `CHALLENGE_FAIL` → `STRONG_CHALLENGE_PASS` → `CHALLENGE_COMPARABLE`.

Run GREEN: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 7: Commit**

```powershell
git add src/waveforge/verification/challenge.py tests/test_branching_challenge.py
git commit -m "feat: add independent branching challenge evaluator"
```

### Task 3: Exhaustive search funnel and machine artifacts

**Files:**
- Create: `src/waveforge/experiments/run_branching_challenge.py`
- Modify: `tests/test_branching_challenge.py`

**Interfaces:**
- Consumes: candidate iterator/geometry and challenge evaluator.
- Produces: `run_search(output_dir, parameters=None)`, `run_comparison(output_dir, search_result)`, and CLI `python -m waveforge.experiments.run_branching_challenge --output artifacts/gate2a_challenge`.

- [ ] **Step 1: Write failing reduced-funnel orchestration test**

Inject 24 deterministic parameters and a fake evaluator keyed only by candidate ID. Assert:

- every input candidate receives one `64` evaluation;
- exactly 20 receive `128` evaluation;
- exactly 5 receive `256` evaluation;
- sorting uses `(unrounded worst_peak, candidate_id)`;
- final winner comes from `256`, not `64`;
- output row counts are `24`, `20`, `5`.

- [ ] **Step 2: Run orchestration test RED**

Run: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 3: Implement search funnel**

Keep a lazy cache of branch-distance fields keyed only by `(x_junction,y_junction)`. Within each `(x_sink,x_junction,y_junction)` group compute trunk distances once and emit all five ratios. Store every row with parameters, design hash, fraction, three peaks, worst/average peak, maximum residual, evaluation time and cumulative rank.

Before returning, assert exact production counts when `parameters is None`: `41055`, `20`, `5`. Write CSV atomically through a sibling `.tmp` path then rename; `.tmp` is never committed.

- [ ] **Step 4: Run orchestration test GREEN**

Run: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 5: Add failing comparison/artifact test**

With a synthetic winner and tiny raw CSV fixtures, assert `waveforge_vs_tree.csv` has three seeds, `challenge_robustness.csv` has `84` rows, all improvements use tree in the denominator, and candidate registry records `post_result_challenge=true`, spec hash and exact axes/count.

- [ ] **Step 6: Run RED, implement comparison/artifact serialization, then GREEN**

Run RED: `pytest tests/test_branching_challenge.py -q`

Implement raw locked WaveForge metric loading, tree perturbation solves, separate morphology rows, verdict serialization and fail-closed integrity checks.

Run GREEN: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 7: Commit**

```powershell
git add src/waveforge/experiments/run_branching_challenge.py tests/test_branching_challenge.py
git commit -m "exp: orchestrate strong branching baseline challenge"
```

### Task 4: Scientific figures and artifact integrity

**Files:**
- Modify: `src/waveforge/experiments/run_branching_challenge.py`
- Modify: `tests/test_branching_challenge.py`

**Interfaces:**
- Produces: `best_tree_design.png`, `best_tree_temperature_maps.png`, and complete `challenge_verdict.json` artifact hashes.

- [ ] **Step 1: Write failing plotting-integrity test**

Run a three-candidate fixture into a temporary output directory. Capture winner metrics before/after plotting, assert bitwise equality, non-empty PNG signatures, and that the verdict hash manifest covers every final challenge artifact except itself.

- [ ] **Step 2: Run RED**

Run: `pytest tests/test_branching_challenge.py -q`

- [ ] **Step 3: Implement figures**

Render the strict `64×64` winner with source/sink/junction/segment overlays. Re-solve and render three `256×256` nominal temperature fields using a shared color scale. Plotting receives copies and never mutates metric arrays.

- [ ] **Step 4: Run GREEN and focused lint**

Run: `pytest tests/test_branching_challenge.py tests/test_branching_baseline.py -q`

Run: `ruff check src/waveforge/design/branching_baseline.py src/waveforge/verification/challenge.py src/waveforge/experiments/run_branching_challenge.py tests/test_branching_baseline.py tests/test_branching_challenge.py`

- [ ] **Step 5: Commit**

```powershell
git add src/waveforge/experiments/run_branching_challenge.py tests/test_branching_challenge.py
git commit -m "feat: add branching challenge scientific artifacts"
```

### Task 5: Production exhaustive challenge and interpretation

**Files:**
- Modify: `docs/lab_journal.md`
- Create through registered CLI: all remaining `artifacts/gate2a_challenge/*` outputs.

**Interfaces:**
- Consumes: locked spec commit and implementation.
- Produces: final Stage B data, verdict and scientific interpretation.

- [ ] **Step 1: Run pre-production verification**

Run: `pytest -q`

Run: `ruff check .`

Run: `ruff format --check src tests`

Expected: tests/lint/production-code formatting PASS. Do not gate on the separately documented old Markdown plan format finding.

- [ ] **Step 2: Record implementation SHA and run exhaustive search**

Run:

```powershell
python -m waveforge.experiments.run_branching_challenge `
  --output artifacts/gate2a_challenge
```

Require machine exit `0`, registry count `41055`, finalist counts `20/5`, finite residual-qualified rows and a valid challenge status.

- [ ] **Step 3: Independently inspect raw outputs**

Recalculate winner selection from CSV with a separate Python command. Recalculate the three nominal and 84 perturbation improvements without importing `classify_challenge`. Confirm file hashes and exact material fraction.

- [ ] **Step 4: Write lab-journal interpretation**

Record winner parameters, `256×256 Tmax`, all three WaveForge improvements, robustness counts, morphology diagnostics and one of the three valid challenge verdicts. Explicitly answer whether the original 50% effect was comparator-driven, whether geometry is qualitatively different, and whether contribution is scientific inverse design or primarily automation/software.

- [ ] **Step 5: Run final verification**

Run: `pytest -q`

Run: `ruff check .`

Run: `ruff format --check src tests`

Validate every JSON/CSV and required PNG. Confirm `git diff -- artifacts/gate2_design` is empty relative to `v0.3-gate2a-inverse-design-validated`.

- [ ] **Step 6: Commit**

```powershell
git add artifacts/gate2a_challenge docs/lab_journal.md
git commit -m "exp: challenge WaveForge with optimized branching tree"
```

Stop after reporting Stage B interpretation before any conditional Stage C dataset generation.
